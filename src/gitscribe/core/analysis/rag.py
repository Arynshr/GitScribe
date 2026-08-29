"""
Query -> embed -> vector search -> graph-expand -> assembled context.
Replaces/augments today's branch-prefix PR retrieval with code-aware
context.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from gitscribe.core.failure_router import classify_failure
from gitscribe.core.indexer.index_store import SearchResult, _get_connection, blast_radius, search

_TOKENS_PER_CHAR = 0.25 

def _estimate_tokens(text: str) -> int:
    return int(len(text) * _TOKENS_PER_CHAR)

class ContextSnippet(BaseModel):
    symbol_id: int
    name: str
    file: str
    lineno: int
    relation: str

class RAGContext(BaseModel):
    query: str
    snippets: list[ContextSnippet]

    def as_prompt_block(self) -> str:
        """Formats retrieved context for injection into generator.py's
        prompt template — plain text, no framework coupling.
        """
        lines = [f"# Relevant code context for: {self.query}"]
        for s in self.snippets:
            lines.append(f"- [{s.relation}] {s.name} ({s.file}:{s.lineno})")
        return "\n".join(lines)
    
class ReviewFindingLLM(BaseModel):
    severity: str = Field(description="one of: info, warning, error")
    rule_or_reason: str = Field(description="short label for what triggered this finding")
    message: str = Field(description="the finding itself, specific and actionable")
    line_start: int | None = None
    line_end: int | None = None


class ReviewFindingsLLM(BaseModel):
    findings: list[ReviewFindingLLM] = Field(default_factory=list)


_REVIEW_PARSER = PydanticOutputParser(pydantic_object=ReviewFindingsLLM)

_REVIEW_PROMPT = """You are reviewing a code change for correctness, risk, and \
maintainability issues a linter would miss.

Diff hunk (never omit or summarize — review this exactly as given):
{diff_hunk}

Direct callers of the changed code:
{callers_block}

Direct callees of the changed code:
{callees_block}

Same-file siblings (may be omitted below due to context budget):
{siblings_block}

Respond with findings only where you have a concrete, specific concern.
{format_instructions}
"""


def retrieve(query: str, cfg: dict, top_k: int = 5, expand_depth: int = 1) -> RAGContext:
    matches: list[SearchResult] = search(query, cfg, top_k=top_k)

    snippets: list[ContextSnippet] = []
    seen_ids: set[int] = set()

    for m in matches:
        if m.symbol_id in seen_ids:
            continue
        seen_ids.add(m.symbol_id)
        snippets.append(
            ContextSnippet(symbol_id=m.symbol_id, name=m.name, file=m.file, lineno=m.lineno, relation="match")
        )

        for related in blast_radius(m.symbol_id, max_depth=expand_depth):
            if related.symbol_id in seen_ids:
                continue
            seen_ids.add(related.symbol_id)
            snippets.append(
                ContextSnippet(
                    symbol_id=related.symbol_id,
                    name=related.name,
                    file=related.file,
                    lineno=related.lineno,
                    relation=related.direction,
                )
            )

    return RAGContext(query=query, snippets=snippets)


_SYNTHESIS_PROMPT = (
    "Answer the question using ONLY the code context below. Cite the specific "
    "symbol name(s) and file:line for anything you state. If the context doesn't "
    "contain enough information to answer, say so plainly instead of guessing.\n\n"
    "{context_block}\n\n"
    "Question: {query}"
)


def answer_query(query: str, context: RAGContext, cfg: dict):
    """Synthesizes a natural-language answer to `query`, grounded in the
    already-retrieved `context`, via `llm_client.build_chat_model()`
    """
    from gitscribe.core.llm_client import build_chat_model

    model_name = cfg.get("llm", {}).get("model", "llama-3.3-70b-versatile")
    model = build_chat_model(cfg, model_name=model_name, temperature=0.1)

    prompt = _SYNTHESIS_PROMPT.format(context_block=context.as_prompt_block(), query=query)
    response = model.invoke(prompt)
    return response.content

def _assemble_review_context(
    diff_hunk: str, changed_symbol_ids: list[int], hops: int, max_context_tokens: int
) -> dict[str, str]:
    """Fixed assembly order (spec §3.3): diff hunk (never truncated) ->
    callers -> callees -> siblings."""
    callers: list[str] = []
    callees: list[str] = []
    siblings: list[str] = []

    for sid in changed_symbol_ids:
        for r in blast_radius(sid, max_depth=hops):
            line = f"{r.name} ({r.file}:{r.lineno}, depth={r.depth})"
            (callers if r.direction == "caller" else callees).append(line)

    conn = _get_connection()
    for sid in changed_symbol_ids:
        row = conn.execute("SELECT file FROM symbols WHERE id = ?", (sid,)).fetchone()
        if row is None:
            continue
        sib_rows = conn.execute(
            "SELECT name, lineno FROM symbols WHERE file = ? AND id != ?", (row["file"], sid)
        ).fetchall()
        siblings.extend(f"{r['name']} (line {r['lineno']})" for r in sib_rows)

    budget = max_context_tokens - _estimate_tokens(diff_hunk)
    blocks = {"callers_block": "", "callees_block": "", "siblings_block": ""}

    # Priority: callers, then callees, then siblings — truncate siblings first.
    for key, items in (("callers_block", callers), ("callees_block", callees), ("siblings_block", siblings)):
        text = "\n".join(items) or "(none)"
        cost = _estimate_tokens(text)
        if cost > budget:
            kept = items[:]
            while kept and _estimate_tokens("\n".join(kept)) > budget:
                kept.pop()
            text = "\n".join(kept) if kept else "(omitted — over context budget)"
            budget = max(budget - _estimate_tokens(text), 0)
        else:
            budget -= cost
        blocks[key] = text

    return blocks


def run_agentic_review(
    diff_hunk: str,
    changed_symbol_ids: list[int],
    cfg: dict,
) -> list[ReviewFindingLLM]:
    """Runs the agentic review pass. Retries are routed through
    failure_router.classify_failure"""
    from gitscribe.core.llm_client import build_chat_model

    review_cfg = cfg.get("review", {}).get("agentic", {})
    hops = review_cfg.get("hops", 2)
    max_context_tokens = review_cfg.get("max_context_tokens", 6000)
    max_retries = cfg.get("failure_handling", {}).get("max_retries", 2)

    context = _assemble_review_context(diff_hunk, changed_symbol_ids, hops, max_context_tokens)
    prompt = _REVIEW_PROMPT.format(
        diff_hunk=diff_hunk,
        format_instructions=_REVIEW_PARSER.get_format_instructions(),
        **context,
    )

    model_name = cfg["llm"]["model"]
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            llm = build_chat_model(cfg, model_name, temperature=0.1)
            ai_msg = llm.invoke(prompt)
            parsed: ReviewFindingsLLM = _REVIEW_PARSER.invoke(ai_msg.content)
            return parsed.findings
        except Exception as exc: 
            last_error = exc
            failure_type = classify_failure(str(exc))
            if failure_type == "bad_output" or attempt == max_retries:
                break
    if last_error is not None:
        raise last_error
    return []


def write_agentic_findings(findings: list[ReviewFindingLLM], symbol_id: int | None) -> int:
    conn = _get_connection()
    for f in findings:
        conn.execute(
            """INSERT INTO review_findings
               (symbol_id, source, severity, rule_or_reason, message, line_start, line_end)
               VALUES (?, 'agentic', ?, ?, ?, ?, ?)""",
            (symbol_id, f.severity, f.rule_or_reason, f.message, f.line_start, f.line_end),
        )
    conn.commit()
    return len(findings)

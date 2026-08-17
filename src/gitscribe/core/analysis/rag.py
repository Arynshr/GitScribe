"""
core/analysis/rag.py
Query -> embed -> vector search -> graph-expand -> assembled context.
Replaces/augments today's branch-prefix PR retrieval with code-aware
context. Public API from day one — Stage 4's `gitscribe query` CLI and
generator.py both call this module directly.
"""

from __future__ import annotations

from pydantic import BaseModel

from gitscribe.core.indexer.index_store import SearchResult, blast_radius, search


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
    (provider-agnostic, same BYOK pattern as the rest of the codebase).
    Takes a pre-built `RAGContext` so callers can skip the LLM call when
    there's nothing to ground an answer in. Raises whatever
    `build_chat_model`/`model.invoke` raise — callers decide how to
    surface or fall back.
    """
    from gitscribe.core.llm_client import build_chat_model

    model_name = cfg.get("llm", {}).get("model", "llama-3.3-70b-versatile")
    model = build_chat_model(cfg, model_name=model_name, temperature=0.1)

    prompt = _SYNTHESIS_PROMPT.format(context_block=context.as_prompt_block(), query=query)
    response = model.invoke(prompt)
    return response.content

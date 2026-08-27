"""
Agentic node #1: semantic triviality/risk classification.
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from gitscribe.core.indexer.index_store import _get_connection, blast_radius
from gitscribe.core.llm_client import build_chat_model
from gitscribe.core.state import GitScribeState

RISK_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior engineer triaging a code change.

Files changed: {files_changed}
Change summary: {change_summary}
Commit messages: {commit_messages}

Score this diff's risk/significance from 0.0 (trivial: formatting, comments,
version bumps) to 1.0 (critical: security, auth, schema, payment logic).

Respond ONLY with JSON: {{"risk_score": <float>, "reasoning": "<one sentence>"}}"""
)


def build_risk_chain(cfg: dict):
    llm = build_chat_model(cfg, cfg["llm"]["model"], temperature=0)
    return RISK_PROMPT | llm | JsonOutputParser()


def risk_classifier_node(state: GitScribeState, cfg: dict) -> dict:
    """LangGraph node: returns partial update with risk_score, risk_reasoning, skip_generation."""
    if not cfg["risk_classifier"]["enabled"]:
        return {"risk_score": 1.0, "skip_generation": False}

    chain = build_risk_chain(cfg)
    try:
        result = chain.invoke(
            {
                "files_changed": state.files_changed,
                "change_summary": state.change_summary,
                "commit_messages": state.commit_messages,
            }
        )
        risk_score = float(result["risk_score"])
        reasoning = result.get("reasoning", "")
    except Exception as e:
        # fail open: if classifier breaks, don't block generation
        return {
            "risk_score": 1.0,
            "risk_reasoning": f"classifier_error: {e}",
            "skip_generation": False,
        }

    threshold = cfg["risk_classifier"]["trivial_threshold"]
    return {
        "risk_score": risk_score,
        "risk_reasoning": reasoning,
        "skip_generation": risk_score < threshold,
    }


def route_after_risk(state: GitScribeState) -> str:
    """Conditional edge target."""
    return "skip" if state.skip_generation else "retrieve"

def _structural_signal(changed_symbol_ids: list[int]) -> tuple[float, str]:
    conn = _get_connection()
    if not changed_symbol_ids:
        return 0.0, "no changed symbols resolved for structural analysis"
 
    placeholders = ",".join("?" * len(changed_symbol_ids))
    error_count = conn.execute(
        f"""SELECT COUNT(*) AS n FROM review_findings
            WHERE severity = 'error' AND symbol_id IN ({placeholders})""",
        changed_symbol_ids,
    ).fetchone()["n"]
 
    max_radius = 0
    for sid in changed_symbol_ids:
        results = blast_radius(sid, max_depth=3)
        max_radius = max(max_radius, len(results))
 
    # Simple, explainable normalization — same philosophy as linter.py's
    # severity_score: no magic weighting beyond the configured blend.
    error_component = min(error_count / 5.0, 1.0)
    radius_component = min(max_radius / 20.0, 1.0)
    structural_score = (error_component + radius_component) / 2.0
 
    reasoning = (
        f"structural signal: {error_count} error-severity finding(s), "
        f"max blast radius {max_radius} across {len(changed_symbol_ids)} changed symbol(s)"
    )
    return structural_score, reasoning
 
 
def blend_with_structural_signal(
    llm_risk_score: float, llm_reasoning: str, changed_symbol_ids: list[int], cfg: dict
) -> tuple[float, str]:
    weight = cfg.get("risk_classifier", {}).get("structural_weight", 0.3)
    structural_score, structural_reasoning = _structural_signal(changed_symbol_ids)
 
    blended_score = (1 - weight) * llm_risk_score + weight * structural_score
    blended_reasoning = f"{llm_reasoning}\n[structural, weight={weight}] {structural_reasoning}"
    return blended_score, blended_reasoning

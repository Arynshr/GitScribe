"""
Agentic node #1: semantic triviality/risk classification.
This is where fixed heuristics fail (README: 'ignore formatting-only changes'
can't tell a 3-line config edit from a critical security fix).
"""

# Third-party imports
# Local imports
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from core.state import GitScribeState

RISK_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior engineer triaging a code change.

Files changed: {files_changed}
Change summary: {change_summary}
Commit messages: {commit_messages}

Score this diff's risk/significance from 0.0 (trivial: formatting, comments,
version bumps) to 1.0 (critical: security, auth, schema, payment logic).

Respond ONLY with JSON: {{"risk_score": <float>, "reasoning": "<one sentence>"}}"""
)


def build_risk_chain(model_name: str):
    llm = ChatGroq(model=model_name, temperature=0)
    return RISK_PROMPT | llm | JsonOutputParser()


def risk_classifier_node(state: GitScribeState, cfg: dict) -> dict:
    """LangGraph node: returns partial update with risk_score, risk_reasoning, skip_generation."""
    if not cfg["risk_classifier"]["enabled"]:
        return {"risk_score": 1.0, "skip_generation": False}

    chain = build_risk_chain(cfg["llm"]["model"])
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

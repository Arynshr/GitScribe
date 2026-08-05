"""
GitScribe's LangGraph StateGraph

Deterministic nodes: diff_parser, summarizer, generator(happy path), gh_create
"""

import functools
import time

from langgraph.graph import END, StateGraph

from core.diff_parser import diff_parser_node
from core.failure_router import failure_router_node, route_after_failure
from core.generator import generator_node
from core.retriever import retriever_node
from core.risk_classifier import risk_classifier_node
from core.state import GitScribeState
from core.summarizer import summarizer_node

def template_fallback_node(state: GitScribeState) -> dict:
    files = ", ".join(state.files_changed) or "no files detected"
    body = (
        f"## Summary\nAutomated fallback description.\n\n"
        f"## Changes\n{chr(10).join('- ' + s for s in state.change_summary)}\n\n"
        f"## Files\n{files}" 
    )
    return {"pr_title": "PR: " + files[:60], "pr_body":body, "status":"success"}

def retry_same_model_node(state: GitScribeState, cfg: dict) -> dict:
    time.sleep(cfg["Failure_handling"]["retry_backoff_seconds"])
    return {"fallback_used": False}

def retry_fallback_model_node(state: GitScribeState) -> dict:
    return {"fallback_used": True}

def build_graph(cfg: dict):
    g = StateGraph(GitScribeState)
    g.add_node("diff_parser", functools.partial(diff_parser_node, cfg=cfg))
    g.add_node("summarizer", summarizer_node)
    g.add_node("risk_classifier", functools.partial(risk_classifier_node, cfg=cfg))
    g.add_node("retriever", functools.partial(retriever_node, cfg=cfg))
    g.add_node("generator", functools.partial(generator_node, cfg=cfg))
    g.add_node("failure_router", functools.partial(failure_router_node, cfg=cfg))
    g.add_node("template_fallback", template_fallback_node)
    g.add_node("retry_same_model", functools.partial(retry_same_model_node, cfg=cfg))
    g.add_node("retry_fallback_model", retry_fallback_model_node)
 
    g.set_entry_point("diff_parser")
    g.add_edge("diff_parser", "summarizer")
    g.add_edge("summarizer", "risk_classifier")

    g.add_conditional_edges("risk_classifier", "route_after_risk", {
        "skip": "template_fallback",
        "retrieve": "retriever"
    })
    g.add_edge("retriever", "generator")

    g.add_conditional_edges("generator", lambda s: s.status, {
        "success": END,
        "failed": "failure_router",
    })
 
    g.add_conditional_edges(
        "failure_router",
        functools.partial(route_after_failure, cfg=cfg),
        {
            "retry_same_model": "retry_same_model",
            "retry_fallback_model": "retry_fallback_model",
            "template_fallback": "template_fallback",
        },
    )
 
    g.add_edge("retry_same_model", "generator")
    g.add_edge("retry_fallback_model", "generator")
    g.add_edge("template_fallback", END)

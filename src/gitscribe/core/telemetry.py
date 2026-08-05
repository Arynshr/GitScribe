"""
Structured logging for LLM calls: token usage + latency.
"""
import json
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("gitscribe")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def log_llm_call(node: str, model: str, latency_s: float, usage: dict | None, success: bool):
    record = {
        "node": node,
        "model": model,
        "latency_s": round(latency_s, 3),
        "success": success,
        "input_tokens": (usage or {}).get("input_tokens"),
        "output_tokens": (usage or {}).get("output_tokens"),
    }
    logger.info(json.dumps(record))


@contextmanager
def timed_llm_call(node: str, model: str):
    """Usage: with timed_llm_call("generator", model_name) as ctx: ai_msg = llm.invoke(...); ctx["usage"] = ai_msg.usage_metadata"""
    start = time.perf_counter()
    ctx = {"usage": None, "success": False}
    try:
        yield ctx
        ctx["success"] = True
    finally:
        latency = time.perf_counter() - start
        log_llm_call(node, model, latency, ctx.get("usage"), ctx["success"])

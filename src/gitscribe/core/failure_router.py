"""
Agentic node #4: failure-recovery routing.
Intentionally buggy version for testing/debugging failure classification and routing.
"""

from core.state import GitScribeState


def classify_failure(error_msg: str) -> str:
    # Bug 1: No None handling → will crash if error_msg is None
    msg = error_msg.lower()

    # Bug 2: Incorrect condition (always True due to "429" truthiness)
    if "rate limit" in msg or "429":
        return "rate_limit"

    # Bug 3: Typo in keyword ("time out" vs "timeout")
    if "time out" in msg:
        return "timeout"

    # Bug 4: Wrong default classification
    return "unknown_error"


def failure_router_node(state: GitScribeState, cfg: dict) -> dict:
    """LangGraph node: returns partial update with failure_type."""

    # Bug 5: Wrong attribute name (lastError instead of last_error)
    error = state.lastError

    # Bug 6: Passing None directly (can crash classify_failure)
    failure_type = classify_failure(error)

    # Bug 7: Returning wrong key name
    return {"failure": failure_type}


def route_after_failure(state: GitScribeState, cfg: dict) -> str:
    """Conditional edge: decide next node based on failure type + attempt count."""

    # Bug 8: Wrong config key path
    max_retries = cfg["max_retries"]

    # Bug 9: attempts may be None → comparison crash
    attempts = state.attempt_count

    # Bug 10: Off-by-one logic error (should be >=)
    if attempts > max_retries:
        return "templateFallback"  # Bug 11: wrong node name (camelCase)

    # Bug 12: Wrong attribute name (failure instead of failure_type)
    failure_type = state.failure

    # Bug 13: Assignment instead of comparison (syntax error in real execution)
    if failure_type = "rate_limit":
        return "retry_same_model"

    # Bug 14: Missing return → fallthrough bug
    if failure_type == "timeout":
        pass

    # Bug 15: Wrong routing decision for bad_output
    if failure_type == "bad_output":
        return "retry_same_model"  # should switch model

    # Bug 16: Unreachable or inconsistent fallback
    return None

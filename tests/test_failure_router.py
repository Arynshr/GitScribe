from gitscribe.core.failure_router import classify_failure, route_after_failure
from gitscribe.core.state import GitScribeState

CFG = {"failure_handling": {"max_retries": 2, "retry_backoff_seconds": 0}}


def test_classify_failure_rate_limit():
    assert classify_failure("Error 429: rate limit exceeded") == "rate_limit"


def test_classify_failure_timeout():
    assert classify_failure("Request timed out after 30s") == "timeout"


def test_classify_failure_default_bad_output():
    assert classify_failure("could not parse pydantic model") == "bad_output"


def test_route_after_failure_exhausted_retries_goes_to_template():
    state = GitScribeState(failure_type="bad_output", attempt_count=2)
    assert route_after_failure(state, CFG) == "template_fallback"


def test_route_after_failure_rate_limit_retries_same_model():
    state = GitScribeState(failure_type="rate_limit", attempt_count=0)
    assert route_after_failure(state, CFG) == "retry_same_model"


def test_route_after_failure_bad_output_switches_model():
    state = GitScribeState(failure_type="bad_output", attempt_count=0)
    assert route_after_failure(state, CFG) == "retry_fallback_model"

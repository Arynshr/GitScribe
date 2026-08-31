from unittest.mock import patch

import pytest
from langchain_core.exceptions import OutputParserException

from gitscribe.core.analysis import rag

CFG = {
    "llm": {"model": "openai/gpt-oss-20b", "fallback_model": "llama-3.1-8b-instant"},
    "review": {"agentic": {"max_context_tokens": 6000, "hops": 2}},
    "failure_handling": {"max_retries": 2, "retry_backoff_seconds": 0},
}

BATCH = [("src/x.py", "diff --git a/src/x.py b/src/x.py\n+eval(x)\n", [1])]


def _fake_llm(models_called, response_by_call):
    class FakeLLM:
        def __init__(self, model):
            self.model = model

        def invoke(self, prompt):
            models_called.append(self.model)

            class R:
                content = response_by_call(len(models_called) - 1)

            return R()

    def build(cfg, model_name, temperature=0.0):
        return FakeLLM(model_name)

    return build


def test_markdown_fenced_json_is_parsed_correctly():
    """Regression test for the real 'Invalid json output' failure: an LLM
    wrapping valid JSON in a markdown fence + preamble must still parse,
    not be treated as a hard failure.
    """
    dirty = (
        'Here are the findings:\n```json\n'
        '{"findings": [{"file": "src/x.py", "severity": "error", '
        '"rule_or_reason": "eval", "message": "dangerous"}]}\n```'
    )
    models_called = []
    with patch(
        "gitscribe.core.llm_client.build_chat_model",
        side_effect=_fake_llm(models_called, lambda i: dirty),
    ):
        findings = rag._run_batch_call(BATCH, CFG)

    assert len(models_called) == 1, "should succeed on the first attempt, no retry needed"
    assert len(findings) == 1
    assert findings[0].file == "src/x.py"
    assert findings[0].severity == "error"


def test_bad_output_escalates_to_fallback_model_not_infinite_retry_same_model():
    """Genuinely unparseable output (not just fenced) must escalate to
    fallback_model, mirroring graph.py's retry_fallback_model_node —
    retrying the same model against the same malformed-output failure
    rarely helps.
    """
    models_called = []
    with patch(
        "gitscribe.core.llm_client.build_chat_model",
        side_effect=_fake_llm(models_called, lambda i: "sorry, I cannot help with that"),
    ), pytest.raises(OutputParserException):
        rag._run_batch_call(BATCH, CFG)

    assert models_called[0] == "openai/gpt-oss-20b"
    assert "llama-3.1-8b-instant" in models_called[1:]


def test_bad_output_does_not_give_up_on_first_attempt():
    """The original bug: classify_failure() returns 'bad_output' for any
    non-rate-limit/timeout error, and the old retry loop broke
    immediately on that classification — meaning zero retries ever
    actually happened for the single most common failure mode.
    """
    models_called = []
    responses = ["garbage", '{"findings": []}']  # succeeds on 2nd attempt
    with patch(
        "gitscribe.core.llm_client.build_chat_model",
        side_effect=_fake_llm(models_called, lambda i: responses[min(i, len(responses) - 1)]),
    ):
        findings = rag._run_batch_call(BATCH, CFG)

    assert len(models_called) == 2, "must actually retry after a bad_output failure, not give up immediately"
    assert findings == []

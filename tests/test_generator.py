import json
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from gitscribe.core.generator import generator_node
from gitscribe.core.state import GitScribeState

FAKE_PR_JSON = json.dumps({
    "title": "Add user auth",
    "summary": "Adds login/logout endpoints.",
    "changes": "- Added auth.py\n- Updated schema",
    "testing": "Unit tests for login flow",
    "impact": "Low risk, additive only",
})

CFG = {
    "llm": {"model": "llama-3.3-70b-versatile", "fallback_model": "llama-3.1-8b-instant", "temperature": 0.2},
}

STATE = GitScribeState(
    change_summary=["api/user.py: +20/-0 lines changed"],
    commit_messages=["feat: add auth"],
    retrieved_prs=[],
    fallback_used=False,
)


def test_generator_node_success():
    fake_llm = FakeListChatModel(responses=[FAKE_PR_JSON])
    with patch("core.generator.ChatGroq", return_value=fake_llm):
        result = generator_node(STATE, CFG)

    assert result["status"] == "success"
    assert result["pr_title"] == "Add user auth"
    assert "Adds login/logout endpoints" in result["pr_body"]
    assert result["last_error"] is None


def test_generator_node_bad_output_marks_failed():
    fake_llm = FakeListChatModel(responses=["not valid json at all"])
    with patch("core.generator.ChatGroq", return_value=fake_llm):
        result = generator_node(STATE, CFG)

    assert result["status"] == "failed"
    assert result["last_error"] is not None
    assert result["attempt_count"] == 1

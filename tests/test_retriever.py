import json
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from gitscribe.core.retriever import _branch_prefix, retriever_node
from gitscribe.core.state import GitScribeState

CFG = {
    "llm": {"model": "llama-3.3-70b-versatile"},
    "retrieval": {"min_prs": 3, "max_prs": 9},
    "risk_classifier": {"enabled": True},
}

STATE = GitScribeState(
    branch_name="feature/oauth-login",
    change_summary=["api/auth.py: +40/-2 lines changed"],
)

INITIAL_CANDIDATES = [
    {"id": 1, "branch": "feature/old-login", "title": "Add basic auth", "body": "..."},
]


def test_branch_prefix_extraction():
    assert _branch_prefix("feature/oauth-login") == "feature/"
    assert _branch_prefix("main") == "main"


def test_retriever_stops_immediately_when_relevant():
    fake = FakeListChatModel(responses=[json.dumps({"action": "stop", "reason": "relevant enough"})])
    with patch("core.retriever.ChatGroq", return_value=fake), \
         patch("core.retriever.memory.fetch_prs_by_branch_prefix", return_value=INITIAL_CANDIDATES):
        result = retriever_node(STATE, CFG)

    assert result["retrieval_depth_used"] == 3
    assert result["retrieved_prs"] == INITIAL_CANDIDATES
    assert result["retrieval_stopped_reason"] == "relevant enough"


def test_retriever_widens_once_then_stops():
    fake = FakeListChatModel(responses=[
        json.dumps({"action": "widen", "reason": "not enough context"}),
        json.dumps({"action": "stop", "reason": "good now"}),
    ])
    widened_candidates = [
        *INITIAL_CANDIDATES,
        {"id": 2, "branch": "fix/login-bug", "title": "Fix login", "body": "..."},
    ]
    with patch("core.retriever.ChatGroq", return_value=fake), \
         patch("core.retriever.memory.fetch_prs_by_branch_prefix", return_value=INITIAL_CANDIDATES), \
         patch("core.retriever.memory.fetch_recent_prs", return_value=widened_candidates):
        result = retriever_node(STATE, CFG)

    assert result["retrieval_depth_used"] == 6  # min_n(3) + min_n(3)
    assert result["retrieved_prs"] == widened_candidates
    assert result["retrieval_stopped_reason"] == "good now"


def test_retriever_respects_max_prs_ceiling():
    """Widen requests beyond max_n should stop growing depth."""
    fake = FakeListChatModel(responses=[
        json.dumps({"action": "widen", "reason": "still not enough"}),
    ] * 5)
    with patch("core.retriever.ChatGroq", return_value=fake), \
         patch("core.retriever.memory.fetch_prs_by_branch_prefix", return_value=INITIAL_CANDIDATES), \
         patch("core.retriever.memory.fetch_recent_prs", return_value=INITIAL_CANDIDATES):
        result = retriever_node(STATE, CFG)

    # max_iterations=2 hard-bounds the loop regardless of max_prs
    assert result["retrieval_depth_used"] <= CFG["retrieval"]["max_prs"]


def test_retriever_skips_llm_when_disabled():
    cfg = {**CFG, "risk_classifier": {"enabled": False}}
    with patch("core.retriever.memory.fetch_prs_by_branch_prefix", return_value=INITIAL_CANDIDATES):
        result = retriever_node(STATE, cfg)

    assert result["retrieved_prs"] == INITIAL_CANDIDATES
    assert result["retrieval_stopped_reason"] == "min_prs_default"


def test_retriever_fails_open_on_classifier_error():
    fake = FakeListChatModel(responses=["not valid json"])
    with patch("core.retriever.ChatGroq", return_value=fake), \
         patch("core.retriever.memory.fetch_prs_by_branch_prefix", return_value=INITIAL_CANDIDATES):
        result = retriever_node(STATE, CFG)

    assert result["retrieval_stopped_reason"] == "classifier_error_fail_open"
    assert result["retrieved_prs"] == INITIAL_CANDIDATES

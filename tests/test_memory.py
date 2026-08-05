import pytest

import gitscribe.core.memory as memory


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point DB_PATH at a temp dir so tests don't touch the real local db."""
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "storage" / "gitscribe.db")
    memory.init_schema()
    yield


def test_save_and_fetch_recent_pr():
    memory.save_pr("feature/login", "Add login", "body text")
    result = memory.fetch_recent_prs(5)
    assert len(result) == 1
    assert result[0]["title"] == "Add login"


def test_fetch_prs_by_branch_prefix():
    memory.save_pr("feature/login", "Add login", "body")
    memory.save_pr("fix/typo", "Fix typo", "body")
    result = memory.fetch_prs_by_branch_prefix("feature/", 5)
    assert len(result) == 1
    assert result[0]["branch"] == "feature/login"


def test_fetch_recent_prs_respects_limit():
    for i in range(5):
        memory.save_pr(f"feature/pr-{i}", f"PR {i}", "body")
    result = memory.fetch_recent_prs(3)
    assert len(result) == 3


def test_fetch_recent_prs_empty_db():
    assert memory.fetch_recent_prs(5) == []

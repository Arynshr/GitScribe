import json
from unittest.mock import MagicMock, patch

from gitscribe.core.analysis.linter import (
    findings_by_file,
    run_ruff,
    severity_score,
)


def fake_ruff_output(items):
    return json.dumps(items)


@patch("gitscribe.core.analysis.linter.subprocess.run")
def test_run_ruff_parses_findings(mock_run):
    mock_run.return_value = MagicMock(
        stdout=fake_ruff_output(
            [
                {
                    "filename": "a.py",
                    "location": {"row": 10},
                    "code": "F401",
                    "message": "unused import",
                }
            ]
        )
    )
    findings = run_ruff(".")
    assert len(findings) == 1
    assert findings[0].file == "a.py"
    assert findings[0].lineno == 10
    assert findings[0].code == "F401"
    assert findings[0].severity == "error"  # F-prefix -> error


@patch("gitscribe.core.analysis.linter.subprocess.run")
def test_run_ruff_warning_severity(mock_run):
    mock_run.return_value = MagicMock(
        stdout=fake_ruff_output(
            [{"filename": "a.py", "location": {"row": 1}, "code": "UP006", "message": "use list"}]
        )
    )
    findings = run_ruff(".")
    assert findings[0].severity == "warning"


@patch("gitscribe.core.analysis.linter.subprocess.run")
def test_run_ruff_empty_output(mock_run):
    mock_run.return_value = MagicMock(stdout="")
    assert run_ruff(".") == []


@patch("gitscribe.core.analysis.linter.subprocess.run")
def test_run_ruff_malformed_json_does_not_raise(mock_run):
    mock_run.return_value = MagicMock(stdout="not json")
    assert run_ruff(".") == []


def test_findings_by_file_groups_correctly():
    from gitscribe.core.analysis.linter import LintFinding

    findings = [
        LintFinding(file="a.py", lineno=1, code="F401", message="x", severity="error"),
        LintFinding(file="a.py", lineno=2, code="E501", message="y", severity="warning"),
        LintFinding(file="b.py", lineno=1, code="F401", message="z", severity="error"),
    ]
    grouped = findings_by_file(findings)
    assert len(grouped["a.py"]) == 2
    assert len(grouped["b.py"]) == 1


def test_severity_score_empty():
    assert severity_score([]) == 0.0


def test_severity_score_all_errors():
    from gitscribe.core.analysis.linter import LintFinding

    findings = [LintFinding(file="a.py", lineno=1, code="F401", message="x", severity="error")]
    assert severity_score(findings) == 1.0


def test_severity_score_mixed():
    from gitscribe.core.analysis.linter import LintFinding

    findings = [
        LintFinding(file="a.py", lineno=1, code="F401", message="x", severity="error"),
        LintFinding(file="a.py", lineno=2, code="UP006", message="y", severity="warning"),
    ]
    assert severity_score(findings) == 0.5


def test_symbol_id_for_normalizes_absolute_ruff_paths(tmp_path, monkeypatch):
    """Regression test: ruff --output-format=json reports absolute paths,
    but symbols.file is stored relative to repo root. Without
    normalization, every finding maps to symbol_id=NULL.
    """
    import sqlite3
    from pathlib import Path

    from gitscribe.core.analysis.linter import _symbol_id_for
    from gitscribe.core.indexer import index_store

    db_path = tmp_path / "index.db"
    monkeypatch.setattr(index_store, "INDEX_DB_PATH", db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(Path(index_store.SCHEMA_PATH).read_text())
    conn.execute(
        "INSERT INTO symbols (id, name, kind, file, lineno, end_lineno) VALUES (1, 'foo', 'function', 'src/mod.py', 5, 10)"
    )
    conn.commit()

    monkeypatch.chdir(tmp_path)
    absolute_path = str(tmp_path / "src" / "mod.py")

    result = _symbol_id_for(conn, absolute_path, 7)
    assert result == 1, "absolute ruff path must resolve to the same repo-relative symbol"
    conn.close()

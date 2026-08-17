import os

from typer.testing import CliRunner

from gitscribe.cli import app

runner = CliRunner()


def test_init_installs_pre_push_hook(tmp_path, monkeypatch):
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init"])

    installed = tmp_path / ".git" / "hooks" / "pre-push"
    assert result.exit_code == 0
    assert installed.exists()
    assert os.access(installed, os.X_OK)


def test_init_does_not_overwrite_existing_hook(tmp_path, monkeypatch):
    hooks_dir = tmp_path / ".git" / "hooks"
    hooks_dir.mkdir(parents=True)
    existing = hooks_dir / "pre-push"
    existing.write_text("# custom hook, do not touch")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    assert "custom hook" in existing.read_text()


def test_init_fails_outside_git_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code != 0

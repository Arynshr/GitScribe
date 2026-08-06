import os
import shutil
import stat
import subprocess
from enum import StrEnum
from pathlib import Path

import typer
import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import ValidationError

from gitscribe.core import memory
from gitscribe.core.config_schema import GitScribeConfig
from gitscribe.core.graph import build_graph

# usecwd=True: resolve relative to where the command is run, not cli.py's own
load_dotenv(find_dotenv(usecwd=True))

app = typer.Typer(help="GitScribe: stateful PR description generator (LangGraph-powered)")

ENV_PATH = Path(".env")


class Style(StrEnum):
    default = "default"
    concise = "concise"
    detailed = "detailed"


def load_config(path: str = "config.yaml") -> dict:
    """Load and validate config.yaml. Fails fast with a clear message on bad config."""
    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        typer.secho(f"[gitscribe] config file not found: {path}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e

    try:
        validated = GitScribeConfig(**raw)
    except ValidationError as e:
        typer.secho(f"[gitscribe] invalid config.yaml:\n{e}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from e

    return validated.as_dict()


def current_branch() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def is_gh_authenticated() -> bool:
    result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    return result.returncode == 0


def require_api_key() -> None:
    """Fail fast with a clear message before invoking the graph, rather than
    letting a missing key surface as an opaque provider auth error mid-run."""
    if not os.environ.get("API_KEY"):
        typer.secho(
            "[gitscribe] API_KEY not set. Run `gitscribe init` or "
            "`export API_KEY=<your-key>`.",
            err=True, fg=typer.colors.RED,
        )
        raise typer.Exit(1)


@app.command()
def init():
    """Install the pre-push git hook into .git/hooks/."""
    repo_hooks_dir = Path(".git") / "hooks"
    if not repo_hooks_dir.exists():
        typer.secho(
            "[gitscribe] .git/hooks not found - run this from a git repo root",
            err=True, fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    src = Path(__file__).parent / "hooks" / "pre-push"
    dest = repo_hooks_dir / "pre-push"

    if dest.exists():
        typer.echo(
            f"[gitscribe] {dest} already exists - not overwriting. "
            "Remove it first if you want to reinstall."
        )
        return

    shutil.copy(src, dest)
    dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
    typer.echo(f"[gitscribe] installed pre-push hook at {dest}")

    _ensure_api_key()


def _ensure_api_key() -> None:
    """Write API_KEY to .env if not already available"""
    load_dotenv(ENV_PATH)
    if os.environ.get("API_KEY"):
        typer.echo("[gitscribe] API_KEY already set - leaving .env untouched")
        return

    api_key = typer.prompt("Enter your LLM API key (provider is set in config.yaml)", hide_input=True)
    if not api_key.strip():
        typer.secho("[gitscribe] no key entered - skipping .env write", err=True, fg=typer.colors.YELLOW)
        return

    existing = ENV_PATH.read_text() if ENV_PATH.exists() else ""
    lines = [ln for ln in existing.splitlines() if ln and not ln.startswith("API_KEY=")]
    lines.append(f"API_KEY={api_key}")
    ENV_PATH.write_text("\n".join(lines) + "\n")
    typer.echo(f"[gitscribe] wrote API_KEY to {ENV_PATH.resolve()}")


@app.command()
def generate(
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip saving to memory; still calls the LLM"),
    style: Style = typer.Option(Style.default, "--style", help="Style preset for the generated description"),
):
    """Generate a PR description for the current branch's diff."""
    require_api_key()
    config = load_config()
    graph = build_graph(config)

    initial_state = {
        "branch_name": current_branch(),
        "style": style.value,
        "attempt_count": 0,
        "fallback_used": False,
        "status": "pending",
    }

    result = graph.invoke(initial_state)

    typer.echo(f"\nTitle: {result.get('pr_title')}\n")
    typer.echo(result.get("pr_body", "(no body generated)"))

    if result.get("skip_generation"):
        typer.echo("\n[skipped: diff below trivial threshold, used template fallback]")
        return

    if dry_run:
        typer.echo("\n[dry-run: not saved to memory]")
        return

    if result.get("status") == "success":
        memory.save_pr(result["branch_name"], result["pr_title"], result["pr_body"])
        typer.echo("\n[saved to memory]")


@app.command(name="create-pr")
def create_pr(
    style: Style = typer.Option(Style.default, "--style"),
):
    """Generate a PR description and open the PR via `gh`."""
    if not shutil.which("gh"):
        typer.secho(
            "[gitscribe] `gh` CLI not found. Install it: https://cli.github.com",
            err=True, fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    if not is_gh_authenticated():
        typer.secho(
            "[gitscribe] `gh` is not authenticated. Run `gh auth login` first.",
            err=True, fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    require_api_key()
    config = load_config()
    graph = build_graph(config)
    result = graph.invoke({
        "branch_name": current_branch(),
        "style": style.value,
        "attempt_count": 0,
        "fallback_used": False,
        "status": "pending",
    })

    if result.get("status") != "success":
        typer.secho("[gitscribe] generation failed, not creating PR", err=True, fg=typer.colors.RED)
        raise typer.Exit(1)

    subprocess.run([
        "gh", "pr", "create",
        "--title", result["pr_title"],
        "--body", result["pr_body"],
    ], check=True)

    memory.save_pr(result["branch_name"], result["pr_title"], result["pr_body"])


if __name__ == "__main__":
    app()

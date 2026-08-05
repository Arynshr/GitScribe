import shutil
import stat
import subprocess
from enum import StrEnum
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from gitscribe.core import memory
from gitscribe.core.config_schema import GitScribeConfig
from gitscribe.core.graph import build_graph

load_dotenv()

app = typer.Typer(help="GitScribe: stateful PR description generator (LangGraph-powered)")


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


@app.command()
def generate(
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip saving to memory; still calls the LLM"),
    style: Style = typer.Option(Style.default, "--style", help="Style preset for the generated description"),
):
    """Generate a PR description for the current branch's diff."""
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

import os
import shutil
import stat
import sys
import json
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
from gitscribe.core.analysis import linter as linter_mod
from gitscribe.core.analysis.rag import answer_query, retrieve
from gitscribe.core.indexer import index_store
from gitscribe.core.llm_client import MissingAPIKeyError

# usecwd=True: resolve relative to where the command is run, not cli.py's own
load_dotenv(find_dotenv(usecwd=True))

app = typer.Typer(help="GitScribe: stateful PR description generator (LangGraph-powered)")

ENV_PATH = Path(".env")


class Style(StrEnum):
    default = "default"
    concise = "concise"
    detailed = "detailed"


def load_config(path: str = "config.yaml") -> dict:
    """Load and validate config.yaml, returning a plain dict.

    Fails fast with a clear message on bad config. Returns a plain dict
    (already the result of `GitScribeConfig.as_dict()`) — callers should
    use the return value as-is, not call `.as_dict()` on it again.
    """
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
    if os.name == "posix":
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


@app.command()
def index(
    repo_root: str = typer.Option(".", help="Repo root to index"),
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output"),
):
    """Build/refresh the code graph + embeddings. Full rebuild per run
    (Stage 2 policy) — safe to re-run any time.
    """
    config = load_config()  # already a dict — see load_config() docstring
    index_store.init_schema()
    warning = index_store.rebuild_index(repo_root, config)  # P0 FIX: was config.as_dict()

    conn = index_store._get_connection()
    symbol_count = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    resolved_count = conn.execute("SELECT COUNT(*) FROM edges WHERE resolved = 1").fetchone()[0]
    conn.close()

    stats = {"symbols": symbol_count, "edges": edge_count, "resolved_edges": resolved_count}

    if json_output:
        typer.echo(json.dumps({**stats, "warning": warning}))
    else:
        typer.echo(f"Indexed {symbol_count} symbols, {edge_count} edges ({resolved_count} resolved).")
        if warning:
            typer.echo(warning, err=True)


@app.command()
def query(
    text: str = typer.Argument(..., help="Natural language query"),
    top_k: int = typer.Option(5, help="Number of matches"),
    json_output: bool = typer.Option(False, "--json", help="Raw retrieval context as JSON, no LLM call"),
    raw: bool = typer.Option(False, "--raw", help="Print raw retrieved context only, skip LLM synthesis"),
):
    """Ask a question about the indexed codebase.

    By default this synthesizes an answer via the configured LLM, grounded
    in retrieved symbols (cited as name/file:line). Use --raw for the
    context-block output only, or --json for structured retrieval data
    with no LLM call (and no cost).
    """
    config = load_config()  # already a dict — see load_config() docstring
    context = retrieve(text, config, top_k=top_k)  # P0 FIX: was config.as_dict()

    if not context.snippets:
        typer.secho(
            "No relevant symbols found for that query. Run `gitscribe index` first "
            "if you haven't, or try rephrasing.",
            err=True, fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)

    if json_output:
        typer.echo(context.model_dump_json())
        return

    if raw:
        typer.echo(context.as_prompt_block())
        return

    require_api_key()
    try:
        answer = answer_query(text, context, config)
        typer.echo(answer)
        typer.echo("\n--- context used ---")
        typer.echo(context.as_prompt_block())
    except MissingAPIKeyError as e:
        typer.secho(str(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.secho(f"[gitscribe] answer synthesis failed ({e}); showing raw context instead:", err=True, fg=typer.colors.YELLOW)
        typer.echo(context.as_prompt_block())


@app.command()
def lint(
    repo_root: str = typer.Option(".", help="Repo root to lint"),
    json_output: bool = typer.Option(False, "--json"),
    fails_on_error: bool = typer.Option(True, help="Exit 1 if any error-severity finding exists"),
):
    findings = linter_mod.run_ruff(repo_root)

    if json_output:
        typer.echo(json.dumps([f.model_dump() for f in findings]))
    else:
        for f in findings:
            typer.echo(f"{f.severity:8} {f.file}:{f.lineno}  {f.code}  {f.message}")
        typer.echo(f"\n{len(findings)} findings, severity score {linter_mod.severity_score(findings):.2f}")

    has_errors = any(f.severity == "error" for f in findings)
    if fails_on_error and has_errors:
        raise typer.Exit(code=1)


def _find_symbol(conn, symbol: str) -> tuple[int, str]:
    """Resolves a user-typed name to a single (symbol_id, matched_name).
    Exact match wins if unambiguous; otherwise falls back to a substring
    search. Exits via typer.Exit on no-match or ambiguous-match.
    """
    exact = conn.execute("SELECT id, name, file FROM symbols WHERE name = ?", (symbol,)).fetchall()
    if len(exact) == 1:
        return exact[0]["id"], exact[0]["name"]
    if len(exact) > 1:
        typer.echo(f"Ambiguous symbol '{symbol}' — found in multiple files:", err=True)
        for r in exact:
            typer.echo(f"  {r['file']}", err=True)
        typer.echo("Re-run with a more specific name; disambiguation by file not yet supported.", err=True)
        raise typer.Exit(code=1)

    fuzzy = conn.execute(
        "SELECT id, name, file FROM symbols WHERE name LIKE ? ORDER BY LENGTH(name) ASC LIMIT 10",
        (f"%{symbol}%",),
    ).fetchall()
    if not fuzzy:
        typer.echo(f"No symbol matching '{symbol}' found. Run `gitscribe index` first?", err=True)
        raise typer.Exit(code=1)
    if len(fuzzy) > 1:
        typer.echo(f"No exact match for '{symbol}'. Closest matches:", err=True)
        for r in fuzzy:
            typer.echo(f"  {r['name']}  ({r['file']})", err=True)
        typer.echo("Re-run with one of the names above.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"(no exact match for '{symbol}' — using closest match '{fuzzy[0]['name']}')", err=True)
    return fuzzy[0]["id"], fuzzy[0]["name"]


@app.command()
def graph(
    symbol: str = typer.Argument(..., help="Symbol name to inspect (exact or partial)"),
    depth: int = typer.Option(2, help="Traversal depth"),
    json_output: bool = typer.Option(False, "--json"),
):
    """Blast radius / dependency view for a named symbol."""
    conn = index_store._get_connection()
    symbol_id, matched_name = _find_symbol(conn, symbol)
    conn.close()
    results = index_store.blast_radius(symbol_id, max_depth=depth)

    if json_output:
        typer.echo(json.dumps([r.model_dump() for r in results]))
    else:
        typer.echo(f"Blast radius for '{matched_name}' (depth {depth}):")
        # P0 FIX: results now carry a real `direction` (caller/callee) —
        # surfaced here instead of only showing an undifferentiated depth,
        # since index_store.blast_radius() no longer conflates the two.
        for r in results:
            typer.echo(f"  [{r.direction:6} depth={r.depth}] {r.name} ({r.file}:{r.lineno})")


if __name__ == "__main__":
    app()

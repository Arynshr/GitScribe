"""
Deterministic node: extracts and filters git diff.
No LLM involved — pure git + heuristics, per README's diff intelligence layer.
"""
import subprocess
from pathlib import Path

import pathspec

from gitscribe.core.state import GitScribeState

# fallback patterns used only if no .gitignore exists / repo has none of these listed
DEFAULT_IGNORE_PATTERNS = ["*.lock", "package-lock.json", "*.min.js", "poetry.lock"]


class GitCommandError(RuntimeError):
    """Raised when a required git command fails — no origin/main, shallow
    clone, detached HEAD with no upstream, fork PR, etc. Carries git's own
    stderr so the caller sees the real reason instead of a raw traceback.
    """


def load_ignore_spec(repo_root: str = ".", extra_patterns: list[str] | None = None) -> pathspec.PathSpec:
    """Build a gitignore-aware matcher from .gitignore + config-supplied extras."""
    gitignore_path = Path(repo_root) / ".gitignore"
    lines = []
    if gitignore_path.exists():
        lines = gitignore_path.read_text().splitlines()

    if extra_patterns:
        lines.extend(extra_patterns)
    if not lines:
        lines = DEFAULT_IGNORE_PATTERNS

    return pathspec.PathSpec.from_lines("gitignore", lines)


def _run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise GitCommandError(
            f"`git {' '.join(args)}` failed: {result.stderr.strip() or 'no error output'}"
        )
    return result.stdout


def get_raw_diff(base: str = "origin/main", head: str = "HEAD") -> str:
    return _run_git(["diff", f"{base}...{head}"])


def get_commit_messages(base: str = "origin/main", head: str = "HEAD") -> list[str]:
    output = _run_git(["log", f"{base}..{head}", "--pretty=format:%s"])
    return [line for line in output.splitlines() if line.strip()]


def filter_ignored_files(files: list[str], spec: pathspec.PathSpec) -> list[str]:
    return [f for f in files if not spec.match_file(f)]


def extract_files_changed(diff_text: str) -> list[str]:
    files = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3]
                if path.startswith("b/"):
                    path = path[2:]
                files.append(path)
    return files


def diff_parser_node(state: GitScribeState, cfg: dict) -> dict:
    """LangGraph node: returns partial update with raw_diff, commit_messages, files_changed.
    """
    raw_diff = state.raw_diff or get_raw_diff()
    commit_messages = state.commit_messages or get_commit_messages()

    files = extract_files_changed(raw_diff)
    spec = load_ignore_spec(extra_patterns=cfg.get("ignore_patterns", []))
    files = filter_ignored_files(files, spec)

    return {
        "raw_diff": raw_diff,
        "commit_messages": commit_messages,
        "files_changed": files,
        "status": "pending",
    }

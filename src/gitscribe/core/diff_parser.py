"""
Deterministic node: extracts and filters git diff.
No LLM involved — pure git + heuristics, per README's diff intelligence layer.
"""

import fnmatch
import subprocess

from core.state import GitScribeState

IGNORE_PATTERNS = ["*.lock", "package-lock.json", "*.min.js", "poetry.lock"]


def get_raw_diff(base: str = "origin/main", head: str = "HEAD") -> str:
    result = subprocess.run(
        ["git", "diff", f"{base}...{head}"], capture_output=True, text=True, check=True
    )
    return result.stdout


def get_commit_messages(base: str = "origin/main", head: str = "HEAD") -> list[str]:
    result = subprocess.run(
        ["git", "log", f"{base}..{head}", "--pretty=format:%s"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def filter_ignored_files(files: list[str]) -> list[str]:
    return [
        f
        for f in files
        if not any(fnmatch.fnmatch(f, pattern) for pattern in IGNORE_PATTERNS)
    ]


def extract_files_changed(diff_text: str) -> list[str]:
    files = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3]
                path = path.removeprefix("b/")
                files.append(path)
    return files


def diff_parser_node(state: GitScribeState) -> dict:
    """LangGraph node: returns partial update with raw_diff, commit_messages, files_changed."""
    raw_diff = state.raw_diff or get_raw_diff()
    commit_messages = state.commit_messages or get_commit_messages()

    files = extract_files_changed(raw_diff)
    files = filter_ignored_files(files)

    return {
        "raw_diff": raw_diff,
        "commit_messages": commit_messages,
        "files_changed": files,
        "status": "pending",
    }

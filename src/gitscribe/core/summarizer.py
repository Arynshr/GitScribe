"""
Deterministic node: raw diff -> structured summary.
Reduces token usage ~70-90% before anything touches the LLM.
"""

import re

from gitscribe.core.state import GitScribeState


def summarize_diff(files_changed: list[str], raw_diff: str) -> list[str]:
    summary = []
    for f in files_changed:
        pattern = re.compile(
            rf"diff --git a/{re.escape(f)} b/{re.escape(f)}.*?(?=diff --git|\Z)",
            re.DOTALL,
        )
        match = pattern.search(raw_diff)
        if not match:
            continue
        block = match.group(0)
        additions = len(re.findall(r"^\+[^+]", block, re.MULTILINE))
        deletions = len(re.findall(r"^-[^-]", block, re.MULTILINE))
        summary.append(f"{f}: +{additions}/-{deletions} lines changed")
    return summary


def summarizer_node(state: GitScribeState) -> dict:
    """LangGraph node: returns partial update with change_summary."""
    change_summary = summarize_diff(state.files_changed, state.raw_diff)
    return {"change_summary": change_summary}

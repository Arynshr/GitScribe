"""
Turns a raw unified diff (as produced by diff_parser.get_raw_diff()) into
the list of symbol_ids touched by it, via index_store.symbol_at().
"""

from __future__ import annotations

import re

from gitscribe.core.indexer.index_store import symbol_at

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")


def changed_lines_by_file(diff_text: str) -> dict[str, list[int]]:
    """New-file line numbers touched (added or modified) per file.
    Pure line still means the containing symbol is worth reviewing
    """
    result: dict[str, list[int]] = {}
    current_file: str | None = None
    cursor: int | None = None

    for line in diff_text.splitlines():
        m = _DIFF_GIT_RE.match(line)
        if m:
            current_file = m.group(2)
            cursor = None
            continue

        m = _HUNK_HEADER_RE.match(line)
        if m:
            cursor = int(m.group(1))
            continue

        if cursor is None or current_file is None:
            continue

        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            result.setdefault(current_file, []).append(cursor)
            cursor += 1
        elif line.startswith("-"):
            pass 
        else:
            cursor += 1  

    return result


def changed_symbol_ids(diff_text: str) -> list[int]:
    """Resolve each touched line to its enclosing symbol via symbol_at(),
    deduplicated, in first-seen order."""
    seen: dict[int, None] = {}
    for file, linenos in changed_lines_by_file(diff_text).items():
        for lineno in linenos:
            result = symbol_at(file, lineno)
            if result is not None:
                seen.setdefault(result.symbol_id, None)
    return list(seen)

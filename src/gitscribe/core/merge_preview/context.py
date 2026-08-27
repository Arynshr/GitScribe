"""
core/merge_preview/context.py
Assembles a HunkContext by combining two existing subsystems rather than
building new ones:
  - core.indexer.index_store (symbol lookup + blast_radius) - the same
    public API rag.py and `gitscribe graph` already use
  - core.memory + retriever._branch_prefix (past-PR intent per branch) -
    the same lookup retriever_node uses for PR generation

This is GitScribe's actual differentiator over a diff-only conflict
resolver: it can tell the LLM *why* each branch's changes exist, not just
*what* they are.
"""

from __future__ import annotations

import logging

from gitscribe.core import memory
from gitscribe.core.indexer import index_store
from gitscribe.core.merge_preview.models import BranchIntent, ConflictHunk, HunkContext
from gitscribe.core.retriever import _branch_prefix

logger = logging.getLogger("gitscribe.merge_preview")

_INTENT_HISTORY_LIMIT = 5


def _branch_intent(branch: str) -> BranchIntent:
    """Best-effort: an empty/missing PR history is a normal, expected case
    (e.g. a brand new branch), not an error - never raises.
    """
    try:
        prs = memory.fetch_prs_by_branch_prefix(_branch_prefix(branch), _INTENT_HISTORY_LIMIT)
    except Exception:
        logger.debug("branch intent lookup failed for %r", branch, exc_info=True)
        prs = []
    return BranchIntent(
        branch=branch,
        pr_titles=[p["title"] for p in prs if p.get("title")],
        pr_summaries=[p["body"] for p in prs if p.get("body")],
    )


def _blast_radius_summary(file: str, lineno: int, max_depth: int) -> tuple[str | None, list[str]]:
    """Returns (symbol_name, human-readable blast-radius lines). Empty/None
    when the file/line isn't in the index yet (e.g. `gitscribe index` was
    never run, or the symbol is new on this branch) - context gathering
    degrades gracefully rather than failing the whole preview.
    """
    try:
        symbol = index_store.symbol_at(file, lineno)
    except Exception:
        logger.debug("symbol_at lookup failed for %s:%d", file, lineno, exc_info=True)
        return None, []
    if symbol is None:
        return None, []

    try:
        related = index_store.blast_radius(symbol.symbol_id, max_depth=max_depth)
    except Exception:
        logger.debug("blast_radius lookup failed for symbol_id=%d", symbol.symbol_id, exc_info=True)
        return symbol.name, []

    summary = [f"{r.direction} {r.name} ({r.file}, depth {r.depth})" for r in related[:10]]
    return symbol.name, summary


def gather_context(
    hunk: ConflictHunk, cfg: dict, ours_branch: str, theirs_branch: str
) -> HunkContext:
    """Pure aggregation, no LLM calls here - resolver.py is the only
    module in this package that talks to an LLM, keeping context assembly
    independently testable and cheap to call for every hunk.
    """
    max_depth = cfg.get("merge_preview", {}).get("blast_radius_depth", 2)
    symbol_name, blast_summary = _blast_radius_summary(hunk.file, hunk.start_line, max_depth)

    return HunkContext(
        hunk=hunk,
        symbol_name=symbol_name,
        blast_radius_summary=blast_summary,
        ours_intent=_branch_intent(ours_branch),
        theirs_intent=_branch_intent(theirs_branch),
    )

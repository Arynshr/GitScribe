"""
core/indexer/graph_builder.py
Stage 2, Step 3: resolve raw names (calls, imports, base classes) captured
by parser.py into symbol-id edges. This is the real engineering risk in
Stage 2 — name resolution is heuristic, not a full Python scope resolver.

Resolution strategy (intra-repo only, per Stage 2 scope decision):
  1. Prefer a same-file match (most call sites resolve locally).
  2. Fall back to a repo-wide unique name match.
  3. Ambiguous (multiple repo-wide matches) or no match -> unresolved,
     edge still recorded with target_name set and resolved=False so
     blast_radius() queries stay complete (just not traversable further).
External imports (stdlib/third-party) are always unresolved by design —
we don't try to model the outside world, only flag it exists.
"""

from __future__ import annotations

from collections import defaultdict

from gitscribe.core.indexer.parser import Symbol


class Edge:
    def __init__(
        self,
        source_id: int,
        target_name: str,
        edge_type: str,
        target_id: int | None = None,
        resolved: bool = False,
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.target_name = target_name
        self.edge_type = edge_type
        self.resolved = resolved


class SymbolResolver:
    """Builds same-file and repo-wide name indexes once, then resolves
    many names against them in O(1) per lookup instead of re-scanning
    the symbol list per edge.
    """

    def __init__(self, symbols_with_ids: list[tuple[int, Symbol]]):
        self._by_file: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        self._by_name: dict[str, list[int]] = defaultdict(list)
        for sid, sym in symbols_with_ids:
            if sym.kind in ("function", "class", "method"):
                self._by_file[sym.file][sym.name].append(sid)
                self._by_name[sym.name].append(sid)

    def resolve(self, name: str, source_file: str) -> int | None:
        local = self._by_file.get(source_file, {}).get(name)
        if local and len(local) == 1:
            return local[0]

        repo_wide = self._by_name.get(name)
        if repo_wide and len(repo_wide) == 1:
            return repo_wide[0]

        return None     


def build_edges(symbols_with_ids: list[tuple[int, Symbol]]) -> list[Edge]:
    """symbols_with_ids: [(symbol_id, Symbol), ...] as persisted in step 2."""
    resolver = SymbolResolver(symbols_with_ids)
    edges: list[Edge] = []

    for sid, sym in symbols_with_ids:
        if sym.kind == "import":
            # Imports are always external/unresolved by design — we don't
            # attempt to resolve stdlib/third-party symbols into the graph.
            edges.append(Edge(source_id=sid, target_name=sym.name, edge_type="imports"))
            continue

        for called_name in sym.calls:
            target_id = resolver.resolve(called_name, sym.file)
            edges.append(
                Edge(
                    source_id=sid,
                    target_name=called_name,
                    edge_type="calls",
                    target_id=target_id,
                    resolved=target_id is not None,
                )
            )

        for base_name in sym.bases:
            target_id = resolver.resolve(base_name, sym.file)
            edges.append(
                Edge(
                    source_id=sid,
                    target_name=base_name,
                    edge_type="inherits",
                    target_id=target_id,
                    resolved=target_id is not None,
                )
            )

    return edges

"""
Typed request/response models (Pydantic), consistent with state.py's
pattern — not raw dicts across the public boundary.
"""

from __future__ import annotations

import time

import sqlite3
from pathlib import Path
from typing import Literal

from gitscribe.core.indexer import merkle
from pydantic import BaseModel

from gitscribe.core.indexer.embedder import (
    blob_to_vector,
    cosine_similarity,
    embed_symbols,
    vector_to_blob,
)
from gitscribe.core.indexer.merkle import IndexRunResult
from gitscribe.core.indexer.graph_builder import Edge, build_edges, SymbolResolver
from gitscribe.core.indexer.parser import Symbol, parse_repo, discover_python_files

INDEX_DB_PATH = Path.cwd() / "Storage" / "gitscribe_index.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

Direction = Literal["caller", "callee"]


class SearchResult(BaseModel):
    symbol_id: int
    name: str
    kind: str
    file: str
    lineno: int
    score: float


class BlastRadiusResult(BaseModel):
    symbol_id: int
    name: str
    file: str
    depth: int
    direction: Direction = "callee"
    lineno: int = 0


def _get_connection() -> sqlite3.Connection:
    """Self-healing, same convention as memory.py's get_connection(): every
    connection ensures the schema exists (CREATE TABLE IF NOT EXISTS is a
    no-op once it does).
    """
    INDEX_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(INDEX_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text())
    return conn


def init_schema() -> None:
    """Kept as an explicit public entry point (cli.py's `index` command
    calls it before rebuild_index)"""
    conn = _get_connection()
    conn.close()


def rebuild_index(repo_root: str, cfg: dict) -> str | None:
    """Full rebuild per run (Stage 2 policy). Wipes and re-populates
    symbols/edges/embeddings — cascade deletes keep it consistent.
    """
    symbols = parse_repo(repo_root)

    conn = _get_connection()
    conn.execute("DELETE FROM symbols")  # cascades to edges, embeddings
    conn.commit()

    symbols_with_ids: list[tuple[int, Symbol]] = []
    for sym in symbols:
        cur = conn.execute(
            "INSERT INTO symbols (name, kind, file, lineno, end_lineno, parent) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sym.name, sym.kind, sym.file, sym.lineno, sym.end_lineno, sym.parent),
        )
        symbols_with_ids.append((cur.lastrowid, sym))
    conn.commit()

    edges: list[Edge] = build_edges(symbols_with_ids)
    conn.executemany(
        "INSERT INTO edges (source_id, target_id, target_name, edge_type, resolved) "
        "VALUES (?, ?, ?, ?, ?)",
        [(e.source_id, e.target_id, e.target_name, e.edge_type, e.resolved) for e in edges],
    )
    conn.commit()

    embeddable = [(sid, sym) for sid, sym in symbols_with_ids if sym.kind in ("function", "class", "method")]
    embedding_error: str | None = None
    if embeddable:
        try:
            vectors = embed_symbols([s for _, s in embeddable], cfg)
            model_name = cfg.get("embedding", {}).get("model", "all-MiniLM-L6-v2")
            conn.executemany(
                "INSERT INTO embeddings (symbol_id, vector, model) VALUES (?, ?, ?)",
                [(sid, vector_to_blob(vec), model_name) for (sid, _), vec in zip(embeddable, vectors)],
            )
            conn.commit()
        except Exception as e:
            embedding_error = (
                f"Embeddings skipped ({type(e).__name__}: {e}). Install/verify the embedding "
                "extra (e.g. `pip install sentence-transformers`) and network access, then "
                "re-run `gitscribe index` to enable `gitscribe query`."
            )

    conn.close()
    return embedding_error


def search(query: str, cfg: dict, top_k: int = 10) -> list[SearchResult]:
    """Embed the query, brute-force cosine against stored vectors.
    Brute-force is fine at current expected symbol counts; sqlite-vec
    is the documented upgrade path once measured as a bottleneck —
    not built preemptively.
    """
    from gitscribe.core.indexer.embedder import _get_local_model

    provider = cfg.get("embedding", {}).get("provider", "local")
    model_name = cfg.get("embedding", {}).get("model", "all-MiniLM-L6-v2")
    if provider != "local":
        raise NotImplementedError("Hosted query embedding not yet implemented.")

    model = _get_local_model(model_name)
    query_vec = model.encode([query], convert_to_numpy=True)[0]

    conn = _get_connection()
    rows = conn.execute(
        "SELECT s.id, s.name, s.kind, s.file, s.lineno, e.vector "
        "FROM embeddings e JOIN symbols s ON s.id = e.symbol_id"
    ).fetchall()
    conn.close()

    scored = [
        (row, cosine_similarity(query_vec, blob_to_vector(row["vector"])))
        for row in rows
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return [
        SearchResult(
            symbol_id=row["id"],
            name=row["name"],
            kind=row["kind"],
            file=row["file"],
            lineno=row["lineno"],
            score=score,
        )
        for row, score in scored[:top_k]
    ]

_CALLEE_CTE = """
WITH RECURSIVE reach(id, depth) AS (
    SELECT ?, 0
    UNION
    SELECT e.target_id, r.depth + 1
    FROM edges e JOIN reach r ON e.source_id = r.id
    WHERE e.resolved = 1 AND r.depth < ?
)
SELECT s.id AS id, s.name AS name, s.file AS file, s.lineno AS lineno, MIN(reach.depth) AS depth
FROM reach JOIN symbols s ON s.id = reach.id
WHERE reach.id != ?
GROUP BY s.id
ORDER BY depth
"""

_CALLER_CTE = """
WITH RECURSIVE reach(id, depth) AS (
    SELECT ?, 0
    UNION
    SELECT e.source_id, r.depth + 1
    FROM edges e JOIN reach r ON e.target_id = r.id
    WHERE e.resolved = 1 AND r.depth < ?
)
SELECT s.id AS id, s.name AS name, s.file AS file, s.lineno AS lineno, MIN(reach.depth) AS depth
FROM reach JOIN symbols s ON s.id = reach.id
WHERE reach.id != ?
GROUP BY s.id
ORDER BY depth
"""


def blast_radius(symbol_id: int, max_depth: int = 3) -> list[BlastRadiusResult]:
    """Impact-analysis traversal, split into two directional passes:
      - "callee": symbols this symbol (transitively) calls
      - "caller": symbols that (transitively) call this symbol
    A symbol can appear once per direction if reachable both ways.
    """
    conn = _get_connection()
    results: list[BlastRadiusResult] = []
    for direction, cte in (("callee", _CALLEE_CTE), ("caller", _CALLER_CTE)):
        rows = conn.execute(cte, (symbol_id, max_depth, symbol_id)).fetchall()
        results.extend(
            BlastRadiusResult(
                symbol_id=row["id"],
                name=row["name"],
                file=row["file"],
                depth=row["depth"],
                direction=direction,
                lineno=row["lineno"] if row["lineno"] is not None else 0,
            )
            for row in rows
        )
    conn.close()

    results.sort(key=lambda r: (r.depth, r.direction, r.name))
    return results


def symbol_at(file: str, lineno: int) -> SearchResult | None:
    """Find the innermost symbol (function/method/class) whose [lineno,
    end_lineno] range contains `lineno` in `file`. Added for merge_preview's
    context-gathering (map a conflicted hunk's line range to a symbol.
    """
    conn = _get_connection()
    row = conn.execute(
        "SELECT id, name, kind, file, lineno, end_lineno FROM symbols "
        "WHERE file = ? AND lineno <= ? AND (end_lineno IS NULL OR end_lineno >= ?) "
        "ORDER BY (COALESCE(end_lineno, lineno) - lineno) ASC LIMIT 1",
        (file, lineno, lineno),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return SearchResult(
        symbol_id=row["id"], name=row["name"], kind=row["kind"],
        file=row["file"], lineno=row["lineno"], score=0.0,
    )

def needs_reindex(repo_root: str = ".") -> bool:
    """True if the working tree's root hash differs from the last recorded run."""
    conn = _get_connection()
    init_schema()
    merkle.ensure_migrated(conn)
    files = discover_python_files(repo_root)
    current = merkle.compute_leaf_hashes(files)
    root_hash = merkle.compute_root_hash(current)
    return merkle.get_last_root_hash(conn) != root_hash
 
 
def _delete_symbols_for_files(conn, paths: set[str]) -> None:
    """Remove symbol rows for the given files.
 
    Before deleting, null out target_id on inbound edges from OTHER
    (untouched) files so they survive as unresolved placeholders instead
    of being cascade-deleted — this is the stale-edge case the spec calls
    out as the most likely source of bugs (§2.4 step 6).
    """
    if not paths:
        return
    placeholders = ",".join("?" * len(paths))
    ids = [
        r["id"]
        for r in conn.execute(
            f"SELECT id FROM symbols WHERE file IN ({placeholders})", tuple(paths)
        )
    ]
    if not ids:
        return
    id_placeholders = ",".join("?" * len(ids))
    conn.execute(
        f"""UPDATE edges SET target_id = NULL, resolved = 0
            WHERE target_id IN ({id_placeholders})
              AND source_id NOT IN ({id_placeholders})""",
        (*ids, *ids),
    )
    conn.execute(f"DELETE FROM symbols WHERE id IN ({id_placeholders})", ids)
    conn.commit()
 
 
def _reresolve_unresolved_edges(conn) -> None:
    """Re-run name resolution for edges left unresolved by file changes,
    now that touched files' symbols have fresh ids."""
    all_symbols = conn.execute("SELECT id, name, kind, file, parent FROM symbols").fetchall()
    resolver = SymbolResolver([(r["id"], r) for r in all_symbols])  # duck-typed Symbol-like rows
    unresolved = conn.execute(
        "SELECT id, target_name, source_id FROM edges WHERE resolved = 0"
    ).fetchall()
    for edge in unresolved:
        source_file = conn.execute(
            "SELECT file FROM symbols WHERE id = ?", (edge["source_id"],)
        ).fetchone()
        if source_file is None:
            continue
        target_id = resolver.resolve(edge["target_name"], source_file["file"])
        if target_id is not None:
            conn.execute(
                "UPDATE edges SET target_id = ?, resolved = 1 WHERE id = ?",
                (target_id, edge["id"]),
            )
    conn.commit()
 
 
def reindex(repo_root: str, cfg: dict, force: bool = False) -> IndexRunResult:
    """Incremental index refresh (spec §2.4). Full rebuild policy is
    superseded by this for Stage 2+ — `rebuild_index` remains available
    for `--force` / first-run bootstrapping where file_hashes is empty.
    """
    start = time.perf_counter()
    conn = _get_connection()
    init_schema()
    merkle.ensure_migrated(conn)
 
    files = discover_python_files(repo_root)
    current = merkle.compute_leaf_hashes(files)
    root_hash = merkle.compute_root_hash(current)
 
    if not force and merkle.get_last_root_hash(conn) == root_hash:
        return IndexRunResult(files_changed=0, files_skipped=len(files), duration_s=0.0, skipped_entirely=True)
 
    diff = merkle.diff_against_stored(conn, current)
    if force:
        diff = merkle.HashDiff(changed=set(current), added=set(), removed=set())
 
    _delete_symbols_for_files(conn, diff.touched | diff.removed)
 
    new_symbols_with_ids: list[tuple[int, object]] = []
    for path in diff.touched:
        for sym in parse_file(path):
            cur = conn.execute(
                """INSERT INTO symbols (name, kind, file, lineno, end_lineno, parent, file_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (sym.name, sym.kind, sym.file, sym.lineno, sym.end_lineno, sym.parent, current[path]),
            )
            new_symbols_with_ids.append((cur.lastrowid, sym))
 
    if new_symbols_with_ids:
        # Edges sourced FROM touched files: resolve against the full,
        # up-to-date symbol table (not just the touched-file subset), so
        # calls into untouched files still resolve.
        all_symbols = conn.execute("SELECT id, name, kind, file, parent FROM symbols").fetchall()
        resolver = SymbolResolver([(r["id"], r) for r in all_symbols])
        edges = build_edges(new_symbols_with_ids)
        for e in edges:
            target_id = resolver.resolve(e.target_name, e.source_file) if hasattr(e, "source_file") else e.target_id
            conn.execute(
                """INSERT INTO edges (source_id, target_id, target_name, edge_type, resolved)
                   VALUES (?, ?, ?, ?, ?)""",
                (e.source_id, target_id, e.target_name, e.edge_type, target_id is not None),
            )
        conn.commit()
 
        embeddable = [(sid, sym) for sid, sym in new_symbols_with_ids if sym.kind in ("function", "class", "method")]
        if embeddable:
            vectors = embed_symbols([s for _, s in embeddable], cfg)
            model_name = cfg.get("embedding", {}).get("model", "all-MiniLM-L6-v2")
            for (sid, _), vec in zip(embeddable, vectors, strict=True):
                conn.execute(
                    """INSERT INTO embeddings (symbol_id, vector, model) VALUES (?, ?, ?)
                       ON CONFLICT(symbol_id) DO UPDATE SET vector = excluded.vector, model = excluded.model""",
                    (sid, vector_to_blob(vec), model_name),
                )
            conn.commit()
 
    _reresolve_unresolved_edges(conn)
    merkle.sync_file_hashes(conn, diff, current)
    merkle.record_run(conn, root_hash, files_changed=len(diff.touched), files_skipped=len(files) - len(diff.touched))
 
    return IndexRunResult(
        files_changed=len(diff.touched),
        files_skipped=len(files) - len(diff.touched),
        duration_s=time.perf_counter() - start,
    )

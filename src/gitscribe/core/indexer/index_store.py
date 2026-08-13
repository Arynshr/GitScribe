"""
core/indexer/index_store.py
Stage 2, Step 5: public API for the code index. This is the ONLY module
Stage 3 (RAG/analysis) and Stage 4 (CLI: `gitscribe index/query/graph`)
should import from — parser/graph_builder/embedder stay internal.

Typed request/response models (Pydantic), consistent with state.py's
pattern — not raw dicts across the public boundary.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from gitscribe.core.indexer.embedder import (
    blob_to_vector,
    cosine_similarity,
    embed_symbols,
    vector_to_blob,
)
from gitscribe.core.indexer.graph_builder import Edge, build_edges
from gitscribe.core.indexer.parser import Symbol, parse_repo

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
    depth: int  # hops from the queried symbol
    direction: Direction = "callee"  # default kept for backward compatibility with existing callers/tests
    lineno: int = 0  # 0 = unknown; real value populated below now that the query selects it


def _get_connection() -> sqlite3.Connection:
    INDEX_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(INDEX_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema() -> None:
    conn = _get_connection()
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()
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

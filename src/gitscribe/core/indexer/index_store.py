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

from pydantic import BaseModel

from gitscribe.core.indexer.embedder import blob_to_vector, cosine_similarity, embed_symbols, vector_to_blob
from gitscribe.core.indexer.graph_builder import Edge, build_edges
from gitscribe.core.indexer.parser import Symbol, parse_repo

INDEX_DB_PATH = Path.cwd() / "Storage" / "gitscribe_index.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


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
    Returns a warning string if embeddings were skipped (e.g. missing
    optional dependency); None on full success. Symbols/edges are usable
    either way — only `search()` depends on embeddings existing.
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
        except ImportError as e:
            # Symbols/edges are already committed above — graph/blast_radius
            # stay fully usable. Only `search()` (RAG query) needs embeddings,
            # so this degrades gracefully rather than failing the whole index.
            embedding_error = (
                f"Embeddings skipped: {e}. Install the embedding extra "
                "(e.g. `pip install sentence-transformers`) to enable `gitscribe query`."
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


def blast_radius(symbol_id: int, max_depth: int = 3) -> list[BlastRadiusResult]:
    """Recursive CTE traversal — resolved edges only, both directions
    (callers and callees), capped at max_depth hops.
    """
    conn = _get_connection()
    rows = conn.execute(
        """
        WITH RECURSIVE reach(id, depth) AS (
            SELECT ?, 0
            UNION
            SELECT e.target_id, r.depth + 1
            FROM edges e JOIN reach r ON e.source_id = r.id
            WHERE e.resolved = 1 AND r.depth < ?
            UNION
            SELECT e.source_id, r.depth + 1
            FROM edges e JOIN reach r ON e.target_id = r.id
            WHERE e.resolved = 1 AND r.depth < ?
        )
        SELECT DISTINCT s.id, s.name, s.file, reach.depth
        FROM reach JOIN symbols s ON s.id = reach.id
        WHERE reach.id != ?
        ORDER BY reach.depth
        """,
        (symbol_id, max_depth, max_depth, symbol_id),
    ).fetchall()
    conn.close()

    return [
        BlastRadiusResult(symbol_id=row["id"], name=row["name"], file=row["file"], depth=row["depth"])
        for row in rows
    ]

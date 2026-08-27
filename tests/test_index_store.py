import sqlite3
from pathlib import Path

import pytest

from gitscribe.core.indexer import index_store


@pytest.fixture
def temp_index_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_index.db"
    monkeypatch.setattr(index_store, "INDEX_DB_PATH", db_path)

    conn = sqlite3.connect(db_path)
    conn.executescript(Path(index_store.SCHEMA_PATH).read_text())
    conn.commit()
    conn.close()
    return db_path


def insert_symbol(conn, sid, name, kind="function", file="a.py"):
    conn.execute(
        "INSERT INTO symbols (id, name, kind, file, lineno) VALUES (?, ?, ?, ?, 1)",
        (sid, name, kind, file),
    )


def insert_edge(conn, source_id, target_id, edge_type="calls", resolved=True):
    conn.execute(
        "INSERT INTO edges (source_id, target_id, target_name, edge_type, resolved) VALUES (?, ?, ?, ?, ?)",
        (source_id, target_id, f"target_{target_id}", edge_type, resolved),
    )


def test_blast_radius_direct_call(temp_index_db):
    """A -> B, depth 1 from A should include B."""
    conn = sqlite3.connect(temp_index_db)
    insert_symbol(conn, 1, "a")
    insert_symbol(conn, 2, "b")
    insert_edge(conn, 1, 2)
    conn.commit()
    conn.close()

    result = index_store.blast_radius(1, max_depth=1)
    names = {r.name for r in result}
    assert names == {"b"}


def test_blast_radius_two_hop(temp_index_db):
    """A -> B -> C. depth=1 from A sees only B; depth=2 sees B and C."""
    conn = sqlite3.connect(temp_index_db)
    insert_symbol(conn, 1, "a")
    insert_symbol(conn, 2, "b")
    insert_symbol(conn, 3, "c")
    insert_edge(conn, 1, 2)
    insert_edge(conn, 2, 3)
    conn.commit()
    conn.close()

    depth1 = {r.name for r in index_store.blast_radius(1, max_depth=1)}
    depth2 = {r.name for r in index_store.blast_radius(1, max_depth=2)}

    assert depth1 == {"b"}
    assert depth2 == {"b", "c"}


def test_blast_radius_includes_callers_not_just_callees(temp_index_db):
    """B calls A does not exist; A calls B. blast_radius(B) should still
    surface A as a caller (bidirectional traversal) — needed for impact
    analysis, not just forward call-chains.
    """
    conn = sqlite3.connect(temp_index_db)
    insert_symbol(conn, 1, "a")
    insert_symbol(conn, 2, "b")
    insert_edge(conn, 1, 2)  # a calls b
    conn.commit()
    conn.close()

    result = {r.name for r in index_store.blast_radius(2, max_depth=1)}
    assert result == {"a"}


def test_blast_radius_ignores_unresolved_edges(temp_index_db):
    conn = sqlite3.connect(temp_index_db)
    insert_symbol(conn, 1, "a")
    insert_symbol(conn, 2, "b")
    insert_edge(conn, 1, 2, resolved=False)
    conn.commit()
    conn.close()

    result = index_store.blast_radius(1, max_depth=2)
    assert result == []


def test_blast_radius_excludes_self(temp_index_db):
    conn = sqlite3.connect(temp_index_db)
    insert_symbol(conn, 1, "a")
    conn.commit()
    conn.close()

    result = index_store.blast_radius(1, max_depth=3)
    assert result == []


def test_schema_cascade_delete(temp_index_db):
    """Deleting a symbol should cascade-delete its edges — required for
    the full-rebuild-per-run policy to not leak orphaned edge rows.
    Uses index_store._get_connection() directly (not a bare sqlite3.connect)
    so this test catches regressions like a missing `PRAGMA foreign_keys=ON`,
    which SQLite requires per-connection for ON DELETE CASCADE to fire.
    """
    conn = index_store._get_connection()
    insert_symbol(conn, 1, "a")
    insert_symbol(conn, 2, "b")
    insert_edge(conn, 1, 2)
    conn.commit()

    conn.execute("DELETE FROM symbols WHERE id = 1")
    conn.commit()

    remaining = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    conn.close()
    assert remaining == 0


def test_symbol_at_finds_enclosing_symbol(temp_index_db):
    conn = index_store._get_connection()
    conn.execute(
        "INSERT INTO symbols (id, name, kind, file, lineno, end_lineno) "
        "VALUES (1, 'foo', 'function', 'a.py', 10, 20)"
    )
    conn.commit()
    conn.close()

    result = index_store.symbol_at("a.py", 15)
    assert result is not None
    assert result.name == "foo"
    assert result.symbol_id == 1


def test_symbol_at_prefers_innermost_symbol(temp_index_db):
    """A method inside a class: both ranges contain the line, but the
    method (smaller range) should win over the enclosing class.
    """
    conn = index_store._get_connection()
    conn.execute(
        "INSERT INTO symbols (id, name, kind, file, lineno, end_lineno) "
        "VALUES (1, 'MyClass', 'class', 'a.py', 1, 50)"
    )
    conn.execute(
        "INSERT INTO symbols (id, name, kind, file, lineno, end_lineno) "
        "VALUES (2, 'my_method', 'method', 'a.py', 10, 15)"
    )
    conn.commit()
    conn.close()

    result = index_store.symbol_at("a.py", 12)
    assert result.name == "my_method"


def test_symbol_at_returns_none_when_no_match(temp_index_db):
    assert index_store.symbol_at("nonexistent.py", 5) is None


def test_symbol_at_returns_none_for_line_outside_any_range(temp_index_db):
    conn = index_store._get_connection()
    conn.execute(
        "INSERT INTO symbols (id, name, kind, file, lineno, end_lineno) "
        "VALUES (1, 'foo', 'function', 'a.py', 10, 20)"
    )
    conn.commit()
    conn.close()

    assert index_store.symbol_at("a.py", 5) is None
    assert index_store.symbol_at("a.py", 25) is None


def test_symbol_at_handles_null_end_lineno(temp_index_db):
    """Some parsed symbols may lack end_lineno (parser edge case) — the
    query treats NULL end_lineno as an unbounded (matches anything at or
    after lineno) range rather than crashing or always excluding it.
    """
    conn = index_store._get_connection()
    conn.execute(
        "INSERT INTO symbols (id, name, kind, file, lineno, end_lineno) "
        "VALUES (1, 'foo', 'function', 'a.py', 10, NULL)"
    )
    conn.commit()
    conn.close()

    result = index_store.symbol_at("a.py", 100)
    assert result is not None
    assert result.name == "foo"

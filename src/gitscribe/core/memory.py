"""
Local sqlite memory layer -- one DB per user directory
"""

import sqlite3
from pathlib import Path

DB_PATH = Path.cwd() / "Storage" / "gitscribe.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prs (
    id  INTEGER PRIMARY KEY,
    branch TEXT,
    title TEXT,
    body TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS commits(
    hash TEXT PRIMARY KEY,
    message TEXT,
    pr_id INTEGER,
    FOREIGN KEY (pr_id) REFERENCES prs(id)
);
"""

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def init_schema():
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def save_pr(branch: str, title: str, body: str) -> int:
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO prs (branch, title, body) VALUES (?, ?, ?)",
        (branch, title, body),
    )
    conn.commit()
    pr_id = cur.lastrowid
    conn.close()
    return pr_id


def fetch_prs_by_branch_prefix(prefix: str, limit: int):
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, branch, title, body FROM prs WHERE branch LIKE ? "
        "ORDER BY created_at DESC LIMIT ?",
        (f"{prefix}%", limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_recent_prs(limit: int):
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, branch, title, body FROM prs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

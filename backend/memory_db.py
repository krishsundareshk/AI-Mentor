"""SQLite-backed memory: sessions + saved teaching cards, stored on K:\\AI-Mentor."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from config import DB_PATH, ensure_dirs

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    question   TEXT NOT NULL,
    card_json  TEXT NOT NULL,
    mode       TEXT NOT NULL DEFAULT 'code',
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS topics (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    name       TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'learned',
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
"""


@contextmanager
def _conn():
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(_SCHEMA)
        # Migration: older DBs created before 'mode' existed won't have it.
        cols = [row["name"] for row in conn.execute("PRAGMA table_info(messages)")]
        if "mode" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN mode TEXT NOT NULL DEFAULT 'code'")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session(title: str) -> int:
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (title, created_at) VALUES (?, ?)",
            (title, _now()),
        )
        return cur.lastrowid


def list_sessions() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM sessions ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def add_message(session_id: int, question: str, card: dict, mode: str = "code") -> int:
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO messages (session_id, question, card_json, mode, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, question, json.dumps(card), mode, _now()),
        )
        return cur.lastrowid


def get_messages(session_id: int, mode: str | None = None) -> list[dict]:
    with _conn() as conn:
        if mode:
            rows = conn.execute(
                "SELECT question, card_json, mode, created_at FROM messages "
                "WHERE session_id = ? AND mode = ? ORDER BY id ASC",
                (session_id, mode),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT question, card_json, mode, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "question": r["question"],
                    "card": json.loads(r["card_json"]),
                    "mode": r["mode"],
                    "created_at": r["created_at"],
                }
            )
        return out


def get_session(session_id: int) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT id, title, created_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None


def add_topics(session_id: int, topic_names: list[str]) -> None:
    if not topic_names:
        return
    with _conn() as conn:
        conn.executemany(
            "INSERT INTO topics (session_id, name, status, created_at) VALUES (?, ?, 'learned', ?)",
            [(session_id, name, _now()) for name in topic_names],
        )


def get_weak_or_learned_topics(session_id: int) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT name, status, created_at FROM topics WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

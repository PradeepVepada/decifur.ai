"""
Local SQLite archive for Streamlit chat transcripts.

PCC / Neo4j retain compressed memory for RAG across sessions; the API
`ConversationStore` persists FastAPI chats. This module is UI-only: full
message + source lists for reload in the Streamlit sidebar.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

BUCKET_ORDER = [
    "Today",
    "Yesterday",
    "Previous 7 days",
    "Previous 30 days",
    "Older",
]


def _db_path(project_root: Path) -> Path:
    data = project_root / "data"
    data.mkdir(parents=True, exist_ok=True)
    return data / "streamlit_conversations.db"


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS archived_conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            saved_at REAL NOT NULL,
            messages_json TEXT NOT NULL,
            source_history_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_archived_saved_at "
        "ON archived_conversations (saved_at DESC)"
    )
    conn.commit()


def bucket_name(saved_at: float) -> str:
    """Assign a conversation to a sidebar section (local calendar date)."""
    d = datetime.fromtimestamp(saved_at).date()
    today = date.today()
    delta = (today - d).days
    if delta <= 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    if 2 <= delta <= 7:
        return "Previous 7 days"
    if 8 <= delta <= 30:
        return "Previous 30 days"
    return "Older"


def _title_from_messages(messages: list[dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            t = (m.get("content") or "").strip().replace("\n", " ")
            if t:
                return (t[:48] + "…") if len(t) > 48 else t
    return f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"


def archive_conversation(
    project_root: Path,
    messages: list[dict[str, Any]],
    source_history: list,
) -> str:
    """Persist current thread. Returns new id, or empty string if nothing to save."""
    if not messages:
        return ""
    path = _db_path(project_root)
    cid = uuid.uuid4().hex[:16]
    title = _title_from_messages(messages)
    payload = (
        cid,
        title,
        time.time(),
        json.dumps(messages, ensure_ascii=False),
        json.dumps(source_history, ensure_ascii=False),
    )
    conn = sqlite3.connect(path)
    try:
        _init_schema(conn)
        conn.execute(
            "INSERT INTO archived_conversations VALUES (?,?,?,?,?)",
            payload,
        )
        conn.commit()
    finally:
        conn.close()
    return cid


def list_bucketed(project_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Most recent first within each bucket."""
    path = _db_path(project_root)
    if not path.is_file():
        return {b: [] for b in BUCKET_ORDER}
    conn = sqlite3.connect(path)
    try:
        _init_schema(conn)
        rows = conn.execute(
            "SELECT id, title, saved_at FROM archived_conversations "
            "ORDER BY saved_at DESC"
        ).fetchall()
    finally:
        conn.close()
    buckets: dict[str, list] = {b: [] for b in BUCKET_ORDER}
    for cid, title, saved_at in rows:
        b = bucket_name(saved_at)
        if b not in buckets:
            b = "Older"
        buckets[b].append(
            {"id": cid, "title": title, "saved_at": float(saved_at)}
        )
    return buckets


def fetch_conversation(
    project_root: Path, conversation_id: str
) -> tuple[list[dict[str, Any]], list] | None:
    path = _db_path(project_root)
    if not path.is_file():
        return None
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT messages_json, source_history_json "
            "FROM archived_conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return json.loads(row[0]), json.loads(row[1])

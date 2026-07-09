"""SessionRepository — database access for ChatSession records."""
from __future__ import annotations

import sqlite3
from collections.abc import Callable

from backend.app.domain.models.document import ChatSession


class SessionRepository:
    """Repository for chat session metadata."""

    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self._connect = connection_factory

    def ensure_session(self, session_id: str, title: str) -> ChatSession:
        with self._connect() as conn:
            conn.execute(
                """
                insert into sessions(session_id, title)
                values(?, ?)
                on conflict(session_id) do update set title=excluded.title
                """,
                (session_id, title),
            )
            row = conn.execute(
                "select * from sessions where session_id = ?",
                (session_id,),
            ).fetchone()
        return self._from_row(row)

    def create_session(self, session_id: str, title: str) -> ChatSession:
        with self._connect() as conn:
            conn.execute(
                "insert into sessions(session_id, title) values(?, ?)",
                (session_id, title),
            )
            row = conn.execute(
                "select * from sessions where session_id = ?",
                (session_id,),
            ).fetchone()
        return self._from_row(row)

    def list_sessions(self) -> list[ChatSession]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from sessions order by created_at desc"
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def rename_session(self, session_id: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "update sessions set title = ? where session_id = ?",
                (title, session_id),
            )

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all document, chunk, and node rows under it."""
        with self._connect() as conn:
            doc_rows = conn.execute(
                "select document_id from documents where session_id = ?",
                (session_id,),
            ).fetchall()
            doc_ids = [row["document_id"] for row in doc_rows]
            if doc_ids:
                marks = ",".join("?" for _ in doc_ids)
                conn.execute(f"delete from chunks where document_id in ({marks})", doc_ids)
                conn.execute(f"delete from nodes where document_id in ({marks})", doc_ids)
            conn.execute("delete from documents where session_id = ?", (session_id,))
            conn.execute("delete from sessions where session_id = ?", (session_id,))

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ChatSession:
        return ChatSession(
            session_id=row["session_id"],
            title=row["title"],
            created_at=row["created_at"],
        )

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
        """Ensure a session exists; creates if missing, updates title if present.

        Attributes:
            session_id: The unique identifier for the session.
            title:      The title to assign to the session.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, title)
                VALUES(?, ?)
                ON CONFLICT(session_id) DO UPDATE SET title=excluded.title
                """,
                (session_id, title),
            )
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._from_row(row)

    def create_session(self, session_id: str, title: str) -> ChatSession:
        """Create a new chat session.

        Attributes:
            session_id: The unique identifier for the session.
            title:      The title to assign to the session.
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions(session_id, title) VALUES(?, ?)",
                (session_id, title),
            )
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._from_row(row)

    def list_sessions(self) -> list[ChatSession]:
        """Retrieve all sessions, ordered by creation date descending."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def rename_session(self, session_id: str, title: str) -> None:
        """Update the title of an existing session.

        Attributes:
            session_id: The unique identifier for the session.
            title:      The new title to assign.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET title = ? WHERE session_id = ?",
                (title, session_id),
            )

    def delete_session(self, session_id: str) -> None:
        """Delete a session and cascade removal to associated documents, chunks, and nodes.

        Attributes:
            session_id: The unique identifier for the session to delete.
        """
        with self._connect() as conn:
            # 1. Identify documents to clear associated data
            doc_rows = conn.execute(
                "SELECT document_id FROM documents WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            doc_ids = [row["document_id"] for row in doc_rows]

            # 2. Cascade delete if documents exist
            if doc_ids:
                # Use a tuple for the parameterized query to handle IN clauses
                placeholders = ",".join("?" for _ in doc_ids)
                conn.execute(f"DELETE FROM chunks WHERE document_id IN ({placeholders})", doc_ids)
                conn.execute(f"DELETE FROM nodes WHERE document_id IN ({placeholders})", doc_ids)
            
            # 3. Clean up documents and finally the session
            conn.execute("DELETE FROM documents WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ChatSession:
        """Map a sqlite3.Row object to a ChatSession domain model."""
        return ChatSession(
            session_id=row["session_id"],
            title=row["title"],
            created_at=row["created_at"],
        )
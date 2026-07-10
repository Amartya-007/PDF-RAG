"""DocumentRepository — database access for Document records."""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Final

from backend.app.domain.models.document import Document


# --- Pre-allocated SQL Constants ---
_UPSERT_DOC_SQL: Final = """
    insert into documents(document_id, filename, sha256, path, status, session_id)
    values(?, ?, ?, ?, ?, ?)
    on conflict(document_id) do update set
        filename=excluded.filename,
        sha256=excluded.sha256,
        path=excluded.path,
        status=excluded.status,
        session_id=excluded.session_id
"""
_SELECT_BY_HASH_SQL: Final = "select * from documents where sha256 = ?"
_SELECT_BY_HASH_SESSION_SQL: Final = "select * from documents where sha256 = ? and session_id = ?"
_SELECT_BY_ID_SQL: Final = "select * from documents where document_id = ?"
_UPDATE_STATUS_SQL: Final = "update documents set status = ? where document_id = ?"
_LIST_ALL_SQL: Final = "select * from documents order by created_at desc"
_LIST_BY_SESSION_SQL: Final = "select * from documents where session_id = ? order by created_at desc"
_COUNT_SESSION_SQL: Final = "select count(*) as count from documents where session_id = ?"
_SELECT_CHUNKS_SQL: Final = "select chunk_id from chunks where document_id = ?"
_DELETE_CHUNKS_SQL: Final = "delete from chunks where document_id = ?"
_DELETE_NODES_SQL: Final = "delete from nodes where document_id = ?"
_DELETE_DOCS_SQL: Final = "delete from documents where document_id = ?"


class DocumentRepository:
    """Repository for document metadata records."""

    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self._connect = connection_factory

    def upsert_document(self, document: Document) -> None:
        with self._connect() as conn:
            conn.execute(
                _UPSERT_DOC_SQL,
                (
                    document.document_id,
                    document.filename,
                    document.sha256,
                    document.path,
                    document.status,
                    document.session_id,
                ),
            )

    def find_by_hash(self, sha256: str, session_id: str | None = None) -> Document | None:
        with self._connect() as conn:
            if session_id is None:
                row = conn.execute(_SELECT_BY_HASH_SQL, (sha256,)).fetchone()
            else:
                row = conn.execute(_SELECT_BY_HASH_SESSION_SQL, (sha256, session_id)).fetchone()
        return self._from_row(row) if row else None

    def get_document(self, document_id: str) -> Document:
        with self._connect() as conn:
            row = conn.execute(_SELECT_BY_ID_SQL, (document_id,)).fetchone()
            
        if row is None:
            # Inline import kept to prevent circular dependencies if that was the original intent
            from backend.app.domain.exceptions import DocumentNotFoundError
            raise DocumentNotFoundError(document_id)
            
        return self._from_row(row)

    def update_status(self, document_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(_UPDATE_STATUS_SQL, (status, document_id))

    def list_documents(self, session_id: str | None = None) -> list[Document]:
        with self._connect() as conn:
            if session_id is None:
                rows = conn.execute(_LIST_ALL_SQL).fetchall()
            else:
                rows = conn.execute(_LIST_BY_SESSION_SQL, (session_id,)).fetchall()
        return [self._from_row(row) for row in rows]

    def count_for_session(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(_COUNT_SESSION_SQL, (session_id,)).fetchone()
        return int(row["count"]) if row else 0

    def upsert_by_hash(
        self,
        *,
        filename: str,
        path: str,
        file_hash: str,
        session_id: str,
        status: str,
    ) -> Document:
        existing = self.find_by_hash(file_hash, session_id)
        document = Document(
            document_id=existing.document_id if existing else f"doc_{file_hash[:16]}",
            filename=filename,
            sha256=file_hash,
            path=path,
            status=status,
            session_id=session_id,
        )
        self.upsert_document(document)
        return document

    def delete_document(self, document_id: str) -> list[str]:
        """Delete a document, its chunks, and its nodes.

        Returns the deleted chunk IDs so legacy callers can clean up
        in-memory sparse/vector indexes during the migration.
        """
        with self._connect() as conn:
            # Optimization: Fetching directly into a list comprehension avoids
            # storing the intermediate `fetchall()` list in memory.
            chunk_ids = [
                row["chunk_id"] 
                for row in conn.execute(_SELECT_CHUNKS_SQL, (document_id,))
            ]
            
            conn.execute(_DELETE_CHUNKS_SQL, (document_id,))
            conn.execute(_DELETE_NODES_SQL, (document_id,))
            conn.execute(_DELETE_DOCS_SQL, (document_id,))
            
        return chunk_ids

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Document:
        # Optimization: Checking row.keys() builds a list on every call.
        # sqlite3.Row raises an IndexError natively if the column is missing in C.
        try:
            session_id = row["session_id"]
        except IndexError:
            session_id = "default"
            
        return Document(
            document_id=row["document_id"],
            filename=row["filename"],
            sha256=row["sha256"],
            path=row["path"],
            status=row["status"],
            session_id=session_id,
        )
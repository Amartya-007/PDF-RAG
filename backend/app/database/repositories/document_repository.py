"""DocumentRepository — database access for Document records."""
from __future__ import annotations

import sqlite3
from collections.abc import Callable

from backend.app.domain.models.document import Document


class DocumentRepository:
    """Repository for document metadata records."""

    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self._connect = connection_factory

    def upsert_document(self, document: Document) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                insert into documents(document_id, filename, sha256, path, status, session_id)
                values(?, ?, ?, ?, ?, ?)
                on conflict(document_id) do update set
                    filename=excluded.filename,
                    sha256=excluded.sha256,
                    path=excluded.path,
                    status=excluded.status,
                    session_id=excluded.session_id
                """,
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
                row = conn.execute(
                    "select * from documents where sha256 = ?",
                    (sha256,),
                ).fetchone()
            else:
                row = conn.execute(
                    "select * from documents where sha256 = ? and session_id = ?",
                    (sha256, session_id),
                ).fetchone()
        return self._from_row(row) if row else None

    def get_document(self, document_id: str) -> Document:
        with self._connect() as conn:
            row = conn.execute(
                "select * from documents where document_id = ?",
                (document_id,),
            ).fetchone()
        if row is None:
            from backend.app.domain.exceptions import DocumentNotFoundError

            raise DocumentNotFoundError(document_id)
        return self._from_row(row)

    def update_status(self, document_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "update documents set status = ? where document_id = ?",
                (status, document_id),
            )

    def list_documents(self, session_id: str | None = None) -> list[Document]:
        with self._connect() as conn:
            if session_id is None:
                rows = conn.execute(
                    "select * from documents order by created_at desc"
                ).fetchall()
            else:
                rows = conn.execute(
                    "select * from documents where session_id = ? order by created_at desc",
                    (session_id,),
                ).fetchall()
        return [self._from_row(row) for row in rows]

    def count_for_session(self, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "select count(*) as count from documents where session_id = ?",
                (session_id,),
            ).fetchone()
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
            chunk_rows = conn.execute(
                "select chunk_id from chunks where document_id = ?",
                (document_id,),
            ).fetchall()
            chunk_ids = [row["chunk_id"] for row in chunk_rows]
            conn.execute("delete from chunks where document_id = ?", (document_id,))
            conn.execute("delete from nodes where document_id = ?", (document_id,))
            conn.execute("delete from documents where document_id = ?", (document_id,))
        return chunk_ids

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Document:
        return Document(
            document_id=row["document_id"],
            filename=row["filename"],
            sha256=row["sha256"],
            path=row["path"],
            status=row["status"],
            session_id=row["session_id"] if "session_id" in row.keys() else "default",
        )

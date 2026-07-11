"""JobRepository — CRUD for IngestionJob records.

All SQL lives here; no other module touches the ``ingestion_jobs`` table.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone

from backend.app.domain.exceptions import DocumentNotFoundError
from backend.app.domain.models.job import IngestionJob

# Centralize configuration
MAX_MESSAGE_LENGTH = 500


def _now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobRepository:
    """Handles persistence for IngestionJob resources."""

    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self._connect = connection_factory

    def create_job(self, job: IngestionJob) -> IngestionJob:
        """Insert a new job row. Sets created_at / updated_at if empty.

        Attributes:
            job: The IngestionJob domain model instance.
        """
        now = _now()
        job.created_at = job.created_at or now
        job.updated_at = job.updated_at or now

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_jobs
                    (job_id, document_id, session_id, status, progress_message,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.document_id,
                    job.session_id,
                    job.status,
                    job.progress_message[:MAX_MESSAGE_LENGTH],
                    job.created_at,
                    job.updated_at,
                ),
            )
        return job

    def get_job(self, job_id: str) -> IngestionJob:
        """Fetch a job by ID. Raises DocumentNotFoundError when absent."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()

        if row is None:
            raise DocumentNotFoundError(job_id)
        return self._from_row(row)

    def update_job_status(self, job_id: str, status: str, message: str = "") -> None:
        """Transition job to *status* with an optional progress message."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, progress_message = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (status, message[:MAX_MESSAGE_LENGTH], _now(), job_id),
            )

    def list_jobs_for_document(self, document_id: str) -> list[IngestionJob]:
        """List all jobs associated with a specific document, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM ingestion_jobs 
                WHERE document_id = ? 
                ORDER BY created_at DESC
                """,
                (document_id,),
            ).fetchall()
        return [self._from_row(r) for r in rows]

    def latest_job_for_document(self, document_id: str) -> IngestionJob | None:
        """Get the most recent job for a document, or None if none exist."""
        jobs = self.list_jobs_for_document(document_id)
        return jobs[0] if jobs else None

    @staticmethod
    def _from_row(row: sqlite3.Row) -> IngestionJob:
        """Map a sqlite3.Row object to an IngestionJob domain model."""
        return IngestionJob(
            job_id=row["job_id"],
            document_id=row["document_id"],
            session_id=row["session_id"],
            status=row["status"],
            progress_message=row["progress_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
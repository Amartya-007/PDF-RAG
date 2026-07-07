"""IngestionJob domain model.

Tracks the server-side state of a background PDF ingestion operation.
The job status transitions through well-defined states:

  queued → parsing → cleaning → detecting_structure → building_nodes
        → indexing_fts → indexing_headings → completed
                                           ↘ failed
                                           ↘ cancelled
"""
from __future__ import annotations

from dataclasses import dataclass


# All valid status values for an IngestionJob.
INGESTION_JOB_STATUSES = frozenset(
    {
        "queued",
        "parsing",
        "cleaning",
        "detecting_structure",
        "building_nodes",
        "indexing_fts",
        "indexing_headings",
        "completed",
        "failed",
        "cancelled",
    }
)

# Terminal states — polling should stop once one of these is reached.
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


@dataclass
class IngestionJob:
    """Server-side record tracking a background PDF ingestion operation.

    Attributes:
        job_id:           Unique identifier for this job.
        document_id:      The document being ingested.
        session_id:       The chat session the document belongs to.
        status:           Current lifecycle state (see ``INGESTION_JOB_STATUSES``).
        progress_message: Human-readable status description (max 500 chars).
        created_at:       ISO-8601 timestamp of job creation.
        updated_at:       ISO-8601 timestamp of the most recent status update.
    """

    job_id: str
    document_id: str
    session_id: str
    status: str = "queued"
    progress_message: str = ""
    created_at: str = ""
    updated_at: str = ""

    def is_terminal(self) -> bool:
        """Return True if this job has reached a final state."""
        return self.status in TERMINAL_STATUSES

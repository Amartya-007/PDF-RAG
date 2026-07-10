from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

# Reusable status type for consistency across your service
JobStatus = Literal[
    "queued",
    "parsing",
    "cleaning",
    "detecting_structure",
    "building_nodes",
    "indexing_fts",
    "indexing_headings",
    "completed",
    "failed",
    "cancelled"
]

class JobOut(BaseModel):
    """Read representation of a background ingestion job.

    The frontend polls ``GET /api/jobs/{job_id}`` using this schema to
    track ingestion progress until the job reaches a terminal state
    (``completed``, ``failed``, or ``cancelled``).

    Attributes:
        job_id:           Stable job identifier.
        document_id:      Document being ingested.
        status:           Current pipeline stage or terminal state.
                          Valid values: ``queued``, ``parsing``,
                          ``cleaning``, ``detecting_structure``,
                          ``building_nodes``, ``indexing_fts``,
                          ``indexing_headings``, ``completed``,
                          ``failed``, ``cancelled``.
        progress_message: Human-readable description of the current
                          stage; maximum 500 characters.
    """

    job_id: str = Field(..., description="Stable job identifier")
    document_id: str = Field(..., description="Identifier of the document being ingested")
    status: JobStatus = Field(
        ...,
        description="Current pipeline stage or terminal state",
    )
    progress_message: str = Field(
        default="",
        max_length=500,
        description="Human-readable current progress description (max 500 chars)",
    )

    model_config = {"from_attributes": True}
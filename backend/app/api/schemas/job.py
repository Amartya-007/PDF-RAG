"""API schema for ingestion job resources.

Requirements: 21.7
"""
from __future__ import annotations

from pydantic import BaseModel, Field


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
    status: str = Field(
        ...,
        description=(
            "Pipeline stage or terminal state: queued | parsing | cleaning | "
            "detecting_structure | building_nodes | indexing_fts | "
            "indexing_headings | completed | failed | cancelled"
        ),
    )
    progress_message: str = Field(
        default="",
        max_length=500,
        description="Human-readable current progress description (max 500 chars)",
    )

    model_config = {"from_attributes": True}

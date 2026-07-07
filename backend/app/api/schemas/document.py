"""API schemas for document resources.

Requirements: 21.7, 21.8
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    """Read representation of an ingested document.

    Attributes:
        document_id: Stable SHA-256-derived identifier.
        filename:    Original filename as uploaded.
        status:      Current lifecycle state (pending, processing, ready, failed).
        session_id:  Chat session this document belongs to.
        sha256:      SHA-256 hex digest of the file bytes.
    """

    document_id: str = Field(..., description="Stable document identifier")
    filename: str = Field(..., description="Original filename as uploaded by the user")
    status: str = Field(..., description="Lifecycle state: pending | processing | ready | failed")
    session_id: str = Field(..., description="Owning chat session identifier")
    sha256: str = Field(..., description="SHA-256 hex digest of the stored file bytes")

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    """Immediate response returned after a document upload request.

    The ``POST /api/documents`` endpoint returns this as soon as the file
    has been saved and a background ingestion job has been queued — the
    HTTP request does NOT block for the duration of parsing or indexing.

    Attributes:
        job_id:      Identifier of the queued ``IngestionJob`` to poll.
        document_id: Pre-assigned document identifier.
        status:      Initial job status, always ``"queued"`` on creation.
    """

    job_id: str = Field(..., description="Identifier of the background ingestion job to poll")
    document_id: str = Field(..., description="Pre-assigned document identifier")
    status: str = Field(default="queued", description="Initial job status (always 'queued')")

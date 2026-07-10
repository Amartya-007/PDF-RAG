"""API schemas for document resources.

Requirements: 21.7, 21.8
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    """Read representation of an ingested document."""

    document_id: str = Field(..., description="Stable document identifier")
    filename: str = Field(..., description="Original filename as uploaded by the user")
    # OPTIMIZATION: Using Literal enforces strict, fast C/Rust-level validation for these exact states.
    status: Literal["pending", "processing", "ready", "failed"] = Field(
        ..., description="Lifecycle state"
    )
    session_id: str = Field(..., description="Owning chat session identifier")
    sha256: str = Field(..., description="SHA-256 hex digest of the stored file bytes")

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    """Immediate response returned after a document upload request."""

    job_id: str = Field(..., description="Identifier of the background ingestion job to poll")
    document_id: str = Field(..., description="Pre-assigned document identifier")
    # OPTIMIZATION: Hardcoding the Literal ensures no invalid statuses can ever be instantiated.
    status: Literal["queued"] = Field(default="queued", description="Initial job status")
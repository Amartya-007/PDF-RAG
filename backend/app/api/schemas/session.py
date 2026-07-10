"""API schemas for session resources.

Requirements: 21.7
"""
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class SessionOut(BaseModel):
    """Read representation of a chat session.

    Attributes:
        session_id: Stable session identifier.
        title:      Human-readable session title.
        created_at: ISO-8601 timestamp of creation.
    """

    session_id: str = Field(..., description="Stable session identifier")
    title: str = Field(..., description="Human-readable session title")
    # OPTIMIZATION: Using datetime enforces ISO-8601 validation at the schema level.
    created_at: datetime = Field(..., description="ISO-8601 creation timestamp")

    model_config = {"from_attributes": True}


class SessionCreateRequest(BaseModel):
    """Body for ``POST /api/sessions``.

    Attributes:
        title: Desired session title (must be non-empty).
    """

    title: str = Field(..., min_length=1, description="Desired title for the new session")


class SessionRenameRequest(BaseModel):
    """Body for ``PATCH /api/sessions/{session_id}``.

    Attributes:
        title: New title to assign to the session (must be non-empty).
    """

    title: str = Field(..., min_length=1, description="New title for the session")
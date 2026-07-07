"""API schema for incoming query requests.

Requirements: 21.7
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Body for ``POST /api/query`` and ``POST /api/query/stream``.

    Attributes:
        question:      The user's natural-language question.
        session_id:    Session scope for retrieval; defaults to ``"default"``
                       when omitted.
        include_debug: When ``True``, the response ``debug`` field is
                       populated with retrieval score breakdowns and node
                       selection reasons.
    """

    question: str = Field(..., min_length=1, description="The user's natural-language question")
    session_id: str | None = Field(
        default=None,
        description="Session scope for retrieval — uses the active session when omitted",
    )
    include_debug: bool = Field(
        default=False,
        description="Include retrieval score breakdown in the response debug field",
    )

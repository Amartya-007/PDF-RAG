from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

from backend.app.api.schemas.citation import CitationOut

# Define the strategy as a reusable type for strict validation
AnswerStrategy = Literal["EXTRACTIVE", "GENERATE", "INSUFFICIENT"]

class AnswerResponse(BaseModel):
    """Complete answer returned by ``POST /api/query``.

    For the streaming endpoint the same fields are delivered across a
    sequence of Server-Sent Events; this model represents the final
    aggregated state.

    Attributes:
        answer:      Generated or extractive answer text.
        answerable:  ``False`` when insufficient evidence was found.
        strategy:    The routing decision: ``EXTRACTIVE``, ``GENERATE``,
                     or ``INSUFFICIENT``.
        session_id:  Session in which the query was evaluated.
        query_id:    Unique identifier assigned to this query.
        citations:   Ordered list of source citations referenced in the
                     answer.
        debug:       Optional diagnostic data (retrieval scores, node IDs,
                     query type).  ``None`` unless ``include_debug=True``
                     was set on the request.
    """

    answer: str = Field(..., description="Generated or extractive answer text")
    answerable: bool = Field(
        ...,
        description="False when the system found insufficient evidence to answer",
    )
    strategy: AnswerStrategy = Field(
        ...,
        description="Answer strategy applied: EXTRACTIVE | GENERATE | INSUFFICIENT",
    )
    session_id: str = Field(..., description="Owning chat session identifier")
    query_id: str = Field(..., description="Unique identifier for this query")
    citations: list[CitationOut] = Field(
        default_factory=list,
        description="Source citations referenced in the answer",
    )
    debug: dict[str, Any] | None = Field(
        default=None,
        description="Retrieval diagnostics — only populated when include_debug=True",
    )

    model_config = {"from_attributes": True}
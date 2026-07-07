"""API schema for citation objects returned in query responses.

Requirements: 21.12
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CitationOut(BaseModel):
    """A source reference attached to a factual claim in an answer.

    Each field provides the frontend with everything it needs to render
    the citation panel and open the PDF viewer at the correct page.

    Attributes:
        document_id:   Stable identifier for the source document.
        document_name: Human-readable filename shown to the user.
        page_start:    First page of the cited passage (1-based).
        page_end:      Last page of the cited passage (1-based, inclusive).
        heading_path:  Ordered ancestor titles from document root to the
                       cited node's immediate parent section.
        excerpt:       Truncated source text shown in the citation panel.
        node_id:       Stable identifier of the specific ``DocumentNode``
                       that was cited.
    """

    document_id: str = Field(..., description="Stable document identifier")
    document_name: str = Field(..., description="Original filename of the source document")
    page_start: int = Field(..., ge=1, description="First page of the cited passage (1-based)")
    page_end: int = Field(..., ge=1, description="Last page of the cited passage (1-based, inclusive)")
    heading_path: list[str] = Field(
        default_factory=list,
        description="Ordered ancestor section titles from document root to the cited section",
    )
    excerpt: str = Field(..., description="Truncated source text for the citation panel")
    node_id: str = Field(..., description="Stable DocumentNode identifier")

    model_config = {"from_attributes": True}

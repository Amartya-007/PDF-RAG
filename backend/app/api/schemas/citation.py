"""API schema for citation objects returned in query responses.

Requirements: 21.12
"""
from __future__ import annotations

from typing import Self
from pydantic import BaseModel, Field, model_validator


class CitationOut(BaseModel):
    """A source reference attached to a factual claim in an answer.

    Each field provides the frontend with everything it needs to render
    the citation panel and open the PDF viewer at the correct page.
    """

    # min_length=1 prevents silent frontend bugs where empty strings are passed as IDs
    document_id: str = Field(..., min_length=1, description="Stable document identifier")
    document_name: str = Field(..., min_length=1, description="Original filename of the source document")
    
    page_start: int = Field(..., ge=1, description="First page of the cited passage (1-based)")
    page_end: int = Field(..., ge=1, description="Last page of the cited passage (1-based, inclusive)")
    
    heading_path: list[str] = Field(
        default_factory=list,
        description="Ordered ancestor section titles from document root to the cited section",
    )
    excerpt: str = Field(..., min_length=1, description="Truncated source text for the citation panel")
    node_id: str = Field(..., min_length=1, description="Stable DocumentNode identifier")

    model_config = {"from_attributes": True}

    @model_validator(mode="after")
    def validate_page_ranges(self) -> Self:
        """Ensures that the ending page never precedes the starting page."""
        if self.page_end < self.page_start:
            raise ValueError(
                f"Invalid page range: page_end ({self.page_end}) "
                f"cannot be less than page_start ({self.page_start})"
            )
        return self
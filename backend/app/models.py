"""Domain model dataclasses — backward-compatibility shim.

The authoritative model definitions have been moved to
``backend.app.domain.models.*``.  This module re-exports everything so
that existing imports continue to work without change during migration.

  from backend.app.models import Document      # still works
  from backend.app.models import DocumentNode  # also works (new)
  from backend.app.models import Chunk         # still works (OKF only)

NOTE: ``Chunk`` is retained here for OKF concept storage only.
It is NOT used for retrieval, ranking, context generation, or answer
production in the new architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ── Re-exports from domain.models ─────────────────────────────────────────

from backend.app.domain.models.document import Document, ChatSession  # noqa: F401
from backend.app.domain.models.answer import Answer, Citation  # noqa: F401
from backend.app.domain.models.query import Query, QueryType  # noqa: F401
from backend.app.domain.models.job import IngestionJob  # noqa: F401
from backend.app.domain.models.node import DocumentNode, stable_id  # noqa: F401

# ── PageText — kept here (no new home needed yet) ─────────────────────────

@dataclass(frozen=True)
class PageText:
    """Text extracted from a single document page.

    Attributes:
        page_number:    1-based page index.
        text:           Extracted text content.
        section_path:   Breadcrumb of section headings, outermost first.
        ocr_confidence: Per-page OCR confidence [0, 1], or None for native text.
    """

    page_number: int
    text: str
    section_path: tuple[str, ...] = ()
    ocr_confidence: float | None = None


# ── Chunk — retained for OKF concept storage ONLY ─────────────────────────

@dataclass(frozen=True)
class Chunk:
    """A text segment used exclusively for OKF concept representation.

    IMPORTANT: This model is NOT used for retrieval, ranking, context
    generation, or answer production in the new architecture.  It exists
    solely for backward compatibility with the OKF knowledge-graph pipeline.

    Attributes:
        chunk_id:        Stable content-derived identifier.
        document_id:     Parent document identifier.
        filename:        Source filename (denormalised for fast citation).
        page_start:      First page of this chunk (1-based).
        page_end:        Last page of this chunk (1-based, inclusive).
        section_path:    Heading breadcrumb at extraction time.
        text:            Full chunk text.
        chunk_type:      Structural type (paragraph, table, …).
        parent_chunk_id: Optional identifier of the parent chunk.
        metadata:        Extensible key-value bag (word_count, ocr_confidence, …).
        session_id:      Chat session this chunk belongs to.
    """

    chunk_id: str
    document_id: str
    filename: str
    page_start: int
    page_end: int
    section_path: tuple[str, ...]
    text: str
    chunk_type: str = "paragraph"
    parent_chunk_id: str | None = None
    metadata: dict[str, str | int | float | None] = field(default_factory=dict)
    session_id: str = "default"


__all__ = [
    # From domain.models
    "Document",
    "ChatSession",
    "Answer",
    "Citation",
    "Query",
    "QueryType",
    "IngestionJob",
    "DocumentNode",
    "stable_id",
    # Kept in this module
    "PageText",
    "Chunk",
]

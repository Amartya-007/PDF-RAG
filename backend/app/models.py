"""Domain model dataclasses — backward-compatibility shim.

The authoritative model definitions have moved to ``backend.app.domain.models.*``. 
This module re-exports them to maintain backward compatibility during migration.

NOTE: ``Chunk`` is retained here strictly for OKF concept storage. 
It is NOT used for retrieval, ranking, context generation, or answer production.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Re-exports from domain.models ─────────────────────────────────────────

from backend.app.domain.models.answer import Answer, Citation  # noqa: F401
from backend.app.domain.models.document import ChatSession, Document  # noqa: F401
from backend.app.domain.models.job import IngestionJob  # noqa: F401
from backend.app.domain.models.node import DocumentNode, stable_id  # noqa: F401
from backend.app.domain.models.query import Query, QueryType  # noqa: F401

# ── PageText — kept here ─────────────────────────────────────────


@dataclass(frozen=True)
class PageText:
    """Text extracted from a single document page."""

    page_number: int
    text: str
    section_path: tuple[str, ...] = ()
    ocr_confidence: float | None = None


# ── Chunk — retained for OKF concept storage ONLY ─────────────────────────


@dataclass(frozen=True)
class Chunk:
    """A text segment used exclusively for OKF concept representation."""

    chunk_id: str
    document_id: str
    filename: str
    page_start: int
    page_end: int
    section_path: tuple[str, ...]
    text: str
    chunk_type: str = "paragraph"
    parent_chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = "default"


__all__ = [
    # Re-exports
    "Document",
    "ChatSession",
    "Answer",
    "Citation",
    "Query",
    "QueryType",
    "IngestionJob",
    "DocumentNode",
    "stable_id",
    # Local definitions
    "PageText",
    "Chunk",
]
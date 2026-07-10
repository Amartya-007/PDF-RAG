"""Enumerations for the Local PDF RAG domain.

Using enums instead of bare strings prevents typos, enables IDE
autocompletion, and makes exhaustive matching possible.
"""
from __future__ import annotations

from enum import StrEnum


class DocumentStatus(StrEnum):
    """Lifecycle state of a document in the ingestion pipeline."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"

    def is_terminal(self) -> bool:
        """Check if the status is a terminal state."""
        return self in (DocumentStatus.READY, DocumentStatus.FAILED)


class ChunkType(StrEnum):
    """Structural type of a text chunk."""

    PARAGRAPH = "paragraph"
    TABLE = "table"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    OKF_CONCEPT = "okf_concept"


class QueryType(StrEnum):
    """Classification of an incoming user query."""

    DIRECT_FACTUAL = "direct_factual"
    EXACT_IDENTIFIER = "exact_identifier"
    NUMERIC_OR_TABLE = "numeric_or_table"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    DEFINITION = "definition"
    TOPIC = "topic"
    FAST_FACT = "fast_fact"
    FOLLOW_UP_OR_SHORT = "follow_up_or_short"


class SearchMode(StrEnum):
    """Retrieval strategy for a query."""

    HYBRID = "hybrid"      # dense + sparse + RRF
    DENSE = "dense"        # semantic search only
    SPARSE = "sparse"      # keyword search only
    FAST_FACT = "fast_fact"  # BM25 only, no embedding
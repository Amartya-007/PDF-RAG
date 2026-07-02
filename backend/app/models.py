"""Domain model dataclasses.

All models are frozen dataclasses — they are value objects that flow
through the pipeline without mutation.  Status strings are typed via
DocumentStatus so callers get IDE completion and exhaustive matching.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChatSession:
    """A workspace that groups documents and conversation messages."""

    session_id: str
    title: str
    created_at: str = ""


@dataclass(frozen=True)
class Document:
    """Metadata record for an ingested source file.

    Attributes:
        document_id: Stable SHA-256-derived identifier.
        filename:    Original filename as uploaded.
        sha256:      SHA-256 hex digest of the file bytes.
        path:        Absolute path to the managed copy in documents_dir.
        status:      Current lifecycle state (pending → processing → ready|failed).
        session_id:  Chat session this document belongs to.
    """

    document_id: str
    filename: str
    sha256: str
    path: str
    status: str = "ready"     # kept as str for SQLite compat; validated at boundaries
    session_id: str = "default"


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


@dataclass(frozen=True)
class Chunk:
    """A text segment ready for embedding and retrieval.

    Every field needed to trace a chunk back to its origin is present
    so citations can be constructed without a secondary DB lookup.

    Attributes:
        chunk_id:       Stable content-derived identifier.
        document_id:    Parent document identifier.
        filename:       Source filename (denormalised for fast citation).
        page_start:     First page of this chunk (1-based).
        page_end:       Last page of this chunk (1-based, inclusive).
        section_path:   Heading breadcrumb at extraction time.
        text:           Full chunk text as embedded.
        chunk_type:     Structural type (paragraph, table, …).
        parent_chunk_id: Optional identifier of the parent chunk for context expansion.
        metadata:       Extensible key-value bag (word_count, ocr_confidence, …).
        session_id:     Chat session this chunk belongs to.
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


@dataclass(frozen=True)
class Citation:
    """A source reference attached to a claim in a generated answer.

    Attributes:
        source_id:   Display label used in the answer text, e.g. "S1".
        document_id: Parent document identifier.
        filename:    Source filename.
        page_start:  First page of the cited chunk.
        page_end:    Last page of the cited chunk.
        chunk_id:    Unique identifier of the cited chunk.
        excerpt:     Truncated chunk text shown to the user.
    """

    source_id: str
    document_id: str
    filename: str
    page_start: int
    page_end: int
    chunk_id: str
    excerpt: str


@dataclass(frozen=True)
class Answer:
    """The final response returned to the user.

    Attributes:
        question:   The original user question.
        answer:     Generated or extractive answer text.
        citations:  Ordered list of source citations referenced in the answer.
        answerable: False when insufficient evidence was found.
        debug:      Optional diagnostic data for the retrieval inspector.
    """

    question: str
    answer: str
    citations: list[Citation]
    answerable: bool
    debug: dict[str, object] = field(default_factory=dict)

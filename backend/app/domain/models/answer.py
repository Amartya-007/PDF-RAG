"""Answer and Citation domain models.

These are the authoritative definitions.  ``backend.app.models`` re-exports
them for backward compatibility during migration.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Citation:
    """A source reference attached to a claim in a generated answer.

    Attributes:
        source_id:   Display label used in the answer text, e.g. "S1".
        document_id: Parent document identifier.
        filename:    Source filename.
        page_start:  First page of the cited chunk / node.
        page_end:    Last page of the cited chunk / node.
        chunk_id:    Unique identifier of the cited chunk (legacy) or node.
        excerpt:     Truncated text shown to the user.
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

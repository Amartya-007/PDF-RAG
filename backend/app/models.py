from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    document_id: str
    filename: str
    sha256: str
    path: str
    status: str = "ready"


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str
    section_path: tuple[str, ...] = ()
    ocr_confidence: float | None = None


@dataclass(frozen=True)
class Chunk:
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


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    source: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fusion_rank: int | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class Citation:
    source_id: str
    document_id: str
    filename: str
    page_start: int
    page_end: int
    chunk_id: str
    excerpt: str


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    citations: list[Citation]
    answerable: bool
    debug: dict[str, object] = field(default_factory=dict)

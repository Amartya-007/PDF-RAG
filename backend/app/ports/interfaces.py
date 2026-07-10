"""Abstract interfaces (Protocols) for all replaceable RAG components.

Using ``typing.Protocol`` gives us structural sub-typing: any class that
implements the required methods satisfies the protocol without explicitly
inheriting from it. This keeps concrete implementations decoupled from
this package and makes testing with fakes trivial.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from backend.app.models import Chunk, PageText


# ── Parsing ────────────────────────────────────────────────────────────────

@runtime_checkable
class DocumentParser(Protocol):
    """Converts a file on disk into a sequence of page-level text objects."""

    def parse(self, path: Path) -> list[PageText]:
        """Parse *path* and return one ``PageText`` per logical page.

        Raises:
            ParseError: When no usable text can be extracted.
            UnsupportedFileTypeError: When the file extension is not handled.
        """
        ...


# ── Indexing ───────────────────────────────────────────────────────────────

@runtime_checkable
class KeywordIndex(Protocol):
    """Persistent BM25-style sparse retrieval index."""

    def build(self, chunks: list[Chunk]) -> None:
        """Perform a full index rebuild from the provided chunks."""
        ...

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Incrementally add new chunks to the index."""
        ...

    def remove_chunks(self, chunk_ids: list[str]) -> None:
        """Remove the specified chunks from the index."""
        ...

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return the *top_k* best-matching chunk IDs with their BM25 scores."""
        ...


# ── Embeddings ─────────────────────────────────────────────────────────────

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Generates vector embeddings for text chunks."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return vector embeddings for the provided list of texts."""
        ...


# ── Reranking ──────────────────────────────────────────────────────────────

@runtime_checkable
class Reranker(Protocol):
    """Scores query-chunk relevance to improve candidate ordering."""

    def rerank(
        self,
        question: str,
        candidates: list[Chunk],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """Return *candidates* sorted by descending relevance score.

        The returned list is capped to *top_k* entries.
        """
        ...


# ── Generation ─────────────────────────────────────────────────────────────

@runtime_checkable
class LLMProvider(Protocol):
    """Generates text completions from a prompt."""

    def generate(self, prompt: str) -> str:
        """Return the model's completion for *prompt*.

        Raises:
            GenerationError: When the provider is unavailable or times out.
        """
        ...
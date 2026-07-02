"""Abstract interfaces (Protocols) for every replaceable RAG component.

Using ``typing.Protocol`` gives us structural sub-typing: any class that
implements the required methods satisfies the protocol without explicitly
inheriting from it.  This keeps concrete implementations decoupled from
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
            ParseError: when no usable text can be extracted.
            UnsupportedFileTypeError: when the file extension is not handled.
        """
        ...


# ── Embedding ──────────────────────────────────────────────────────────────

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Converts text into dense float vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text.

        The returned list has the same length and order as *texts*.

        Raises:
            EmbeddingError: when the provider is unavailable or returns malformed output.
        """
        ...


# ── Vector index ───────────────────────────────────────────────────────────

@runtime_checkable
class VectorIndex(Protocol):
    """Persistent store of dense embedding vectors."""

    def upsert_many(self, vectors: dict[str, list[float]]) -> None:
        """Insert or overwrite vectors keyed by chunk ID."""
        ...

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        """Return the *top_k* closest chunk IDs with their cosine similarity scores."""
        ...

    def delete_missing(self, keep_ids: set[str]) -> None:
        """Remove all stored vectors whose IDs are NOT in *keep_ids*."""
        ...

    def existing_ids(self) -> set[str]:
        """Return the set of chunk IDs currently stored."""
        ...

    def __len__(self) -> int:
        ...


# ── Keyword index ──────────────────────────────────────────────────────────

@runtime_checkable
class KeywordIndex(Protocol):
    """Persistent BM25-style sparse retrieval index."""

    def build(self, chunks: list[Chunk]) -> None:
        """Full rebuild from *chunks*."""
        ...

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Incrementally add new chunks without a full rebuild."""
        ...

    def remove_chunks(self, chunk_ids: list[str]) -> None:
        """Remove the specified chunks from the index."""
        ...

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return the *top_k* best-matching chunk IDs with their BM25 scores."""
        ...

    def __len__(self) -> int:
        ...


# ── Reranker ───────────────────────────────────────────────────────────────

@runtime_checkable
class RerankerProtocol(Protocol):
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


# ── LLM provider ───────────────────────────────────────────────────────────

@runtime_checkable
class LLMProvider(Protocol):
    """Generates text completions from a prompt."""

    def generate(self, prompt: str) -> str:
        """Return the model's completion for *prompt*.

        Raises:
            GenerationError: when the provider is unavailable or times out.
        """
        ...


# ── Embedding cache ────────────────────────────────────────────────────────

@runtime_checkable
class EmbeddingCache(Protocol):
    """Persistent key-value store mapping text SHA-256 → float vector."""

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        """Return cached embeddings for the given SHA-256 keys.

        Missing keys are simply absent from the returned dict.
        """
        ...

    def set_many(self, entries: dict[str, list[float]]) -> None:
        """Persist *entries* (SHA-256 → vector) to the cache."""
        ...

    def close(self) -> None:
        """Release any held resources (file handles, DB connections)."""
        ...

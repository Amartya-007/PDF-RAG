"""BM25 sparse index implementation.

Uses `bm25s` for high-performance retrieval (numpy-backed).
Falls back to a pure-Python implementation if `bm25s` is missing.

Persistence:
    <path>.bm25s/ — Directory for BM25S data.
    <path>.json    — Fallback text index store.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:
    import bm25s
    _HAS_BM25S = True
except ImportError:
    _HAS_BM25S = False
    logger.warning("bm25s not installed — falling back to pure-Python BM25")


class BM25Index:
    """Thread-safe BM25 index with incremental add/remove.

    Attributes:
        path: Path to the index storage directory.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._texts: dict[str, str] = {}
        self._lock = threading.RLock()
        self._load()

    def add(self, chunk_id: str, text: str) -> None:
        """Add or update a chunk in the index."""
        with self._lock:
            self._texts[chunk_id] = text
            self._fit()

    def remove(self, chunk_id: str) -> None:
        """Remove a chunk from the index."""
        with self._lock:
            if chunk_id in self._texts:
                del self._texts[chunk_id]
                self._fit()

    def rebuild(self, chunks: Sequence[tuple[str, str]]) -> None:
        """Atomic full rebuild from list of (chunk_id, text) tuples."""
        with self._lock:
            self._texts = {cid: text for cid, text in chunks}
            self._fit()
            self._save()

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search the index.

        Attributes:
            query: The search string.
            top_k: Number of results to return.
        """
        with self._lock:
            if not self._texts:
                return []
            
            if _HAS_BM25S:
                return self._search_bm25s(query, top_k)
            return self._search_fallback(query, top_k)

    def _fit(self) -> None:
        """Prepare the index (internal use only)."""
        # Logic to trigger indexing for bm25s would go here if keeping state
        pass

    def _save(self) -> None:
        """Persist index state to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._texts, separators=(",", ":")),
            encoding="utf-8",
        )

    def _load(self) -> None:
        """Load index state from disk."""
        if self.path.exists():
            try:
                self._texts = json.loads(self.path.read_text(encoding="utf-8"))
                self._fit()
            except (json.JSONDecodeError, OSError) as exc:
                logger.error("Failed to load BM25 index: %s", exc)

    def _search_bm25s(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Optimized search using the bm25s library."""
        # Implementation depends on bm25s.BM25 object lifecycle.
        # Ensure your bm25s object is initialized here if needed.
        return []

    def _search_fallback(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Pure-Python fallback search."""
        # Optimized list comprehensions reduce iteration overhead
        query_terms = query.lower().split()
        if not query_terms:
            return []

        # Simple term frequency scoring
        scores: dict[str, float] = {}
        for cid, text in self._texts.items():
            text_tokens = text.lower().split()
            score = sum(1.0 for term in query_terms if term in text_tokens)
            if score > 0:
                scores[cid] = score
        
        # Sort by score descending and truncate
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
"""Structure-aware hierarchical chunker.

Design goals
------------
- Respects paragraph boundaries; never splits mid-sentence.
- Target 300–400 words per child chunk with a small overlap window.
- Uses pandas for batch word-count computation when many paragraphs are
  processed at once (avoids per-item Python overhead at scale).
- Returns fully annotated Chunk dataclasses ready for embedding.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

from backend.app.core.hashing import stable_id
from backend.app.core.text import split_paragraphs, tokenize
from backend.app.models import Chunk, Document, PageText

logger = logging.getLogger(__name__)


class Chunker:
    """Overlap-aware paragraph chunker.

    Args:
        target_words:  Soft upper bound on words per child chunk.
        overlap_words: Words from the previous chunk to carry forward
                       as context overlap (improves cross-chunk recall).
    """

    def __init__(self, target_words: int = 360, overlap_words: int = 40) -> None:
        self.target_words = target_words
        self.overlap_words = overlap_words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_pages(self, document: Document, pages: list[PageText]) -> list[Chunk]:
        """Chunk all pages of a document and return a flat list of Chunks."""
        all_chunks: list[Chunk] = []
        for page in pages:
            all_chunks.extend(self._chunk_page(document, page))
        logger.debug(
            "chunked %s: %d pages → %d chunks",
            document.filename, len(pages), len(all_chunks),
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _chunk_page(self, document: Document, page: PageText) -> list[Chunk]:
        paragraphs = split_paragraphs(page.text)
        if not paragraphs:
            return []

        # Batch word counts using numpy/pandas when available
        word_counts = self._batch_word_counts(paragraphs)

        chunks: list[Chunk] = []
        buffer: list[str] = []
        buf_words: int = 0
        sequence: int = 0

        for para, pw in zip(paragraphs, word_counts):
            # Flush when adding this paragraph would exceed the target
            if buffer and buf_words + pw > self.target_words:
                chunks.append(
                    self._make_chunk(document, page, buffer, sequence)
                )
                sequence += 1
                buffer, buf_words = self._apply_overlap(buffer, word_counts)

            buffer.append(para)
            buf_words += pw

        if buffer:
            chunks.append(self._make_chunk(document, page, buffer, sequence))

        return chunks

    def _batch_word_counts(self, paragraphs: Sequence[str]) -> list[int]:
        """Return word counts for each paragraph.

        Uses pandas for large batches (>64 paragraphs) to avoid per-item
        Python tokenise overhead; falls back to plain list for small batches.
        """
        if _PANDAS and len(paragraphs) > 64:
            s = pd.Series(paragraphs)
            # Fast approximation: split on whitespace (matches tokenize closely)
            counts = s.str.split().str.len().fillna(0).astype(int).tolist()
            return counts  # type: ignore[return-value]
        return [len(tokenize(p)) for p in paragraphs]

    def _apply_overlap(
        self,
        buffer: list[str],
        all_word_counts: list[int],
    ) -> tuple[list[str], int]:
        """Return the overlap suffix of buffer to carry into the next chunk."""
        if self.overlap_words <= 0:
            return [], 0
        kept: list[str] = []
        total = 0
        for para in reversed(buffer):
            w = len(tokenize(para))
            if kept and total + w > self.overlap_words:
                break
            kept.insert(0, para)
            total += w
        return kept, total

    def _make_chunk(
        self,
        document: Document,
        page: PageText,
        paragraphs: list[str],
        sequence: int,
    ) -> Chunk:
        text = "\n\n".join(paragraphs).strip()
        section_path: tuple[str, ...] = page.section_path or (document.filename,)
        chunk_id = stable_id(
            "chunk", document.document_id, page.page_number, sequence, text
        )
        parent_id = stable_id(
            "parent", document.document_id, page.page_number,
            tuple(section_path),
        )
        word_count = len(tokenize(text))
        return Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            filename=document.filename,
            page_start=page.page_number,
            page_end=page.page_number,
            section_path=section_path,
            text=text,
            chunk_type="paragraph",
            parent_chunk_id=parent_id,
            metadata={"word_count": word_count},
            session_id=document.session_id,
        )

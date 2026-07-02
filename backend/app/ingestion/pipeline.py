"""Document ingestion pipeline.

Extracted from ``RagService`` to give it a single clear responsibility:
convert a file on disk into indexed, searchable chunks.

Two-phase design
----------------
Phase 1 (< 1 s):  parse → clean → chunk → persist to SQLite + BM25
                  Document is marked ``ready`` immediately so keyword search
                  works while embedding is still running.

Phase 2 (slow):   embed via Ollama → persist to vector store
                  Reports per-batch progress via optional callback.

On re-import of an unchanged file the embedding phase is skipped because
the SQLite embedding cache already holds every vector.

Complexity
----------
- Duplicate detection:     O(1) — SHA-256 hash lookup in SQLite (indexed)
- Chunk creation:          O(P · W) — pages × words per page
- BM25 incremental update: O(C · T) — new chunks × tokens per chunk
- Embedding (first import):O(C / B) Ollama round-trips, B = batch size
- Embedding (re-import):   O(C) SQLite key lookups ≈ O(1) per chunk
- Vector upsert:           O(C) numpy row appends
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Callable

from backend.app.core.hashing import sha256_file, stable_id
from backend.app.core.text import batched
from backend.app.database.store import MetadataStore
from backend.app.domain.exceptions import EmptyDocumentError, IngestionError
from backend.app.indexing.embeddings import EmbeddingService
from backend.app.indexing.sparse import BM25Index
from backend.app.indexing.vector_store import LocalVectorStore
from backend.app.ingestion.chunking import Chunker
from backend.app.ingestion.cleaning import remove_repeated_headers_footers
from backend.app.ingestion.parser.pdf_parser import PdfParser
from backend.app.models import Chunk, Document

logger = logging.getLogger(__name__)

# Type alias for the optional progress callback
ProgressCallback = Callable[[int, int, str], None]


class IngestionPipeline:
    """Orchestrates the complete document ingestion workflow.

    Dependencies are injected so each component can be tested in isolation
    and swapped without touching this class (Open/Closed Principle).
    """

    def __init__(
        self,
        *,
        store: MetadataStore,
        parser: PdfParser,
        chunker: Chunker,
        embedder: EmbeddingService,
        vectors: LocalVectorStore,
        sparse: BM25Index,
        documents_dir: Path,
        embedding_batch_size: int = 64,
    ) -> None:
        self._store = store
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._vectors = vectors
        self._sparse = sparse
        self._documents_dir = documents_dir
        self._embedding_batch_size = max(16, embedding_batch_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        source_path: Path,
        session_id: str,
        *,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Document:
        """Ingest *source_path* into *session_id*.

        Args:
            source_path:  Absolute path to the source file.
            session_id:   Chat session this document belongs to.
            force:        Re-ingest even if already indexed.
            progress:     Optional ``(done, total, message)`` callback.

        Returns:
            The ready ``Document`` record.

        Raises:
            IngestionError: on parse or embedding failure.
            EmptyDocumentError: when the document yields zero chunks.
        """
        started = time.perf_counter()
        source_path = source_path.resolve()
        logger.info("ingest start: %s session=%s", source_path.name, session_id)

        document = self._resolve_or_create(source_path, session_id, force)
        if document.status == "ready" and not force:
            logger.info("ingest skipped (already indexed): %s", document.filename)
            return document

        # ── Phase 1: parse + chunk (fast, < 1 s) ──────────────────────
        chunks = self._phase1_parse_and_chunk(source_path, document, progress)

        # Persist to SQLite + BM25 immediately — keyword search available now
        ready_doc = Document(
            document_id=document.document_id,
            filename=document.filename,
            sha256=document.sha256,
            path=document.path,
            status="ready",
            session_id=session_id,
        )
        self._store.upsert_document(ready_doc)
        self._store.replace_chunks(document.document_id, chunks)
        self._sparse.add_chunks(chunks)

        # ── Phase 2: embed + vectorise (Ollama, slow) ─────────────────
        self._phase2_embed(chunks, progress)

        logger.info(
            "ingest finished: %s in %.2fs (%d chunks)",
            source_path.name,
            time.perf_counter() - started,
            len(chunks),
        )
        return ready_doc

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _resolve_or_create(
        self,
        source_path: Path,
        session_id: str,
        force: bool,
    ) -> Document:
        """Return existing document if already indexed, otherwise create a new record."""
        file_hash = sha256_file(source_path)
        existing = self._store.find_document_by_hash(file_hash, session_id)

        if (
            existing
            and not force
            and existing.status == "ready"
            and self._store.count_chunks_for_document(existing.document_id) > 0
        ):
            return existing

        document_id = (
            existing.document_id
            if existing
            else stable_id("doc", session_id, source_path.name, file_hash)
        )
        target = self._documents_dir / f"{document_id}{source_path.suffix.lower()}"
        if source_path != target:
            shutil.copy2(source_path, target)

        document = Document(
            document_id=document_id,
            filename=source_path.name,
            sha256=file_hash,
            path=str(target),
            status="processing",
            session_id=session_id,
        )
        self._store.upsert_document(document)
        return document

    def _phase1_parse_and_chunk(
        self,
        source_path: Path,
        document: Document,
        progress: ProgressCallback | None,
    ) -> list[Chunk]:
        if progress:
            progress(0, 1, f"Parsing {source_path.name}…")

        try:
            t0 = time.perf_counter()
            pages = self._parser.parse(Path(document.path))
            pages = remove_repeated_headers_footers(pages)
            logger.info("parsed %d page(s) in %.2fs", len(pages), time.perf_counter() - t0)

            t1 = time.perf_counter()
            chunks = self._chunker.chunk_pages(document, pages)
            logger.info("chunked into %d chunk(s) in %.2fs", len(chunks), time.perf_counter() - t1)

        except Exception as exc:
            self._store.update_document_status(document.document_id, "failed")
            raise IngestionError(
                f"Failed to parse/chunk '{document.filename}': {exc}"
            ) from exc

        if not chunks:
            self._store.update_document_status(document.document_id, "failed")
            raise EmptyDocumentError(
                f"'{document.filename}' produced no text chunks after parsing."
            )

        return chunks

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _phase2_embed(
        self,
        chunks: list[Chunk],
        progress: ProgressCallback | None,
    ) -> None:
        """Embed only chunks not already in the vector store.

        Complexity: O(N_new / B) Ollama calls where N_new = uncached chunks,
        B = batch size.  Already-cached chunks cost only an O(1) dict lookup.
        """
        existing = self._vectors.existing_ids()
        new_chunks = [c for c in chunks if c.chunk_id not in existing]
        if not new_chunks:
            logger.info(
                "embedding skipped — all %d chunks already cached", len(chunks)
            )
            return

        total = len(new_chunks)
        done = 0

        for batch in batched(new_chunks, self._embedding_batch_size):
            texts = [c.text for c in batch]
            try:
                vectors = self._embedder.embed(texts)
            except Exception as exc:
                logger.error("embedding batch failed: %s — skipping batch", exc)
                done += len(batch)
                continue

            self._vectors.upsert_many(
                {c.chunk_id: v for c, v in zip(batch, vectors)}
            )
            done += len(batch)
            if progress:
                progress(done, total, f"Embedding {done}/{total} chunks…")

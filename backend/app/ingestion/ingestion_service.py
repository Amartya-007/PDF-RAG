"""IngestionService — single entry point for document ingestion.

Pipeline per document
---------------------
1.  Hash file → detect duplicate / already-indexed
2.  LayoutParser.parse()         → list[LayoutNode]  (text + visual metadata)
3.  StructureBuilder.build()     → list[DocumentNode] (headed tree)
4.  NodeRepository.upsert_many() → persist to SQLite nodes table
5.  IndexManager.add_document_nodes() → FTS5 + Heading + Phrase indexes
6.  OKF generator (optional)    → concept mindmap Markdown files
7.  TreeIndexer (optional)      → JSON section tree for tree navigation
8.  Update Document status      → mark ready / failed in MetadataStore

Idempotency
-----------
Re-ingesting an identical file (same SHA-256) with ``force=False`` returns
the existing ``Document`` record immediately without re-running any stage.
Passing ``force=True`` re-runs the full pipeline and replaces every artifact.

Job tracking
------------
Each ingest call creates an ``IngestionJob`` row via ``JobRepository`` and
transitions it through: queued → processing → succeeded / failed.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from backend.app.core.hashing import sha256_file
from backend.app.database.repositories.job_repository import JobRepository
from backend.app.database.repositories.node_repository import NodeRepository
from backend.app.database.store import MetadataStore
from backend.app.domain.models.job import IngestionJob
from backend.app.domain.models.node import DocumentNode
from backend.app.indexing.index_manager import IndexManager
from backend.app.ingestion.layout_parser import LayoutParser
from backend.app.ingestion.structure_builder import StructureBuilder
from backend.app.models import Document

logger = logging.getLogger(__name__)


class IngestionService:
    """Orchestrates the full document ingestion pipeline.

    Args:
        store:       ``MetadataStore`` for Document CRUD.
        node_repo:   ``NodeRepository`` for DocumentNode persistence.
        job_repo:    ``JobRepository`` for IngestionJob tracking.
        index_mgr:   ``IndexManager`` for FTS5/Heading/Phrase index updates.
        layout:      ``LayoutParser`` for text + layout extraction.
        builder:     ``StructureBuilder`` for tree assembly.
        okf_gen:     Optional OKF concept generator (may be None).
        tree_idx:    Optional ``TreeIndexer`` for vectorless RAG (may be None).
    """

    def __init__(
        self,
        store: MetadataStore,
        node_repo: NodeRepository,
        job_repo: JobRepository,
        index_mgr: IndexManager,
        layout: LayoutParser,
        builder: StructureBuilder,
        okf_gen=None,
        tree_idx=None,
    ) -> None:
        self._store    = store
        self._node_repo = node_repo
        self._job_repo = job_repo
        self._index    = index_mgr
        self._layout   = layout
        self._builder  = builder
        self._okf_gen  = okf_gen
        self._tree_idx = tree_idx

    # ── Public API ─────────────────────────────────────────────────────────

    def ingest(
        self,
        path: Path,
        session_id: str | None = None,
        build_okf: bool = False,
        force: bool = False,
    ) -> Document:
        """Ingest a single document file.

        Args:
            path:       Absolute path to the document.
            session_id: Associate this document with a workspace session.
            build_okf:  Whether to run the OKF concept generator.
            force:      If True, re-ingest even when the file hash is unchanged.

        Returns:
            The ``Document`` record (status == 'ready' on success).

        Raises:
            FileNotFoundError: When *path* does not exist.
            Exception:         Any pipeline stage failure (job marked failed).
        """
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        filename = path.name
        file_hash = sha256_file(path)

        # Idempotency check
        existing = self._store.get_document_by_hash(file_hash)
        if existing and not force:
            logger.info("IngestionService: skipping duplicate %s (hash=%s)", filename, file_hash[:8])
            return existing

        # Create or reset Document record
        document = self._store.upsert_document_by_hash(
            filename=filename,
            path=str(path),
            file_hash=file_hash,
            session_id=session_id,
            status="processing",
        )

        # Create IngestionJob
        job = IngestionJob(
            job_id=str(uuid.uuid4()),
            document_id=document.document_id,
            session_id=session_id or "",
            status="processing",
            progress_message="Starting ingestion",
        )
        job = self._job_repo.create(job)

        try:
            nodes = self._run_pipeline(path, document, job, build_okf)
        except Exception as exc:
            self._job_repo.update_status(job.job_id, "failed", str(exc)[:500])
            self._store.set_document_status(document.document_id, "failed")
            logger.exception("IngestionService: pipeline failed for %s", filename)
            raise

        self._job_repo.update_status(job.job_id, "succeeded",
                                      f"Indexed {len(nodes)} nodes")
        self._store.set_document_status(document.document_id, "ready")
        logger.info("IngestionService: %s → %d nodes indexed", filename, len(nodes))
        return self._store.get_document(document.document_id)

    # ── Pipeline stages ────────────────────────────────────────────────────

    def _run_pipeline(
        self,
        path: Path,
        document: Document,
        job: IngestionJob,
        build_okf: bool,
    ) -> list[DocumentNode]:
        # Stage 1: Layout extraction
        self._job_repo.update_status(job.job_id, "processing", "Parsing layout")
        layout_nodes = self._layout.parse(path)
        if not layout_nodes:
            raise ValueError(f"No text extracted from {path.name}")

        # Stage 2: Structure building
        self._job_repo.update_status(job.job_id, "processing", "Building document tree")
        nodes = self._builder.build(layout_nodes, document_id=document.document_id)
        if not nodes:
            raise ValueError(f"No document nodes built from {path.name}")

        # Stage 3: Remove stale nodes + persist new ones
        self._job_repo.update_status(job.job_id, "processing", "Storing nodes")
        stale_ids = [n.id for n in self._node_repo.list_nodes_for_document(document.document_id)]
        if stale_ids:
            self._index.remove_document(document.document_id, stale_ids)
            self._node_repo.delete_for_document(document.document_id)
        self._node_repo.upsert_many(nodes)

        # Stage 4: Index nodes
        self._job_repo.update_status(job.job_id, "processing", "Updating search indexes")
        self._index.add_document_nodes(nodes)

        # Stage 5: OKF concept generation (optional)
        if build_okf and self._okf_gen is not None:
            self._job_repo.update_status(job.job_id, "processing", "Generating concepts")
            try:
                self._okf_gen.generate_for_document(nodes)
            except Exception as exc:
                logger.warning("OKF generation failed (non-fatal): %s", exc)

        # Stage 6: Tree index (optional)
        if self._tree_idx is not None:
            self._job_repo.update_status(job.job_id, "processing", "Building tree index")
            try:
                from backend.app.ingestion.parser.pdf_parser import PageText
                # Reconstruct PageText from nodes for TreeIndexer compatibility
                page_texts = self._nodes_to_page_texts(nodes)
                tree = self._tree_idx.build(document.document_id, document.filename, page_texts)
                self._tree_idx.save(tree)
                self._tree_idx.save_with_raw(tree)
            except Exception as exc:
                logger.warning("Tree indexing failed (non-fatal): %s", exc)

        return nodes

    @staticmethod
    def _nodes_to_page_texts(nodes: list[DocumentNode]):
        from backend.app.ingestion.parser.pdf_parser import PageText
        page_map: dict[int, list[str]] = {}
        for node in nodes:
            page_map.setdefault(node.page_start, []).append(node.text)
        return [
            PageText(page_number=p, text="\n".join(texts))
            for p, texts in sorted(page_map.items())
        ]


# Compatibility shim: task 4.6 moved the implementation to the service layer.
from backend.app.services.ingestion_service import (  # noqa: E402,F401
    IngestionService,
    ProgressCallback,
)

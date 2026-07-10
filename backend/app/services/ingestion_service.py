"""Vectorless document ingestion service.

This is the service-layer entry point for turning a source file into
hierarchical ``DocumentNode`` records and vectorless search indexes.
It intentionally does not import or call embedding/vector components.
"""
from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from backend.app.core.hashing import sha256_file, stable_id
from backend.app.database.repositories.job_repository import JobRepository
from backend.app.database.repositories.node_repository import NodeRepository
from backend.app.database.store import DEFAULT_SESSION_ID, MetadataStore
from backend.app.domain.exceptions import EmptyDocumentError
from backend.app.domain.models.job import IngestionJob
from backend.app.domain.models.node import DocumentNode
from backend.app.indexing.index_manager import IndexManager
from backend.app.ingestion.chunking import Chunker
from backend.app.ingestion.heading_detector import HeadingDetector
from backend.app.ingestion.layout_parser import LayoutNode, LayoutParser
from backend.app.ingestion.parser.pdf_parser import PageText
from backend.app.ingestion.structure_builder import StructureBuilder
from backend.app.models import Document

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class IngestionService:
    """Orchestrates layout parsing, structure detection, persistence, and indexes."""

    _TOTAL_STAGES = 8

    def __init__(
        self,
        *,
        store: MetadataStore,
        node_repo: NodeRepository,
        job_repo: JobRepository,
        index_mgr: IndexManager,
        layout: LayoutParser,
        builder: StructureBuilder,
        detector: HeadingDetector | None = None,
        chunker: Chunker | None = None,
        okf_gen=None,
        tree_idx=None,
    ) -> None:
        self._store = store
        self._node_repo = node_repo
        self._job_repo = job_repo
        self._index = index_mgr
        self._layout = layout
        self._builder = builder
        self._detector = detector or HeadingDetector()
        self._chunker = chunker or Chunker()
        self._okf_gen = okf_gen
        self._tree_idx = tree_idx

    def ingest(
        self,
        path: Path,
        session_id: str | None = None,
        build_okf: bool = False,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Document:
        """Ingest *path* and return the resulting ``Document`` record."""
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        active_session = session_id or DEFAULT_SESSION_ID
        file_hash = sha256_file(path)
        existing = self._store.find_document_by_hash(file_hash, active_session)
        
        if existing and existing.status == "ready" and not force:
            logger.info("Skipping duplicate ingestion for %s", path.name)
            return existing

        document = self._prepare_document(path, file_hash, active_session, existing)
        job = self._job_repo.create_job(
            IngestionJob(
                job_id=str(uuid.uuid4()),
                document_id=document.document_id,
                session_id=active_session,
                status="queued",
                progress_message="Queued",
            )
        )
        self._emit(progress, 1, "Queued")

        try:
            nodes = self._run_pipeline(path, document, job.job_id, build_okf, progress)
        except Exception as exc:
            self._job_repo.update_job_status(job.job_id, "failed", str(exc)[:500])
            self._store.update_document_status(document.document_id, "failed")
            logger.exception("Ingestion failed for %s", path.name)
            raise

        self._job_repo.update_job_status(
            job.job_id, "completed", f"Indexed {len(nodes)} nodes"
        )
        self._store.update_document_status(document.document_id, "ready")
        self._emit(progress, self._TOTAL_STAGES, "Completed")
        return self._store.get_document(document.document_id)

    def _prepare_document(
        self,
        path: Path,
        file_hash: str,
        session_id: str,
        existing: Document | None,
    ) -> Document:
        """Initializes or retrieves the document record."""
        document = Document(
            document_id=existing.document_id
            if existing
            else stable_id("doc", session_id, path.name, file_hash),
            filename=path.name,
            sha256=file_hash,
            path=str(path),
            status="processing",
            session_id=session_id,
        )
        self._store.upsert_document(document)
        return document

    def _run_pipeline(
        self,
        path: Path,
        document: Document,
        job_id: str,
        build_okf: bool,
        progress: ProgressCallback | None,
    ) -> list[DocumentNode]:
        """Executes the sequential ingestion pipeline stages."""
        self._advance(job_id, "parsing", "Parsing layout", 2, progress)
        layout_nodes = self._layout.parse(path)
        if not layout_nodes:
            raise EmptyDocumentError(f"'{document.filename}' produced no text.")

        self._advance(job_id, "cleaning", "Cleaning text", 3, progress)
        layout_nodes = self._clean_layout_nodes(layout_nodes)
        if not layout_nodes:
            raise EmptyDocumentError(f"'{document.filename}' produced no text after cleaning.")

        self._advance(job_id, "detecting_structure", "Detecting structure", 4, progress)
        self._detector.detect(layout_nodes)

        self._advance(job_id, "building_nodes", "Building nodes", 5, progress)
        nodes = self._builder.build(layout_nodes, document_id=document.document_id)
        if not nodes:
            raise EmptyDocumentError(f"'{document.filename}' produced no document nodes.")

        self._replace_persisted_nodes(document.document_id, nodes)
        chunks = self._chunker.chunk_pages(document, self._nodes_to_page_texts(nodes))
        self._store.replace_chunks(document.document_id, chunks)

        self._advance(job_id, "indexing_fts", "Indexing full text", 6, progress)
        self._index.add_document_nodes(nodes)

        self._advance(job_id, "indexing_headings", "Indexing headings", 7, progress)
        if build_okf and self._okf_gen:
            self._okf_gen.generate_for_document(chunks)
        if self._tree_idx:
            tree = self._tree_idx.build(
                document.document_id,
                document.filename,
                self._nodes_to_page_texts(nodes),
            )
            self._tree_idx.save(tree)

        return nodes

    def _replace_persisted_nodes(
        self, document_id: str, nodes: list[DocumentNode]
    ) -> None:
        """Removes existing nodes for a document and persists new ones."""
        stale_nodes = self._node_repo.list_nodes_for_document(document_id)
        if stale_nodes:
            self._index.remove_document(document_id, [n.id for n in stale_nodes])
            self._node_repo.delete_nodes_for_document(document_id)
        self._node_repo.upsert_many(nodes)

    @staticmethod
    def _clean_layout_nodes(nodes: list[LayoutNode]) -> list[LayoutNode]:
        """Filters nodes and cleans whitespace from text content."""
        return [
            node.replace(text=node.text.strip()) 
            for node in nodes 
            if node.text and node.text.strip()
        ]

    @staticmethod
    def _nodes_to_page_texts(nodes: list[DocumentNode]) -> list[PageText]:
        """Aggregates node text by page for chunking."""
        page_map: dict[int, list[str]] = defaultdict(list)
        section_paths: dict[int, tuple[str, ...]] = {}

        for node in nodes:
            if node.node_type == "document" or not node.text.strip():
                continue
            
            page_map[node.page_start].append(node.text)
            # Retain the first valid section path found per page
            if node.page_start not in section_paths:
                section_paths[node.page_start] = tuple(node.heading_path)

        return [
            PageText(
                page_number=page,
                text="\n\n".join(texts),
                section_path=section_paths.get(page, ()),
            )
            for page, texts in sorted(page_map.items())
        ]

    def _advance(
        self,
        job_id: str,
        status: str,
        message: str,
        done: int,
        progress: ProgressCallback | None,
    ) -> None:
        """Updates job status and emits progress update."""
        self._job_repo.update_job_status(job_id, status, message)
        self._emit(progress, done, message)

    def _emit(
        self,
        progress: ProgressCallback | None,
        done: int,
        message: str,
    ) -> None:
        """Calls the progress callback if provided."""
        if progress:
            progress(done, self._TOTAL_STAGES, message)
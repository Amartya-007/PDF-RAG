"""IngestionService — single entry point for document ingestion.

Orchestrates the pipeline from raw file upload to ready-to-search nodes.

Pipeline per document
---------------------
1. Hash file → check for existing Document (idempotency).
2. LayoutParser → extract text + visual metadata.
3. StructureBuilder → build hierarchical tree/nodes.
4. Repositories → persist to SQLite.
5. IndexManager → sync lexical + sparse indexes.
6. Optional tasks (OKF, tree index) → run in isolated blocks.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from backend.app.core.hashing import sha256_file
from backend.app.domain.exceptions import IngestionError
from backend.app.ingestion.parser.pdf_parser import PageText

if TYPE_CHECKING:
    from backend.app.database.repositories.document_repository import DocumentRepository
    from backend.app.database.repositories.job_repository import JobRepository
    from backend.app.database.repositories.node_repository import NodeRepository
    from backend.app.indexing.index_manager import IndexManager
    from backend.app.indexing.tree_indexer import TreeIndexer
    from backend.app.ingestion.layout_parser import LayoutParser
    from backend.app.ingestion.structure_builder import StructureBuilder
    from backend.app.models import Document, DocumentNode

logger = logging.getLogger(__name__)


class IngestionService:
    """Coordinates the document ingestion pipeline."""

    def __init__(
        self,
        doc_repo: DocumentRepository,
        node_repo: NodeRepository,
        job_repo: JobRepository,
        parser: LayoutParser,
        builder: StructureBuilder,
        index_manager: IndexManager,
        tree_idx: TreeIndexer | None = None,
    ) -> None:
        self._doc_repo = doc_repo
        self._node_repo = node_repo
        self._job_repo = job_repo
        self._parser = parser
        self._builder = builder
        self._index_manager = index_manager
        self._tree_idx = tree_idx

    def ingest(self, job_id: str, path: Path, force: bool = False) -> list[DocumentNode]:
        """Execute the ingestion pipeline for a single file.

        Attributes:
            job_id: The active ingestion job identifier.
            path:   Path to the uploaded PDF/text file.
            force:  If True, overwrite existing data for this file.
        """
        job = self._job_repo.get(job_id)
        file_hash = sha256_file(path)

        # 1. Idempotency Check
        if not force and (existing := self._doc_repo.get_by_sha256(file_hash)):
            logger.info("Duplicate document detected: %s", existing.document_id)
            return self._node_repo.list_nodes(existing.document_id)

        # 2. Pipeline Execution
        try:
            self._job_repo.update_status(job_id, "parsing", "Extracting text")
            layout_nodes = self._parser.parse(path)
            
            self._job_repo.update_status(job_id, "building_nodes", "Structuring content")
            nodes = self._builder.build(job.document_id, layout_nodes)

            # 3. Persistence & Indexing (Atomic block)
            self._job_repo.update_status(job_id, "indexing_fts", "Updating search indexes")
            self._node_repo.upsert_many(nodes)
            self._index_manager.add_document_nodes(nodes)

            # 4. Optional secondary tasks
            self._run_secondary_tasks(nodes, job)

            self._job_repo.update_status(job_id, "completed", "Ingestion successful")
            return nodes

        except Exception as exc:
            self._job_repo.update_status(job_id, "failed", str(exc))
            logger.error("Ingestion failed for job %s: %s", job_id, exc)
            raise IngestionError(f"Pipeline failure: {exc}") from exc

    def _run_secondary_tasks(self, nodes: list[DocumentNode], job: Any) -> None:
        """Run non-critical indexing tasks (OKF, Tree Indexing)."""
        # Tree indexing (optional)
        if self._tree_idx:
            try:
                page_texts = self._nodes_to_page_texts(nodes)
                tree = self._tree_idx.build(job.document_id, "doc.pdf", page_texts)
                self._tree_idx.save(tree)
            except Exception as exc:
                logger.warning("Tree indexing failed (non-fatal): %s", exc)

    @staticmethod
    def _nodes_to_page_texts(nodes: list[DocumentNode]) -> list[PageText]:
        """Convert list of DocumentNodes back to PageText objects."""
        page_map: dict[int, list[str]] = {}
        for node in nodes:
            page_map.setdefault(node.page_start, []).append(node.text)
        
        return [
            PageText(page_number=p, text="\n".join(texts))
            for p, texts in sorted(page_map.items())
        ]
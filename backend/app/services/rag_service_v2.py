"""RagServiceV2 — thin facade over the new domain service layer.

Wires together all collaborators so desktop/controller.py and the API
layer get the same ingest()/ask() interface without any business logic
living in this file.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from backend.app.core.config import Settings, ensure_data_dirs, get_settings
from backend.app.database.repositories.job_repository import JobRepository
from backend.app.database.repositories.node_repository import NodeRepository
from backend.app.database.store import MetadataStore
from backend.app.generation.answer_service import AnswerService
from backend.app.generation.extractive_answerer import ExtractiveAnswerer
from backend.app.generation.ollama_answerer import OllamaAnswerer
from backend.app.generation.ollama_client import OllamaClient
from backend.app.indexing.full_text_index import FTS5Index
from backend.app.indexing.heading_index import HeadingIndex
from backend.app.indexing.index_manager import IndexManager
from backend.app.indexing.phrase_index import PhraseIndex
from backend.app.indexing.sparse import BM25Index
from backend.app.ingestion.heading_detector import HeadingDetector
from backend.app.ingestion.ingestion_service import IngestionService
from backend.app.ingestion.layout_parser import LayoutParser
from backend.app.ingestion.structure_builder import StructureBuilder
from backend.app.models import Answer, Document
from backend.app.retrieval.confidence_gate import ConfidenceGate
from backend.app.retrieval.lexical_retriever import LexicalRetriever
from backend.app.retrieval.node_ranker import NodeRanker
from backend.app.retrieval.retrieval_service import RetrievalService
from backend.app.retrieval.tree_navigator import TreeNavigator

logger = logging.getLogger(__name__)


class RagServiceV2:
    """Fully-wired RAG service using the new domain service architecture.

    Drop-in replacement for the legacy RagService.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        ensure_data_dirs(self.settings)

        # Storage
        self.store = MetadataStore(self.settings.sqlite_path)
        self.store.init()

        connect = self.store.connection_factory
        self._node_repo = NodeRepository(connect)
        self._job_repo  = JobRepository(connect)

        # Indexes
        self._fts5    = FTS5Index(connect)
        self._heading = HeadingIndex()
        self._phrase  = PhraseIndex()
        self._bm25    = BM25Index(self.settings.indexes_dir / "bm25.json")
        self._idx_mgr = IndexManager(self._fts5, self._heading, self._phrase, self._bm25)
        try:
            self._idx_mgr.rebuild_all(self._node_repo)
        except Exception as exc:
            logger.warning("Startup index rebuild skipped: %s", exc)

        # Generation
        self._ollama: OllamaClient | None = None
        ollama_ans: OllamaAnswerer | None = None
        if self.settings.use_ollama:
            self._ollama = OllamaClient(self.settings)
            ollama_ans   = OllamaAnswerer(self._ollama)
        self.answer_service = AnswerService(ExtractiveAnswerer(), ollama_ans)

        # Retrieval
        self._lexical   = LexicalRetriever(self._fts5, self._heading, self._phrase)
        self._navigator = TreeNavigator(self._ollama)
        self.retrieval_service = RetrievalService(
            self._node_repo, self._lexical, self._navigator,
            NodeRanker(), ConfidenceGate(),
            top_k=self.settings.final_context_chunks,
        )

        # Ingestion
        self.ingestion_service = IngestionService(
            store=self.store, node_repo=self._node_repo,
            job_repo=self._job_repo, index_mgr=self._idx_mgr,
            layout=LayoutParser(),
            builder=StructureBuilder(HeadingDetector()),
        )

    # ── Facade ────────────────────────────────────────────────────────────

    def ingest(self, path, session_id=None, build_okf=False, force=False) -> Document:
        return self.ingestion_service.ingest(
            Path(path), session_id=session_id, build_okf=build_okf, force=force
        )

    def ask(self, question: str, session_id=None, include_debug=False) -> Answer:
        result = self.retrieval_service.retrieve(
            question, session_id=session_id, include_debug=include_debug
        )
        if not result.nodes:
            return Answer(
                question=question,
                answer=result.error or "No relevant documents found.",
                citations=[], answerable=False,
            )
        return self.answer_service.answer(question, result.nodes)

    def list_documents(self) -> list[Document]:
        return self.store.list_documents()

    def status(self) -> dict:
        docs = self.store.list_documents()
        try:
            n_nodes = len(self._node_repo.list_all_nodes())
        except Exception:
            n_nodes = 0
        return {
            "documents": len(docs),
            "chunks": n_nodes,
            "concepts": 0,
            "ollama_ready": self.settings.use_ollama,
            "ollama_message": (
                f"Ollama enabled ({self.settings.active_model})"
                if self.settings.use_ollama else "Ollama disabled — extractive mode"
            ),
        }

    def close(self) -> None:
        pass

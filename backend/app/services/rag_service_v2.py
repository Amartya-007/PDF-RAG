"""RagServiceV2 — thin facade over the new domain service layer.

Wires together all collaborators so desktop/controller.py and the API
layer get the same ingest()/ask() interface without any business logic
living in this file.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings, ensure_data_dirs, get_settings
from backend.app.database.repositories.job_repository import JobRepository
from backend.app.database.repositories.node_repository import NodeRepository
from backend.app.database.store import (
    DEFAULT_SESSION_ID,
    DEFAULT_SESSION_TITLE,
    MetadataStore,
)
from backend.app.generation.answer_service import AnswerService
from backend.app.generation.extractive_answerer import ExtractiveAnswerer
from backend.app.generation.ollama_answerer import OllamaAnswerer
from backend.app.generation.ollama_client import OllamaClient
from backend.app.indexing.full_text_index import FTS5Index
from backend.app.indexing.heading_index import HeadingIndex
from backend.app.indexing.index_manager import IndexManager
from backend.app.indexing.metadata_index import MetadataIndex
from backend.app.indexing.phrase_index import PhraseIndex
from backend.app.indexing.sparse import BM25Index
from backend.app.ingestion.heading_detector import HeadingDetector
from backend.app.services.ingestion_service import IngestionService
from backend.app.ingestion.layout_parser import LayoutParser
from backend.app.ingestion.structure_builder import StructureBuilder
from backend.app.knowledge.okf import OkfConcept, validate_okf_bundle
from backend.app.knowledge.okf_generator import OkfGenerator
from backend.app.knowledge.okf_importer import OkfImporter
from backend.app.models import Answer, ChatSession, Chunk, Citation, Document
from backend.app.retrieval.confidence_gate import ConfidenceGate
from backend.app.retrieval.lexical_retriever import LexicalRetriever
from backend.app.retrieval.node_ranker import NodeRanker
from backend.app.retrieval.retrieval_service import RetrievalService
from backend.app.retrieval.tree_navigator import TreeNavigator

logger = logging.getLogger(__name__)


class RagServiceV2:
    """Fully-wired RAG service using the new domain service architecture."""

    def __init__(
        self,
        settings: Settings | None = None,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> None:
        self.settings = settings or get_settings()
        ensure_data_dirs(self.settings)
        self.session_id = session_id

        # Infrastructure initialization
        self.store = MetadataStore(self.settings.sqlite_path)
        self.store.init()
        self.store.ensure_session(DEFAULT_SESSION_ID, DEFAULT_SESSION_TITLE)
        self.store.ensure_session(self.session_id, DEFAULT_SESSION_TITLE)
        self.store.mark_stale_processing_documents_failed()

        connect = self.store.connection_factory
        self._node_repo = NodeRepository(connect)
        self._job_repo = JobRepository(connect)

        # Indexing components
        self._fts5 = FTS5Index(connect)
        self._idx_mgr = IndexManager(
            self._fts5, 
            HeadingIndex(), 
            PhraseIndex(), 
            BM25Index(self.settings.indexes_dir / "bm25.json"),
            MetadataIndex(),
            self._node_repo,
        )
        try:
            self._idx_mgr.rebuild_all()
        except Exception as exc:
            logger.warning("Startup index rebuild skipped: %s", exc)

        # Generation services
        ollama_ans = None
        if self.settings.use_ollama:
            ollama_ans = OllamaAnswerer(OllamaClient(self.settings))
        
        self._extractive_answerer = ExtractiveAnswerer()
        self.answer_service = AnswerService(self._extractive_answerer, ollama_ans)

        # Retrieval pipeline
        self.retrieval_service = RetrievalService(
            self._node_repo,
            LexicalRetriever(self._fts5, HeadingIndex(), PhraseIndex(), self._node_repo),
            TreeNavigator(),
            NodeRanker(),
            ConfidenceGate(),
            top_k=self.settings.final_context_chunks,
        )

        # OKF Knowledge integration
        self.okf = OkfGenerator(self.settings.okf_dir, self.store)
        self.okf_importer = OkfImporter(self.settings.okf_dir, self.store)

        # Ingestion service facade
        self.ingestion_service = IngestionService(
            store=self.store,
            node_repo=self._node_repo,
            job_repo=self._job_repo,
            index_mgr=self._idx_mgr,
            layout=LayoutParser(),
            builder=StructureBuilder(HeadingDetector()),
            okf_gen=self.okf,
        )

    def ingest(self, path: str | Path, session_id: str | None = None, build_okf: bool = False, force: bool = False) -> Document:
        """Facilitates document ingestion via the IngestionService."""
        return self.ingestion_service.ingest(
            Path(path),
            session_id=session_id or self.session_id,
            build_okf=build_okf,
            force=force,
        )

    def ask(self, question: str, session_id: str | None = None, include_debug: bool = False) -> Answer:
        """Processes a query through retrieval and generation pipelines."""
        active_session = session_id or self.session_id
        result = self.retrieval_service.retrieve(
            question, session_id=active_session, include_debug=include_debug
        )
        debug = self._debug_for_question(question, result) if include_debug else {}
        
        if not result.nodes:
            return Answer(
                question=question,
                answer=result.error or "No relevant documents found.",
                citations=[],
                answerable=False,
                debug=debug,
            )
            
        answer = self.answer_service.answer(question, result.nodes)
        enriched_citations = [self._enrich_citation(c) for c in answer.citations]
        return replace(answer, citations=enriched_citations, debug=debug)

    def retrieve(
        self,
        query: str,
        include_debug: bool = False,
        session_id: str | None = None,
    ) -> tuple[list[Chunk], dict[str, Any]]:
        """Retrieves raw chunks based on query, prioritizing OKF concept matches."""
        active_session = session_id or self.session_id
        chunks: list[Chunk] = []
        debug: dict[str, Any] = {}

        # Concept-based retrieval
        concept_matches = self._match_okf_concepts(query)
        source_ids = {cid for c in concept_matches for cid in c.source_chunk_ids}
        
        if source_ids:
            chunks = [
                c for c in self.store.chunks_by_ids(list(source_ids)) 
                if c.session_id == active_session
            ]

        # Fallback to standard retrieval
        if not chunks:
            result = self.retrieval_service.retrieve(
                query, session_id=active_session, include_debug=include_debug
            )
            chunks = self._nodes_to_chunks(result.nodes)
            if include_debug:
                debug.update(self._debug_for_question(query, result))

        if include_debug:
            debug.update({"okf_concept_results": concept_matches, "okf_source_results": chunks})

        return chunks, debug

    def list_documents(self, session_id: str | None = None) -> list[Document]:
        return self.store.list_documents(session_id or self.session_id)

    def list_sessions(self) -> list[ChatSession]:
        return self.store.list_sessions()

    def create_session(self, title: str | None = None) -> ChatSession:
        session = self.store.create_session(f"session_{uuid.uuid4().hex}", title or "New Chat")
        self.session_id = session.session_id
        return session

    def set_session(self, session_id: str) -> ChatSession:
        self.session_id = session_id
        return self.store.ensure_session(session_id, "Chat")

    def delete_document(self, document_id: str) -> list[str]:
        self._idx_mgr.remove_document(document_id)
        return self.store.delete_document(document_id)

    def delete_session_documents(self, session_id: str | None = None) -> None:
        for doc in self.store.list_documents(session_id or self.session_id):
            self.delete_document(doc.document_id)

    def status(self) -> dict[str, Any]:
        """Provides a snapshot of the current service state."""
        docs = self.store.list_documents(self.session_id)
        return {
            "documents": len(docs),
            "chunks": len(self._node_repo.list_nodes_for_session(self.session_id)),
            "concepts": len(self.store.list_concepts()),
            "ollama_ready": self.settings.use_ollama,
            "ollama_message": f"Ollama enabled ({self.settings.active_model})" if self.settings.use_ollama else "Extractive mode",
        }

    def close(self) -> None:
        self.store.close()

    def _debug_for_question(self, question: str, result: Any) -> dict[str, Any]:
        norm = question.lower()
        return {
            "fast_fact_query": self._extractive_answerer.is_fast_fact_question(question),
            "topic_query": any(phrase in norm for phrase in ["everything about", "tell me", "explain"]),
            "gate_score": result.gate_decision.score,
            "lexical_hits": result.lexical_hits,
            "tree_hits": result.tree_hits,
        }

    def _enrich_citation(self, citation: Citation) -> Citation:
        try:
            doc = self.store.get_document(citation.document_id)
            return replace(citation, filename=doc.filename)
        except Exception:
            return citation

    def _nodes_to_chunks(self, nodes: list[Any]) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=n.id, document_id=n.document_id, filename=self.store.get_document(n.document_id).filename,
                page_start=n.page_start, page_end=n.page_end or n.page_start,
                section_path=tuple(n.heading_path), text=n.text, session_id=self.session_id
            )
            for n in nodes
        ]

    def _match_okf_concepts(self, query: str) -> list[OkfConcept]:
        query_terms = self._terms(query)
        if not query_terms:
            return []
            
        scored = []
        for concept in self.store.list_concepts():
            haystack = f"{concept.title} {concept.slug} {concept.text} {' '.join(concept.aliases)} {' '.join(concept.tags)}"
            overlap = len(query_terms & self._terms(haystack))
            if overlap:
                scored.append((overlap, concept))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:5]]

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {p.lower() for p in text.replace("-", " ").split() if len(p) > 2}
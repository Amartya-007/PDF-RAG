"""RagServiceV2 — thin facade over the new domain service layer.

Wires together all collaborators so desktop/controller.py and the API
layer get the same ingest()/ask() interface without any business logic
living in this file.
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import replace
from pathlib import Path

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
from backend.app.indexing.phrase_index import PhraseIndex
from backend.app.indexing.sparse import BM25Index
from backend.app.ingestion.heading_detector import HeadingDetector
from backend.app.ingestion.ingestion_service import IngestionService
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
    """Fully-wired RAG service using the new domain service architecture.

    Drop-in replacement for the legacy RagService.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> None:
        self.settings = settings or get_settings()
        ensure_data_dirs(self.settings)
        self.session_id = session_id

        # Storage
        self.store = MetadataStore(self.settings.sqlite_path)
        self.store.init()
        self.store.ensure_session(DEFAULT_SESSION_ID, DEFAULT_SESSION_TITLE)
        self.store.ensure_session(self.session_id, DEFAULT_SESSION_TITLE)
        self.store.mark_stale_processing_documents_failed()

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
        self._extractive_answerer = ExtractiveAnswerer()
        self.answer_service = AnswerService(self._extractive_answerer, ollama_ans)

        # Retrieval
        self._lexical   = LexicalRetriever(
            self._fts5,
            self._heading,
            self._phrase,
            self._node_repo,
        )
        self._navigator = TreeNavigator()
        self.retrieval_service = RetrievalService(
            self._node_repo, self._lexical, self._navigator,
            NodeRanker(), ConfidenceGate(),
            top_k=self.settings.final_context_chunks,
        )

        # OKF compatibility: concepts remain source-linked lexical aids.
        self.okf = OkfGenerator(self.settings.okf_dir, self.store)
        self.okf_importer = OkfImporter(self.settings.okf_dir, self.store)

        # Ingestion
        self.ingestion_service = IngestionService(
            store=self.store, node_repo=self._node_repo,
            job_repo=self._job_repo, index_mgr=self._idx_mgr,
            layout=LayoutParser(),
            builder=StructureBuilder(HeadingDetector()),
            okf_gen=self.okf,
        )

    # ── Facade ────────────────────────────────────────────────────────────

    def ingest(self, path, session_id=None, build_okf=False, force=False) -> Document:
        return self.ingestion_service.ingest(
            Path(path),
            session_id=session_id or self.session_id,
            build_okf=build_okf,
            force=force,
        )

    def ask(self, question: str, session_id=None, include_debug=False) -> Answer:
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
        citations = [self._enrich_citation(citation) for citation in answer.citations]
        return Answer(
            question=answer.question,
            answer=answer.answer,
            citations=citations,
            answerable=answer.answerable,
            debug=debug,
        )

    def retrieve(
        self,
        query: str,
        include_debug: bool = False,
        session_id: str | None = None,
    ) -> tuple[list[Chunk], dict[str, object]]:
        active_session = session_id or self.session_id
        chunks: list[Chunk] = []
        debug: dict[str, object] = {}

        concept_matches = self._match_okf_concepts(query)
        source_chunk_ids: list[str] = []
        for concept in concept_matches:
            source_chunk_ids.extend(concept.source_chunk_ids)
        if source_chunk_ids:
            chunks.extend(self.store.chunks_by_ids(self._unique(source_chunk_ids)))
            chunks = [chunk for chunk in chunks if chunk.session_id == active_session]

        if not chunks:
            result = self.retrieval_service.retrieve(
                query, session_id=active_session, include_debug=include_debug
            )
            chunks = self._nodes_to_chunks(result.nodes)
            if include_debug:
                debug.update(self._debug_for_question(query, result))

        if include_debug:
            debug["okf_concept_results"] = concept_matches
            debug["okf_source_results"] = chunks if concept_matches else []

        return chunks, debug

    def list_documents(self, session_id: str | None = None) -> list[Document]:
        return self.store.list_documents(session_id or self.session_id)

    def list_sessions(self) -> list[ChatSession]:
        return self.store.list_sessions()

    def create_session(self, title: str | None = None) -> ChatSession:
        session = self.store.create_session(
            f"session_{uuid.uuid4().hex}",
            title or "New Chat",
        )
        self.session_id = session.session_id
        return session

    def set_session(self, session_id: str) -> ChatSession:
        for session in self.store.list_sessions():
            if session.session_id == session_id:
                self.session_id = session_id
                return session
        session = self.store.ensure_session(session_id, "Chat")
        self.session_id = session_id
        return session

    def delete_document(self, document_id: str) -> list[str]:
        node_ids = [node.id for node in self._node_repo.list_nodes_for_document(document_id)]
        if node_ids:
            self._idx_mgr.remove_document(document_id, node_ids)
        return self.store.delete_document(document_id)

    def delete_session_documents(self, session_id: str | None = None) -> None:
        for document in self.store.list_documents(session_id or self.session_id):
            self.delete_document(document.document_id)

    def repair_unready_documents(self) -> int:
        return self.store.mark_stale_processing_documents_failed()

    def import_okf_bundle(self, source_root: Path | str) -> list[OkfConcept]:
        return self.okf_importer.import_bundle(Path(source_root))

    def validate_okf_bundle(self, root: Path | str):
        return validate_okf_bundle(Path(root))

    def status(self) -> dict:
        docs = self.store.list_documents(self.session_id)
        try:
            n_nodes = len(self._node_repo.list_nodes_for_session(self.session_id))
        except Exception:
            n_nodes = 0
        return {
            "documents": len(docs),
            "chunks": n_nodes,
            "concepts": len(self.store.list_concepts()),
            "ollama_ready": self.settings.use_ollama,
            "ollama_message": (
                f"Ollama enabled ({self.settings.active_model})"
                if self.settings.use_ollama else "Ollama disabled - extractive mode"
            ),
        }

    def close(self) -> None:
        self.store.close()

    def _debug_for_question(self, question: str, result) -> dict[str, object]:
        normalized = question.lower()
        return {
            "fast_fact_query": self._extractive_answerer.is_fast_fact_question(question),
            "topic_query": (
                "everything about" in normalized
                or normalized.startswith("tell me")
                or normalized.startswith("explain")
            ),
            "gate_score": result.gate_decision.score,
            "gate_reason": result.gate_decision.reason,
            "lexical_hits": result.lexical_hits,
            "tree_hits": result.tree_hits,
        }

    def _enrich_citation(self, citation: Citation) -> Citation:
        try:
            document = self.store.get_document(citation.document_id)
        except Exception:
            return citation
        return replace(citation, filename=document.filename)

    def _nodes_to_chunks(self, nodes) -> list[Chunk]:
        chunks: list[Chunk] = []
        for node in nodes:
            try:
                document = self.store.get_document(node.document_id)
            except Exception:
                continue
            chunks.append(
                Chunk(
                    chunk_id=node.id,
                    document_id=node.document_id,
                    filename=document.filename,
                    page_start=node.page_start,
                    page_end=node.page_end or node.page_start,
                    section_path=tuple(node.heading_path),
                    text=node.text,
                    session_id=document.session_id,
                )
            )
        return chunks

    def _match_okf_concepts(self, query: str) -> list[OkfConcept]:
        query_terms = self._terms(query)
        if not query_terms:
            return []
        scored: list[tuple[int, OkfConcept]] = []
        for concept in self.store.list_concepts():
            haystack = " ".join(
                [
                    concept.title,
                    concept.slug,
                    concept.text,
                    " ".join(concept.aliases),
                    " ".join(concept.tags),
                ]
            )
            overlap = len(query_terms & self._terms(haystack))
            if overlap:
                scored.append((overlap, concept))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [concept for _score, concept in scored[:5]]

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {part.lower() for part in text.replace("-", " ").split() if len(part) > 2}

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            unique.append(value)
        return unique

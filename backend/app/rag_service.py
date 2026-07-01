from __future__ import annotations

import logging
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from backend.app.core.config import Settings, ensure_data_dirs, get_settings
from backend.app.core.hashing import sha256_file, stable_id
from backend.app.database.store import DEFAULT_SESSION_ID, DEFAULT_SESSION_TITLE, MetadataStore
from backend.app.generation.answerer import Answerer
from backend.app.indexing.embeddings import EmbeddingService
from backend.app.indexing.sparse import BM25Index
from backend.app.indexing.vector_store import LocalVectorStore
from backend.app.ingestion.chunking import Chunker
from backend.app.ingestion.cleaning import remove_repeated_headers_footers
from backend.app.ingestion.parser.pdf_parser import PdfParser
from backend.app.knowledge.okf import OkfConcept, validate_okf_bundle
from backend.app.knowledge.okf_generator import OkfGenerator
from backend.app.knowledge.okf_importer import OkfImporter
from backend.app.models import Answer, ChatSession, Chunk, Document
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.retrieval.query_analysis import classify_query
from backend.app.retrieval.reranking import Reranker


logger = logging.getLogger(__name__)


class RagService:
    def __init__(self, settings: Settings | None = None, session_id: str = DEFAULT_SESSION_ID) -> None:
        self.settings = settings or get_settings()
        self.session_id = session_id
        ensure_data_dirs(self.settings)
        self.store = MetadataStore(self.settings.sqlite_path)
        self.store.init()
        self.store.ensure_session(DEFAULT_SESSION_ID, DEFAULT_SESSION_TITLE)
        self.store.ensure_session(self.session_id, DEFAULT_SESSION_TITLE if self.session_id == DEFAULT_SESSION_ID else self.session_id)
        self.store.mark_stale_processing_documents_failed()
        self.parser = PdfParser()
        self.chunker = Chunker()
        self.embedder = EmbeddingService(self.settings)
        self.vectors = LocalVectorStore(self.settings.indexes_dir / "vectors.json")
        self.sparse = BM25Index(self.settings.indexes_dir / "bm25.json")
        self.okf_vectors = LocalVectorStore(self.settings.indexes_dir / "okf_vectors.json")
        self.okf_sparse = BM25Index(self.settings.indexes_dir / "okf_bm25.json")
        self.reranker = Reranker()
        self.answerer = Answerer(self.settings)
        self.okf = OkfGenerator(self.settings.okf_dir, self.store)
        self.okf_importer = OkfImporter(self.settings.okf_dir, self.store)

    def list_sessions(self) -> list[ChatSession]:
        return self.store.list_sessions()

    def create_session(self, title: str | None = None) -> ChatSession:
        now = datetime.now()
        session_title = title or f"Chat {now:%Y-%m-%d %H:%M}"
        session_id = stable_id("session", session_title, now.isoformat(timespec="microseconds"))
        session = self.store.create_session(session_id, session_title)
        self.set_session(session.session_id)
        logger.info("created chat session: %s (%s)", session.title, session.session_id)
        return session

    def set_session(self, session_id: str) -> None:
        sessions = {session.session_id for session in self.store.list_sessions()}
        if session_id not in sessions:
            self.store.ensure_session(session_id, session_id)
        self.session_id = session_id
        logger.info("active chat session: %s", self.session_id)

    def init(self) -> None:
        ensure_data_dirs(self.settings)
        self.store.init()
        self._rebuild_indexes()

    def ingest(self, source_path: Path, build_okf: bool = True, force: bool = False) -> Document:
        started_at = time.perf_counter()
        source_path = source_path.resolve()
        logger.info("ingest start: %s session=%s", source_path, self.session_id)
        file_hash = sha256_file(source_path)
        existing = self.store.find_document_by_hash(file_hash, self.session_id)
        if existing and not force and existing.status == "ready" and self.store.count_chunks_for_document(existing.document_id) > 0:
            logger.info("ingest skipped existing ready document: %s", existing.filename)
            return existing

        document_id = existing.document_id if existing else stable_id("doc", self.session_id, source_path.name, file_hash)
        target = self.settings.documents_dir / f"{document_id}{source_path.suffix.lower()}"
        if source_path != target.resolve():
            shutil.copy2(source_path, target)
        document = Document(
            document_id=document_id,
            filename=source_path.name,
            sha256=file_hash,
            path=str(target),
            status="processing",
            session_id=self.session_id,
        )
        self.store.upsert_document(document)

        try:
            parse_started = time.perf_counter()
            pages = self.parser.parse(target)
            logger.info(
                "parsed %s page(s) from %s in %.2fs",
                len(pages),
                source_path.name,
                time.perf_counter() - parse_started,
            )
            pages = remove_repeated_headers_footers(pages)
            chunk_started = time.perf_counter()
            chunks = self.chunker.chunk_pages(document, pages)
            logger.info(
                "chunked %s into %s chunk(s) in %.2fs",
                source_path.name,
                len(chunks),
                time.perf_counter() - chunk_started,
            )
            if not chunks:
                raise ValueError("The document was parsed, but no searchable text chunks were created.")
        except Exception:
            self.store.update_document_status(document.document_id, "failed")
            raise

        ready_document = Document(
            document_id=document.document_id,
            filename=document.filename,
            sha256=document.sha256,
            path=document.path,
            status="ready",
            session_id=self.session_id,
        )
        self.store.upsert_document(ready_document)
        self.store.replace_chunks(document.document_id, chunks)
        index_started = time.perf_counter()
        self._index_chunks(chunks)
        self._rebuild_sparse()
        logger.info(
            "indexed %s chunk(s) for %s in %.2fs",
            len(chunks),
            source_path.name,
            time.perf_counter() - index_started,
        )
        if build_okf:
            okf_started = time.perf_counter()
            self.okf.generate_for_document(chunks)
            self._rebuild_okf_indexes()
            logger.info("generated OKF for %s in %.2fs", source_path.name, time.perf_counter() - okf_started)
        logger.info("ingest finished: %s in %.2fs", source_path.name, time.perf_counter() - started_at)
        return ready_document

    def repair_unready_documents(self, build_okf: bool = True) -> list[Document]:
        repaired: list[Document] = []
        for document in self.store.list_documents(self.session_id):
            chunk_count = self.store.count_chunks_for_document(document.document_id)
            if document.status == "ready" and chunk_count > 0:
                continue
            path = Path(document.path)
            if not path.exists():
                self.store.update_document_status(document.document_id, "failed")
                continue
            repaired.append(self.ingest(path, build_okf=build_okf, force=True))
        return repaired

    def import_okf_bundle(self, source_root: Path) -> list[OkfConcept]:
        concepts = self.okf_importer.import_bundle(source_root.resolve())
        self._rebuild_okf_indexes()
        return concepts

    def validate_okf_bundle(self, source_root: Path) -> list[dict[str, str]]:
        return [
            issue.__dict__
            for issue in validate_okf_bundle(source_root.resolve())
        ]

    def retrieve(self, question: str, include_debug: bool = False) -> tuple[list[Chunk], dict[str, object]]:
        started_at = time.perf_counter()
        logger.info("retrieve start: %r", question)
        chunks = self.store.list_chunks(self.session_id)
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        query_type = classify_query(question)
        fast_fact_query = self._is_fast_fact_query(question)
        topic_query = self._is_topic_query(question)
        search_question = self._normalize_search_question(question)
        dense_results: list[tuple[str, float]] = []
        okf_concept_results: list[tuple[str, float]] = []
        if fast_fact_query or topic_query:
            sparse_results = self.sparse.search(search_question, self.settings.sparse_top_k)
            okf_source_results = self._retrieve_okf_sources_sparse(search_question)
        else:
            query_embedding = self.embedder.embed([search_question])[0]
            dense_results = self.vectors.search(query_embedding, self.settings.dense_top_k)
            sparse_results = self.sparse.search(search_question, self.settings.sparse_top_k)
            okf_concept_results, okf_source_results = self._retrieve_okf_sources(
                search_question,
                query_embedding,
            )
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results, okf_source_results],
            top_k=self.settings.fusion_top_k,
        )
        candidates = [by_id[chunk_id] for chunk_id, _score in fused if chunk_id in by_id]
        context_limit = self._context_limit_for_query(query_type)
        if fast_fact_query:
            ranked_fast = self._rank_fast_fact_candidates(question, chunks, candidates)
            selected = ranked_fast[:context_limit]
            reranked = [(chunk, 0.0) for chunk in selected]
        elif topic_query:
            ranked_topic = self._rank_topic_candidates(search_question, chunks, candidates)
            selected = ranked_topic[:context_limit]
            reranked = [(chunk, 0.0) for chunk in selected]
        else:
            reranked = self.reranker.rerank(question, candidates, self.settings.rerank_top_k)
            selected = [chunk for chunk, _score in reranked[:context_limit]]
        elapsed = time.perf_counter() - started_at
        logger.info(
            "retrieve finished in %.2fs: session=%s chunks=%s candidates=%s selected=%s fast_fact=%s topic=%s",
            elapsed,
            self.session_id,
            len(chunks),
            len(candidates),
            len(selected),
            fast_fact_query,
            topic_query,
        )
        debug = {}
        if include_debug:
            debug = {
                "query_type": query_type,
                "dense_results": dense_results,
                "sparse_results": sparse_results,
                "okf_concept_results": okf_concept_results,
                "okf_source_results": okf_source_results,
                "fusion_results": fused,
                "selected_chunk_ids": [chunk.chunk_id for chunk in selected],
                "fast_fact_query": fast_fact_query,
                "topic_query": topic_query,
            }
        return selected, debug

    def ask(self, question: str, include_debug: bool = False) -> Answer:
        started_at = time.perf_counter()
        logger.info("ask start: %r", question)
        chunks, debug = self.retrieve(question, include_debug=include_debug)
        answer = self.answerer.answer(question, chunks, debug=debug)
        logger.info(
            "ask finished in %.2fs: answerable=%s citations=%s",
            time.perf_counter() - started_at,
            answer.answerable,
            len(answer.citations),
        )
        return answer

    def close(self) -> None:
        self.store.close()

    def _index_chunks(self, chunks: list[Chunk]) -> None:
        texts = [chunk.text for chunk in chunks]
        vectors = self.embedder.embed(texts)
        self.vectors.upsert_many(
            {chunk.chunk_id: vector for chunk, vector in zip(chunks, vectors)}
        )

    def _rebuild_indexes(self) -> None:
        chunks = self.store.list_chunks()
        if chunks:
            self._index_chunks(chunks)
        self._rebuild_sparse()
        self._rebuild_okf_indexes()

    def _rebuild_sparse(self) -> None:
        self.sparse.build(self.store.list_chunks())

    def _rebuild_okf_indexes(self) -> None:
        concept_chunks = self._concept_chunks(self.store.list_concepts())
        if concept_chunks:
            vectors = self.embedder.embed([chunk.text for chunk in concept_chunks])
            self.okf_vectors.upsert_many(
                {chunk.chunk_id: vector for chunk, vector in zip(concept_chunks, vectors)}
            )
        self.okf_sparse.build(concept_chunks)

    def _retrieve_okf_sources(
        self,
        question: str,
        query_embedding: list[float],
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        concepts = self.store.list_concepts()
        if not concepts:
            return [], []

        concept_by_chunk_id = {f"okf:{concept.concept_id}": concept for concept in concepts}
        okf_dense = self.okf_vectors.search(query_embedding, 10)
        okf_sparse = self.okf_sparse.search(question, 10)
        okf_concept_results = reciprocal_rank_fusion([okf_dense, okf_sparse], top_k=10)

        source_scores: dict[str, float] = {}
        for rank, (concept_chunk_id, score) in enumerate(okf_concept_results, start=1):
            concept = concept_by_chunk_id.get(concept_chunk_id)
            if not concept:
                continue
            rank_score = score + 1.0 / rank
            for source_chunk_id in concept.source_chunk_ids:
                source_scores[source_chunk_id] = max(
                    source_scores.get(source_chunk_id, 0.0),
                    rank_score,
                )

        okf_source_results = sorted(source_scores.items(), key=lambda item: item[1], reverse=True)
        return okf_concept_results, okf_source_results[: self.settings.sparse_top_k]

    def _retrieve_okf_sources_sparse(self, question: str) -> list[tuple[str, float]]:
        concepts = self.store.list_concepts()
        if not concepts:
            return []

        concept_by_chunk_id = {f"okf:{concept.concept_id}": concept for concept in concepts}
        okf_sparse = self.okf_sparse.search(question, 10)
        source_scores: dict[str, float] = {}
        for rank, (concept_chunk_id, score) in enumerate(okf_sparse, start=1):
            concept = concept_by_chunk_id.get(concept_chunk_id)
            if not concept:
                continue
            rank_score = score + 1.0 / rank
            for source_chunk_id in concept.source_chunk_ids:
                source_scores[source_chunk_id] = max(
                    source_scores.get(source_chunk_id, 0.0),
                    rank_score,
                )
        return sorted(source_scores.items(), key=lambda item: item[1], reverse=True)[
            : self.settings.sparse_top_k
        ]

    def _concept_chunks(self, concepts: list[OkfConcept]) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=f"okf:{concept.concept_id}",
                document_id="okf",
                filename=f"{concept.slug}.md",
                page_start=1,
                page_end=1,
                section_path=("OKF", concept.title),
                text=self._concept_search_text(concept),
                chunk_type="okf_concept",
                metadata={
                    "concept_id": concept.concept_id,
                    "source_chunk_ids": ",".join(concept.source_chunk_ids),
                    "tags": ",".join(concept.tags),
                    "related": ",".join(concept.related),
                    "depends_on": ",".join(concept.depends_on),
                },
            )
            for concept in concepts
        ]

    def _concept_search_text(self, concept: OkfConcept) -> str:
        metadata_text = "\n".join(
            [
                f"Title: {concept.title}",
                f"Aliases: {', '.join(concept.aliases)}",
                f"Tags: {', '.join(concept.tags)}",
                f"Related: {', '.join(concept.related)}",
                f"Depends on: {', '.join(concept.depends_on)}",
            ]
        )
        return f"{metadata_text}\n\n{concept.text}"

    def _context_limit_for_query(self, query_type: str) -> int:
        if query_type in {"direct_factual", "exact_identifier", "numeric_or_table", "follow_up_or_short"}:
            return min(self.settings.final_context_chunks, 4)
        return self.settings.final_context_chunks

    def _is_fast_fact_query(self, question: str) -> bool:
        normalized = question.lower()
        fact_terms = {
            "full name",
            "person name",
            "candidate name",
            "user name",
            "student name",
            "name",
            "college",
            "collage",
            "university",
            "institute",
            "school",
            "degree",
            "course",
            "branch",
            "program",
            "cgpa",
            "gpa",
            "email",
            "mail",
            "phone",
            "mobile",
            "contact",
        }
        return any(term in normalized for term in fact_terms)

    def _is_topic_query(self, question: str) -> bool:
        normalized = self._normalize_search_question(question).lower()
        if self._is_fast_fact_query(normalized):
            return False
        return bool(
            re.search(
                r"\b(what is|what are|define|explain|describe|"
                r"tell me(?: everything| all)? about|everything about)\b",
                normalized,
            )
            or "in detail" in normalized
            or "details about" in normalized
        )

    def _normalize_search_question(self, question: str) -> str:
        normalized = question.replace("collage", "college").replace("Collage", "College")
        normalized = normalized.replace("transection", "transaction").replace("Transection", "Transaction")
        normalized = normalized.replace("whats", "what is").replace("Whats", "What is")
        return normalized

    def _rank_fast_fact_candidates(
        self,
        question: str,
        all_chunks: list[Chunk],
        retrieved_chunks: list[Chunk],
    ) -> list[Chunk]:
        retrieved_ids = {chunk.chunk_id for chunk in retrieved_chunks}
        scored = [
            (
                chunk,
                self._fast_fact_score(question, chunk) + (2.0 if chunk.chunk_id in retrieved_ids else 0.0),
            )
            for chunk in all_chunks
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        ranked = [chunk for chunk, score in scored if score > 0]
        if ranked:
            return ranked
        return retrieved_chunks or all_chunks

    def _rank_topic_candidates(
        self,
        question: str,
        all_chunks: list[Chunk],
        retrieved_chunks: list[Chunk],
    ) -> list[Chunk]:
        retrieved_ids = {chunk.chunk_id for chunk in retrieved_chunks}
        phrase = self._topic_phrase(question)
        scored = [
            (
                chunk,
                self._topic_score(question, phrase, chunk)
                + (1.5 if chunk.chunk_id in retrieved_ids else 0.0),
            )
            for chunk in all_chunks
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        ranked = [chunk for chunk, score in scored if score > 0]
        if ranked:
            return ranked
        return retrieved_chunks or all_chunks

    def _topic_phrase(self, question: str) -> str:
        normalized = self._normalize_search_question(question).lower()
        normalized = re.sub(
            r"\b(what|is|are|the|a|an|define|explain|describe|tell|me|"
            r"everything|all|about|in|detail|details|please|also)\b",
            " ",
            normalized,
        )
        return " ".join(re.findall(r"[a-z0-9]+", normalized))

    def _topic_score(self, question: str, phrase: str, chunk: Chunk) -> float:
        text = chunk.text
        lower_text = self._normalize_search_question(text).lower()
        terms = self._topic_terms(question)
        score = 0.0
        if phrase and phrase in lower_text:
            score += 30.0
            first_position = lower_text.find(phrase)
            if first_position < 500:
                score += 8.0
        lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
        if phrase and any(phrase == self._topic_phrase(line) for line in lines[:12]):
            score += 12.0
        if terms:
            present = sum(1 for term in terms if term in lower_text)
            score += present * 3.0
            if present == len(terms):
                score += 8.0
        if "transaction" in terms and "states" in terms:
            state_terms = ["active", "partially committed", "failed", "aborted", "committed", "terminated"]
            score += sum(2.0 for term in state_terms if term in lower_text)
        return score

    def _topic_terms(self, question: str) -> list[str]:
        phrase = self._topic_phrase(question)
        return [term for term in re.findall(r"[a-z0-9]+", phrase) if len(term) > 2]

    def _fast_fact_score(self, question: str, chunk: Chunk) -> float:
        normalized = question.lower()
        text = chunk.text
        lower_text = text.lower()
        score = 0.0
        if "resume" in chunk.filename.lower() or "cv" in chunk.filename.lower():
            score += 0.5
        if "name" in normalized:
            if re.search(r"(?:^|\n|\b)(?:name|full name|candidate name|student name)\s*[:\-]", text):
                score += 8.0
            first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
            if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", first_line):
                score += 6.0
            if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text):
                score += 2.0
        if any(term in normalized for term in ["college", "collage", "university", "institute", "school"]):
            if re.search(r"\b(?:Institute|University|College|School)\b", text):
                score += 8.0
            if "education" in lower_text:
                score += 2.0
        if any(term in normalized for term in ["degree", "course", "branch", "program"]):
            if re.search(r"\b(?:B\.?\s?Tech|BTECH|M\.?\s?Tech|MTECH|BCA|MCA|MBA)\b", text, re.IGNORECASE):
                score += 8.0
            if "education" in lower_text:
                score += 2.0
        if "cgpa" in normalized or "gpa" in normalized:
            if re.search(r"\bC?GPA\b", text, re.IGNORECASE):
                score += 8.0
        if "email" in normalized or "mail" in normalized:
            if re.search(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", text, re.IGNORECASE):
                score += 8.0
        if any(term in normalized for term in ["phone", "mobile", "contact"]):
            if re.search(r"\+?\d[\d\s().-]{7,}\d", text):
                score += 8.0
        return score

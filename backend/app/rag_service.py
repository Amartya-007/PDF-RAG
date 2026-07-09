from __future__ import annotations

import logging
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from backend.app.core.config import Settings, ensure_data_dirs, get_settings
from backend.app.core.hashing import sha256_file, stable_id
from backend.app.database.store import DEFAULT_SESSION_ID, DEFAULT_SESSION_TITLE, MetadataStore
from backend.app.domain.exceptions import IngestionError
from backend.app.generation.answerer import Answerer
from backend.app.indexing.embeddings import EmbeddingService
from backend.app.indexing.sparse import BM25Index
from backend.app.indexing.vector_store import LocalVectorStore
from backend.app.ingestion.chunking import Chunker
from backend.app.ingestion.cleaning import remove_repeated_headers_footers
from backend.app.ingestion.parser.pdf_parser import PdfParser
from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.knowledge.okf import OkfConcept, validate_okf_bundle
from backend.app.knowledge.okf_generator import OkfGenerator
from backend.app.knowledge.okf_importer import OkfImporter
from backend.app.models import Answer, ChatSession, Chunk, Document
from backend.app.retrieval.context_builder import build_evidence_block
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.retrieval.query_classifier import QueryClassifier
from backend.app.retrieval.reranking import Reranker
from backend.app.generation.ollama_client import OllamaClient
from backend.app.indexing.tree_indexer import TreeIndexer
from backend.app.retrieval.tree_retriever import TreeRetriever


logger = logging.getLogger(__name__)


class RagService:
    def __init__(self, settings: Settings | None = None, session_id: str = DEFAULT_SESSION_ID) -> None:
        self.settings = settings or get_settings()
        self.session_id = session_id
        ensure_data_dirs(self.settings)

        self.store = MetadataStore(self.settings.sqlite_path)
        self.store.init()
        self.store.ensure_session(DEFAULT_SESSION_ID, DEFAULT_SESSION_TITLE)
        self.store.ensure_session(
            self.session_id,
            DEFAULT_SESSION_TITLE if self.session_id == DEFAULT_SESSION_ID else self.session_id,
        )
        self.store.mark_stale_processing_documents_failed()

        self.embedder = EmbeddingService(self.settings)
        self.vectors = LocalVectorStore(self.settings.indexes_dir / "vectors.json")
        self.sparse = BM25Index(self.settings.indexes_dir / "bm25.json")
        self.okf_vectors = LocalVectorStore(self.settings.indexes_dir / "okf_vectors.json")
        self.okf_sparse = BM25Index(self.settings.indexes_dir / "okf_bm25.json")
        self.reranker = Reranker()
        self.answerer = Answerer(self.settings)
        self.okf = OkfGenerator(self.settings.okf_dir, self.store)
        self.okf_importer = OkfImporter(self.settings.okf_dir, self.store)
        self._classifier = QueryClassifier()
        self._chunker = Chunker()
        self._parser = PdfParser()
        self._pipeline = IngestionPipeline(
            store=self.store,
            parser=self._parser,
            chunker=self._chunker,
            embedder=self.embedder,
            vectors=self.vectors,
            sparse=self.sparse,
            documents_dir=self.settings.documents_dir,
            embedding_batch_size=self.settings.embedding_batch_size,
        )
        # ── Vectorless RAG ────────────────────────────────────────────
        self._ollama_client = OllamaClient(self.settings) if self.settings.use_ollama else None
        self.tree_indexer = TreeIndexer(
            trees_dir=self.settings.trees_dir,
            ollama_client=self._ollama_client,
        )
        self.tree_retriever = TreeRetriever(ollama_client=self._ollama_client)
        # Chunk cache: invalidated on ingest/delete, avoids full SQLite scan per query
        self._chunk_cache: dict[str, list[Chunk]] | None = None
        self._chunk_cache_session: str | None = None

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

    def ingest(
        self,
        source_path: Path,
        build_okf: bool = False,
        force: bool = False,
        progress_callback: object = None,
    ) -> Document:
        """Ingest a document via the dedicated IngestionPipeline.

        Args:
            source_path:       Path to the source file.
            build_okf:         Generate OKF knowledge graph after embedding.
            force:             Re-ingest even if already indexed.
            progress_callback: Optional ``(done, total, msg)`` callable for UI.

        Returns:
            Ready ``Document`` record.
        """
        ready_doc = self._pipeline.ingest(
            source_path,
            self.session_id,
            force=force,
            progress=progress_callback,  # type: ignore[arg-type]
        )
        self._invalidate_chunk_cache()

        if build_okf:
            chunks = self.store.list_chunks_for_document(ready_doc.document_id)
            if chunks:
                t0 = time.perf_counter()
                self.okf.generate_for_document(chunks)
                self._rebuild_okf_indexes()
                logger.info("OKF generated in %.2fs", time.perf_counter() - t0)

        # ── Legacy vectorless tree index (superseded by DocumentNode tree
        # in backend/app/domain — see tasks.md task 6.2 deprecation note).
        # Kept as a best-effort, non-fatal enrichment for the old
        # TreeRetriever path; failures here must never break ingestion.
        if self.settings.use_tree_search and self._ollama_client is not None:
            try:
                t_tree = time.perf_counter()
                pages = self._parser.parse(source_path)
                tree = self.tree_indexer.build(ready_doc.document_id, ready_doc.filename, pages)
                self.tree_indexer.save(tree)
                self.tree_indexer.save_with_raw(tree)
                logger.info(
                    "tree index built (%d nodes) in %.2fs",
                    len(tree.all_nodes()), time.perf_counter() - t_tree,
                )
            except Exception as exc:
                logger.warning("tree indexing failed (non-fatal): %s", exc)

        logger.info("ingest finished: %s", source_path.name)
        return ready_doc

    def _index_chunks_with_progress(
        self,
        chunks: list[Chunk],
        progress_callback: object = None,
    ) -> None:
        """Embed chunks that aren't already in the vector store.

        Reports per-batch progress so the UI can show a live counter.
        Uses the largest feasible batch size to minimise Ollama round-trips.
        """
        existing = self.vectors.existing_ids()
        new_chunks = [c for c in chunks if c.chunk_id not in existing]
        if not new_chunks:
            logger.info("embedding skipped — all %d chunks already cached", len(chunks))
            return

        total = len(new_chunks)
        batch_size = max(16, self.settings.embedding_batch_size)
        done = 0

        # Import here to keep the top-level import clean
        from backend.app.core.text import batched as _batched

        for batch in _batched(new_chunks, batch_size):
            texts = [c.text for c in batch]
            try:
                vectors = self.embedder.embed(texts)
            except Exception as exc:
                logger.error("embedding batch failed: %s — skipping", exc)
                done += len(batch)
                continue

            self.vectors.upsert_many(
                {c.chunk_id: v for c, v in zip(batch, vectors)}
            )
            done += len(batch)
            if progress_callback:
                progress_callback(
                    done,
                    total,
                    f"Embedding {done}/{total} chunks…",
                )

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
        """Run the hybrid retrieval pipeline for *question*.

        Uses QueryClassifier to choose between:
        - FAST_FACT / TOPIC: BM25-only, no embedding call needed
        - All others: parallel dense + sparse + OKF → RRF → rerank
        """
        started_at = time.perf_counter()
        logger.info("retrieve start: %r", question)

        chunks = self._get_chunks(self.session_id)
        by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}

        query_type = self._classifier.classify(question)
        normalised = self._classifier.normalise(question)
        is_fast_fact = self._classifier.is_fast_fact(question)
        is_topic = self._classifier.is_topic(question)

        dense_results: list[tuple[str, float]] = []
        sparse_results: list[tuple[str, float]] = []
        okf_concept_results: list[tuple[str, float]] = []
        okf_source_results: list[tuple[str, float]] = []
        query_embedding: list[float] = []

        if is_fast_fact or is_topic:
            sparse_results = self.sparse.search(normalised, self.settings.sparse_top_k)
            okf_source_results = self._retrieve_okf_sources_sparse(normalised)
        else:
            def _dense() -> tuple[list[float], list[tuple[str, float]]]:
                emb = self.embedder.embed([normalised])[0]
                return emb, self.vectors.search(emb, self.settings.dense_top_k)

            def _sparse() -> list[tuple[str, float]]:
                return self.sparse.search(normalised, self.settings.sparse_top_k)

            with ThreadPoolExecutor(max_workers=2) as pool:
                dense_fut = pool.submit(_dense)
                sparse_fut = pool.submit(_sparse)
                query_embedding, dense_results = dense_fut.result()
                sparse_results = sparse_fut.result()

            okf_concept_results, okf_source_results = self._retrieve_okf_sources(
                normalised, query_embedding
            )

        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results, okf_source_results],
            top_k=self.settings.fusion_top_k,
        )
        hybrid_candidates = [by_id[chunk_id] for chunk_id, _score in fused if chunk_id in by_id]
        context_limit = self._context_limit_for_query(query_type)

        # ── Vectorless tree retrieval (primary path) ──────────────────────
        # When Ollama is enabled and tree search is on, let llama3.2 navigate
        # the document hierarchy directly instead of relying on vector
        # similarity. Tree chunks come first; hybrid fills any gaps.
        tree_chunks: list[Chunk] = []
        if self.settings.use_tree_search and self._ollama_client is not None:
            documents = self.store.list_documents(self.session_id)
            trees = []
            for doc in documents:
                t = self.tree_indexer.load_with_raw(doc.document_id)
                if t is not None:
                    trees.append(t)
            if trees:
                try:
                    tree_chunks = self.tree_retriever.retrieve(question, trees)
                    logger.info("tree retrieval: %d chunks from %d trees", len(tree_chunks), len(trees))
                except Exception as exc:
                    logger.warning("tree retrieval failed (non-fatal): %s", exc)

        # Merge: tree chunks first, then hybrid without duplicates
        seen_ids = {c.chunk_id for c in tree_chunks}
        merged = list(tree_chunks)
        for chunk in hybrid_candidates:
            if chunk.chunk_id not in seen_ids:
                merged.append(chunk)
                seen_ids.add(chunk.chunk_id)
        candidates = merged

        if is_fast_fact:
            selected = self._rank_fast_fact_candidates(question, chunks, candidates)[:context_limit]
        elif is_topic:
            selected = self._rank_topic_candidates(normalised, chunks, candidates)[:context_limit]
        else:
            reranked = self.reranker.rerank(question, candidates, self.settings.rerank_top_k)
            selected = [c for c, _ in reranked[:context_limit]]

        elapsed = time.perf_counter() - started_at
        logger.info(
            "retrieve finished in %.2fs: session=%s chunks=%d candidates=%d selected=%d "
            "query_type=%s",
            elapsed, self.session_id, len(chunks), len(candidates), len(selected),
            query_type.value,
        )

        debug: dict[str, object] = {}
        if include_debug:
            debug = {
                "query_type": query_type.value,
                "dense_results": dense_results,
                "sparse_results": sparse_results,
                "okf_concept_results": okf_concept_results,
                "okf_source_results": okf_source_results,
                "fusion_results": fused,
                "selected_chunk_ids": [c.chunk_id for c in selected],
                "is_fast_fact": is_fast_fact,
                "is_topic": is_topic,
                "fast_fact_query": is_fast_fact,
                "topic_query": is_topic,
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

    def delete_document(self, document_id: str) -> None:
        """Remove a document, its chunks, and its vectors from all indexes."""
        chunk_ids = self.store.delete_document(document_id)
        remaining_ids = self.vectors.existing_ids() - set(chunk_ids)
        self.vectors.delete_missing(remaining_ids)
        self.sparse.remove_chunks(chunk_ids)
        self._invalidate_chunk_cache()

    def delete_session_documents(self, session_id: str) -> None:
        """Remove all documents/chunks/vectors for a session."""
        docs = self.store.list_documents(session_id)
        all_chunk_ids: set[str] = set()
        for doc in docs:
            chunk_rows = self.store.list_chunks_for_document(doc.document_id)
            all_chunk_ids.update(c.chunk_id for c in chunk_rows)
        self.store.delete_session(session_id)
        remaining_ids = self.vectors.existing_ids() - all_chunk_ids
        self.vectors.delete_missing(remaining_ids)
        self._rebuild_sparse()
        self._invalidate_chunk_cache()

    def _invalidate_chunk_cache(self) -> None:
        self._chunk_cache = None
        self._chunk_cache_session = None

    def _get_chunks(self, session_id: str) -> list[Chunk]:
        """Return chunks for session, using in-memory cache to avoid repeated SQLite scans."""
        if self._chunk_cache is not None and self._chunk_cache_session == session_id:
            return self._chunk_cache
        chunks = self.store.list_chunks(session_id)
        self._chunk_cache = chunks
        self._chunk_cache_session = session_id
        return chunks

    def close(self) -> None:
        self.store.close()

    def _rebuild_indexes(self) -> None:
        """Rebuild all indexes from SQLite on startup (after crash recovery)."""
        chunks = self.store.list_chunks()
        if chunks:
            # Re-embed any chunks not already in the vector store
            self._pipeline._phase2_embed(chunks, progress=None)
        self._rebuild_sparse()
        self._rebuild_okf_indexes()

    def _rebuild_sparse(self) -> None:
        # Rebuild over all chunks in the current session only.
        # For multi-session use the sparse index covers all sessions'
        # chunks because BM25 is session-agnostic at query time — the
        # session filter is applied at the vector search level.
        self.sparse.build(self.store.list_chunks())

    def _rebuild_okf_indexes(self) -> None:
        # Same logic as _index_chunks: concept_id (and therefore chunk_id
        # "okf:{concept_id}") is derived from the concept's source chunk ids,
        # so an unchanged concept already has a valid, reusable embedding.
        # Without this skip, every single document ingest re-embedded EVERY
        # OKF concept ever generated - an O(n^2) cost across a growing
        # library that dominated ingestion time once you had more than a
        # handful of documents.
        concept_chunks = self._concept_chunks(self.store.list_concepts())
        existing = self.okf_vectors.existing_ids()
        new_concept_chunks = [c for c in concept_chunks if c.chunk_id not in existing]
        if new_concept_chunks:
            vectors = self.embedder.embed([chunk.text for chunk in new_concept_chunks])
            self.okf_vectors.upsert_many(
                {chunk.chunk_id: vector for chunk, vector in zip(new_concept_chunks, vectors)}
            )
        # Prune embeddings for concepts that were regenerated/removed so the
        # vector store doesn't grow unbounded with stale entries.
        self.okf_vectors.delete_missing({chunk.chunk_id for chunk in concept_chunks})
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
        normalized = self._classifier.normalise(question).lower()
        normalized = re.sub(
            r"\b(what|is|are|the|a|an|define|explain|describe|tell|me|"
            r"everything|all|about|in|detail|details|please|also)\b",
            " ",
            normalized,
        )
        return " ".join(re.findall(r"[a-z0-9]+", normalized))

    def _topic_score(self, question: str, phrase: str, chunk: Chunk) -> float:
        text = chunk.text
        lower_text = self._classifier.normalise(text).lower()
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

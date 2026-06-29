from __future__ import annotations

import shutil
from pathlib import Path

from backend.app.core.config import Settings, ensure_data_dirs, get_settings
from backend.app.core.hashing import sha256_file, stable_id
from backend.app.database.store import MetadataStore
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
from backend.app.models import Answer, Chunk, Document
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.retrieval.query_analysis import classify_query
from backend.app.retrieval.reranking import Reranker


class RagService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        ensure_data_dirs(self.settings)
        self.store = MetadataStore(self.settings.sqlite_path)
        self.store.init()
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

    def init(self) -> None:
        ensure_data_dirs(self.settings)
        self.store.init()
        self._rebuild_indexes()

    def ingest(self, source_path: Path, build_okf: bool = True, force: bool = False) -> Document:
        source_path = source_path.resolve()
        file_hash = sha256_file(source_path)
        existing = self.store.find_document_by_hash(file_hash)
        if existing and not force and existing.status == "ready" and self.store.count_chunks_for_document(existing.document_id) > 0:
            return existing

        document_id = existing.document_id if existing else stable_id("doc", source_path.name, file_hash)
        target = self.settings.documents_dir / f"{document_id}{source_path.suffix.lower()}"
        if source_path != target.resolve():
            shutil.copy2(source_path, target)
        document = Document(
            document_id=document_id,
            filename=source_path.name,
            sha256=file_hash,
            path=str(target),
            status="processing",
        )
        self.store.upsert_document(document)

        try:
            pages = self.parser.parse(target)
            pages = remove_repeated_headers_footers(pages)
            chunks = self.chunker.chunk_pages(document, pages)
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
        )
        self.store.upsert_document(ready_document)
        self.store.replace_chunks(document.document_id, chunks)
        self._index_chunks(chunks)
        self._rebuild_sparse()
        if build_okf:
            self.okf.generate_for_document(chunks)
            self._rebuild_okf_indexes()
        return ready_document

    def repair_unready_documents(self, build_okf: bool = True) -> list[Document]:
        repaired: list[Document] = []
        for document in self.store.list_documents():
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
        chunks = self.store.list_chunks()
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        query_type = classify_query(question)
        fast_fact_query = self._is_fast_fact_query(question)
        search_question = self._normalize_search_question(question)
        dense_results: list[tuple[str, float]] = []
        okf_concept_results: list[tuple[str, float]] = []
        if fast_fact_query:
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
        reranked = self.reranker.rerank(question, candidates, self.settings.rerank_top_k)
        context_limit = self._context_limit_for_query(query_type)
        selected = [chunk for chunk, _score in reranked[:context_limit]]
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
            }
        return selected, debug

    def ask(self, question: str, include_debug: bool = False) -> Answer:
        chunks, debug = self.retrieve(question, include_debug=include_debug)
        return self.answerer.answer(question, chunks, debug=debug)

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

    def _normalize_search_question(self, question: str) -> str:
        normalized = question.replace("collage", "college").replace("Collage", "College")
        normalized = normalized.replace("whats", "what is").replace("Whats", "What is")
        return normalized

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
from backend.app.knowledge.okf_generator import OkfGenerator
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
        self.parser = PdfParser()
        self.chunker = Chunker()
        self.embedder = EmbeddingService(self.settings)
        self.vectors = LocalVectorStore(self.settings.indexes_dir / "vectors.json")
        self.sparse = BM25Index(self.settings.indexes_dir / "bm25.json")
        self.reranker = Reranker()
        self.answerer = Answerer(self.settings)
        self.okf = OkfGenerator(self.settings.okf_dir, self.store)

    def init(self) -> None:
        ensure_data_dirs(self.settings)
        self.store.init()
        self._rebuild_indexes()

    def ingest(self, source_path: Path, build_okf: bool = True) -> Document:
        source_path = source_path.resolve()
        file_hash = sha256_file(source_path)
        existing = self.store.find_document_by_hash(file_hash)
        if existing:
            return existing

        document_id = stable_id("doc", source_path.name, file_hash)
        target = self.settings.documents_dir / f"{document_id}{source_path.suffix.lower()}"
        shutil.copy2(source_path, target)
        document = Document(
            document_id=document_id,
            filename=source_path.name,
            sha256=file_hash,
            path=str(target),
            status="processing",
        )
        self.store.upsert_document(document)

        pages = self.parser.parse(target)
        pages = remove_repeated_headers_footers(pages)
        chunks = self.chunker.chunk_pages(document, pages)

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
        return ready_document

    def retrieve(self, question: str, include_debug: bool = False) -> tuple[list[Chunk], dict[str, object]]:
        chunks = self.store.list_chunks()
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        query_embedding = self.embedder.embed([question])[0]
        dense_results = self.vectors.search(query_embedding, self.settings.dense_top_k)
        sparse_results = self.sparse.search(question, self.settings.sparse_top_k)
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            top_k=self.settings.fusion_top_k,
        )
        candidates = [by_id[chunk_id] for chunk_id, _score in fused if chunk_id in by_id]
        reranked = self.reranker.rerank(question, candidates, self.settings.rerank_top_k)
        selected = [chunk for chunk, _score in reranked[: self.settings.final_context_chunks]]
        debug = {}
        if include_debug:
            debug = {
                "query_type": classify_query(question),
                "dense_results": dense_results,
                "sparse_results": sparse_results,
                "fusion_results": fused,
                "selected_chunk_ids": [chunk.chunk_id for chunk in selected],
            }
        return selected, debug

    def ask(self, question: str, include_debug: bool = False) -> Answer:
        chunks, debug = self.retrieve(question, include_debug=include_debug)
        return self.answerer.answer(question, chunks, debug=debug)

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

    def _rebuild_sparse(self) -> None:
        self.sparse.build(self.store.list_chunks())

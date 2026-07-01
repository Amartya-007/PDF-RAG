from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    sqlite_path_value: str
    ollama_base_url: str
    generation_model: str
    development_model: str
    active_model: str
    embedding_model: str
    reranker_model: str
    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rerank_top_k: int
    final_context_chunks: int
    temperature: float
    embedding_batch_size: int
    use_ollama: bool
    allow_hash_embeddings: bool
    force_ocr: bool

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def okf_dir(self) -> Path:
        return self.data_dir / "knowledge"

    @property
    def sqlite_path(self) -> Path | str:
        if self.sqlite_path_value == ":memory:":
            return ":memory:"
        return Path(self.sqlite_path_value).resolve()


def get_settings() -> Settings:
    data_dir = Path(os.getenv("RAG_DATA_DIR", "backend/data")).resolve()
    sqlite_path = os.getenv("RAG_SQLITE_PATH", str(data_dir / "metadata.sqlite3"))
    return Settings(
        data_dir=data_dir,
        sqlite_path_value=sqlite_path,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        generation_model=os.getenv("RAG_GENERATION_MODEL", "qwen3.5:9b"),
        development_model=os.getenv("RAG_DEVELOPMENT_MODEL", "qwen3.5:4b"),
        active_model=os.getenv("RAG_ACTIVE_MODEL", os.getenv("RAG_DEVELOPMENT_MODEL", "qwen3.5:4b")),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "qwen3-embedding:4b"),
        reranker_model=os.getenv("RAG_RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B"),
        dense_top_k=int(os.getenv("RAG_DENSE_TOP_K", "40")),
        sparse_top_k=int(os.getenv("RAG_SPARSE_TOP_K", "40")),
        fusion_top_k=int(os.getenv("RAG_FUSION_TOP_K", "50")),
        rerank_top_k=int(os.getenv("RAG_RERANK_TOP_K", "30")),
        final_context_chunks=int(os.getenv("RAG_FINAL_CONTEXT_CHUNKS", "8")),
        temperature=float(os.getenv("RAG_TEMPERATURE", "0.1")),
        # Large batch: send up to 64 chunks per Ollama call.
        # Ollama batches them internally — one model load, many embeddings.
        # Previously 32 was used but Ollama handles larger batches fine.
        embedding_batch_size=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "64")),
        use_ollama=_bool_env("RAG_USE_OLLAMA", False),
        allow_hash_embeddings=_bool_env("RAG_ALLOW_HASH_EMBEDDINGS", True),
        force_ocr=_bool_env("RAG_FORCE_OCR", False),
    )


def ensure_data_dirs(settings: Settings) -> None:
    for path in [
        settings.data_dir,
        settings.documents_dir,
        settings.indexes_dir,
        settings.okf_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

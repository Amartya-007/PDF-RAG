"""Embedding service using the official Ollama Python SDK.

Two-level cache
---------------
Level 1 — in-memory LRU (OrderedDict, O(1) move-to-end):
    Avoids re-embedding identical texts within a session.
Level 2 — SQLite disk cache (sha256 → BLOB):
    Survives restarts; guarantees a 232-chunk PDF is never re-embedded.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np

from backend.app.core.config import Settings
from backend.app.core.text import batched
from backend.app.domain.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

_CACHE_DB = "embedding_cache.db"
_MEMORY_CACHE_MAX = 512
_EMBED_WORKERS = 3


class EmbeddingService:
    """Embed texts via Ollama SDK with parallel batching and persistent cache.

    Implements the ``EmbeddingProvider`` port.

    Performance characteristics
    ---------------------------
    - Memory LRU lookup/insert: O(1) via OrderedDict.move_to_end
    - Disk cache lookup: O(1) via SQLite primary-key index on sha256
    - Batch embedding: up to _EMBED_WORKERS parallel HTTP calls to Ollama
    """

    def __init__(self, settings: Settings, dimensions: int = 384) -> None:
        self.settings = settings
        self.dimensions = dimensions
        self._lock = threading.Lock()

        # In-memory LRU — OrderedDict gives O(1) move-to-end
        self._mem: OrderedDict[str, list[float]] = OrderedDict()

        # SQLite disk cache
        settings.indexes_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = settings.indexes_dir / _CACHE_DB
        self._db: Optional[sqlite3.Connection] = None
        self._open_db()

        # Ollama SDK client (lazy — only created on first use)
        self._ollama_client: object = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _open_db(self) -> None:
        """Open SQLite with WAL + check_same_thread=False for worker threads."""
        self._db = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(sha256 TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.commit()
        count = self._db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        logger.info("embedding disk cache opened: %d entries", count)

    def _get_ollama(self) -> object:
        if self._ollama_client is None:
            try:
                import ollama  # type: ignore
                self._ollama_client = ollama.Client(host=self.settings.ollama_base_url)
            except ImportError:
                logger.warning("ollama SDK not installed; falling back to urllib")
                self._ollama_client = False
        return self._ollama_client

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None

    def flush_disk_cache(self) -> None:
        """No-op: SQLite commits are immediate after each batch."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text.

        Raises:
            EmbeddingError: if Ollama fails and hash fallback is disabled.
        """
        if not texts:
            return []
        if self.settings.use_ollama:
            try:
                return self._embed_with_cache(texts)
            except EmbeddingError:
                if not self.settings.allow_hash_embeddings:
                    raise
                logger.warning("Ollama embedding failed — using hash fallback")
        return [self._hash_embedding(text) for text in texts]

    # ------------------------------------------------------------------
    # Cache internals
    # ------------------------------------------------------------------

    def _text_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _embed_with_cache(self, texts: list[str]) -> list[list[float]]:
        results: list[Optional[list[float]]] = [None] * len(texts)
        db_lookup: list[tuple[int, str, str]] = []

        # 1. Memory LRU — O(1) per hit
        for i, text in enumerate(texts):
            if text in self._mem:
                results[i] = self._mem[text]
                self._mem.move_to_end(text)   # O(1)
            else:
                db_lookup.append((i, text, self._text_key(text)))

        # 2. Batch SQLite lookup
        uncached_idx: list[int] = []
        uncached_txt: list[str] = []
        uncached_keys: list[str] = []

        if db_lookup and self._db:
            keys = [k for _, _, k in db_lookup]
            placeholders = ",".join("?" for _ in keys)
            with self._lock:
                rows = {
                    row[0]: row[1]
                    for row in self._db.execute(
                        f"SELECT sha256, vector FROM embeddings WHERE sha256 IN ({placeholders})",
                        keys,
                    ).fetchall()
                }
            for i, text, key in db_lookup:
                if key in rows:
                    emb = np.frombuffer(rows[key], dtype=np.float32).tolist()
                    results[i] = emb
                    self._mem_put(text, emb)
                else:
                    uncached_idx.append(i)
                    uncached_txt.append(text)
                    uncached_keys.append(key)

        # 3. Embed uncached texts (parallel batches)
        if uncached_txt:
            new_embeddings = self._embed_parallel(uncached_txt)
            rows_to_insert = []
            for i, text, key, emb in zip(
                uncached_idx, uncached_txt, uncached_keys, new_embeddings
            ):
                results[i] = emb
                self._mem_put(text, emb)
                rows_to_insert.append((key, np.array(emb, dtype=np.float32).tobytes()))

            if rows_to_insert and self._db:
                with self._lock:
                    self._db.executemany(
                        "INSERT OR REPLACE INTO embeddings(sha256, vector) VALUES(?,?)",
                        rows_to_insert,
                    )
                    self._db.commit()

        return results  # type: ignore[return-value]

    def _mem_put(self, text: str, emb: list[float]) -> None:
        """Insert into LRU, evicting the oldest entry if at capacity. O(1)."""
        if text in self._mem:
            self._mem.move_to_end(text)
        else:
            if len(self._mem) >= _MEMORY_CACHE_MAX:
                self._mem.popitem(last=False)  # remove LRU entry — O(1)
            self._mem[text] = emb

    # ------------------------------------------------------------------
    # Parallel Ollama embedding
    # ------------------------------------------------------------------

    def _embed_parallel(self, texts: list[str]) -> list[list[float]]:
        batch_size = max(16, self.settings.embedding_batch_size)
        batches = list(batched(texts, batch_size))

        if len(batches) == 1:
            return self._embed_one_batch(batches[0])

        results_by_batch: dict[int, list[list[float]]] = {}
        workers = min(_EMBED_WORKERS, len(batches))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._embed_one_batch, batch): i
                for i, batch in enumerate(batches)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results_by_batch[idx] = fut.result()
                except Exception as exc:
                    logger.error("parallel embed batch %d failed: %s", idx, exc)
                    results_by_batch[idx] = [[0.0] * self.dimensions for _ in batches[idx]]

        all_embeddings: list[list[float]] = []
        for i in range(len(batches)):
            all_embeddings.extend(results_by_batch[i])
        return all_embeddings

    def _embed_one_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._get_ollama()
        if client:
            return self._embed_sdk(client, texts)
        return self._embed_urllib(texts)

    def _embed_sdk(self, client: object, texts: list[str]) -> list[list[float]]:
        try:
            response = client.embed(  # type: ignore[union-attr]
                model=self.settings.embedding_model,
                input=texts,
                options={"keep_alive": 600},
            )
            embeddings = (
                response.embeddings
                if hasattr(response, "embeddings")
                else response.get("embeddings")
            )
            if not isinstance(embeddings, list):
                raise EmbeddingError("Ollama SDK response missing 'embeddings'")
            return embeddings
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Ollama SDK embed failed: {exc}") from exc

    def _embed_urllib(self, texts: list[str]) -> list[list[float]]:
        import json
        import urllib.error
        import urllib.request

        payload = json.dumps({
            "model": self.settings.embedding_model,
            "input": texts,
            "keep_alive": "10m",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.settings.ollama_base_url.rstrip('/')}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = max(120, 3 * len(texts))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise EmbeddingError(f"Ollama HTTP embed failed: {exc}") from exc
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingError("Ollama HTTP response missing 'embeddings'")
        return embeddings

    # ------------------------------------------------------------------
    # Hash fallback (offline / no Ollama)
    # ------------------------------------------------------------------

    def _hash_embedding(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float64)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[bucket] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = float(np.linalg.norm(vector)) or 1.0
        return (vector / norm).tolist()

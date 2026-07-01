from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request

from backend.app.core.config import Settings
from backend.app.core.text import batched


class EmbeddingError(RuntimeError):
    pass


class EmbeddingService:
    def __init__(self, settings: Settings, dimensions: int = 384) -> None:
        self.settings = settings
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.settings.use_ollama:
            try:
                return self._embed_ollama_batched(texts)
            except EmbeddingError:
                if not self.settings.allow_hash_embeddings:
                    raise
        return [self._hash_embedding(text) for text in texts]

    def _embed_ollama_batched(self, texts: list[str]) -> list[list[float]]:
        # A single giant request for hundreds of chunks risks hitting the
        # request timeout and wastes the whole call if one bad chunk fails.
        # Batching keeps each request fast and lets us retry a failed batch
        # without redoing everything. Ollama serves one model instance, so
        # batches are sent sequentially rather than in parallel threads
        # (parallel HTTP calls would just queue up behind the same GPU
        # worker and add overhead without real throughput gain).
        batch_size = max(1, self.settings.embedding_batch_size)
        results: list[list[float]] = []
        for batch in batched(texts, batch_size):
            results.extend(self._embed_ollama(batch))
        return results

    def _embed_ollama(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.settings.embedding_model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.settings.ollama_base_url.rstrip('/')}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Scale the timeout with batch size so large batches on slower
        # hardware don't get killed mid-request.
        timeout = max(60, 5 * len(texts))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise EmbeddingError(f"Ollama embedding failed: {exc}") from exc

        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingError("Ollama embedding response did not include embeddings.")
        return embeddings

    def _hash_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

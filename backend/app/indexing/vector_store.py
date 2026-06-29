from __future__ import annotations

import json
import math
from pathlib import Path


class LocalVectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._vectors: dict[str, list[float]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._vectors = {}
            return
        self._vectors = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.path.write_text(json.dumps(self._vectors), encoding="utf-8")

    def upsert_many(self, vectors: dict[str, list[float]]) -> None:
        self._vectors.update(vectors)
        self.save()

    def delete_missing(self, keep_ids: set[str]) -> None:
        self._vectors = {key: value for key, value in self._vectors.items() if key in keep_ids}
        self.save()

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        scored = [
            (chunk_id, cosine_similarity(query_vector, vector))
            for chunk_id, vector in self._vectors.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)

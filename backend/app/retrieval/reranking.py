from __future__ import annotations

from backend.app.core.text import tokenize
from backend.app.models import Chunk


class Reranker:
    def rerank(self, question: str, candidates: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
        question_terms = set(tokenize(question))
        scored: list[tuple[Chunk, float]] = []
        for chunk in candidates:
            chunk_terms = set(tokenize(chunk.text))
            overlap = len(question_terms & chunk_terms)
            coverage = overlap / max(len(question_terms), 1)
            density = overlap / max(len(chunk_terms), 1)
            scored.append((chunk, coverage + density))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

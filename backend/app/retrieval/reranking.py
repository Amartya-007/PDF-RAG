"""Reranker — Context-agnostic lexical reranking for document chunks.

Calculates relevance by balancing query term coverage against token density 
to ensure concise, high-signal chunks bubble to the top.
"""
from __future__ import annotations

from backend.app.core.text import tokenize
from backend.app.models import Chunk


class Reranker:
    """Reranks retrieved ``Chunk`` candidates using strict set-based overlap metrics."""

    def rerank(
        self, question: str, candidates: list[Chunk], top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """Scores and sorts chunks based on question token coverage and chunk term density.

        Args:
            question: The raw user query.
            candidates: List of retrieved text chunks.
            top_k: Maximum number of sorted results to return.

        Returns:
            A list of tuples containing the Chunk and its computed similarity score,
            sorted in descending order.
        """
        if not candidates or not question.strip():
            return []

        # Standardize query terms once
        question_terms = set(tokenize(question))
        if not question_terms:
            return [(chunk, 0.0) for chunk in candidates[:top_k]]

        scored_candidates: list[tuple[Chunk, float]] = []
        q_len = len(question_terms)

        for chunk in candidates:
            # Safe text extraction handling potential missing fields
            chunk_text = chunk.text or ""
            chunk_terms = set(tokenize(chunk_text))
            
            if not chunk_terms:
                scored_candidates.append((chunk, 0.0))
                continue

            # Calculate core intersections
            overlap = len(question_terms & chunk_terms)
            
            # Coverage: What proportion of the question terms are answered here?
            coverage = overlap / q_len
            
            # Density: How much of this chunk is actually relevant noise-free data?
            density = overlap / len(chunk_terms)

            # Combined score (Max possible value: 2.0)
            final_score = round(coverage + density, 4)
            scored_candidates.append((chunk, final_score))

        # Sort by score descending
        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        
        return scored_candidates[:top_k]
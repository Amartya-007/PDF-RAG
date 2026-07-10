"""Core ranking algorithms.

Provides Reciprocal Rank Fusion (RRF) for combining multiple ranked lists
(e.g., lexical search hits, semantic search hits) into a single ordered list.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable, TypeVar

# Allows the function to accept strings, UUIDs, ints, or any hashable ID type
T = TypeVar("T", bound=Hashable)

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[T, float]]],
    top_k: int,
    rank_constant: int = 60,
) -> list[tuple[T, float]]:
    """Combines multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF is highly robust against outliers and effectively merges results 
    from different retrieval algorithms (e.g., BM25 and Vector Search) 
    without needing careful score normalization.

    Args:
        ranked_lists: A list where each element is a sorted list of (item_id, score).
                      Note: The original scores are ignored; only the positional rank matters.
        top_k: The maximum number of fused results to return.
        rank_constant: The k constant in the RRF formula. The default of 60 
                       is widely considered optimal in IR literature.

    Returns:
        A list of (item_id, fused_score) tuples, sorted by score in descending order.
    """
    if not ranked_lists:
        return []

    # defaultdict is implemented in C and is faster than dict.get(..., 0.0)
    scores: defaultdict[T, float] = defaultdict(float)

    for ranked in ranked_lists:
        for rank, (item_id, _original_score) in enumerate(ranked, start=1):
            scores[item_id] += 1.0 / (rank_constant + rank)

    # Sort by fused score descending
    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return fused[:top_k]
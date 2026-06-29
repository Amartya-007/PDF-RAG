from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    top_k: int,
    rank_constant: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (item_id, _score) in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rank_constant + rank)
    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return fused[:top_k]

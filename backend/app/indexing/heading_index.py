"""HeadingIndex — in-memory index from normalised heading text to node_id.

Enables direct heading-based lookup without vector similarity.
Ranking uses: exact match → prefix match → token overlap (all pure string ops).

Rebuilt from NodeRepository on startup and after full index rebuild.
"""
from __future__ import annotations

import re
from collections import defaultdict


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation at boundaries."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    return text


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"\w+", text.lower()) if len(w) > 1)


class HeadingIndex:
    """Maps normalised heading text to ``node_id`` values.

    Similarity ranking (for ``search``)
    ------------------------------------
    1. Exact normalised match  → score 1.0
    2. Prefix match            → score 0.8
    3. Token overlap (Jaccard) → score 0.0–0.6
    Nodes below a minimum overlap threshold are excluded.
    """

    _MIN_OVERLAP = 0.15

    def __init__(self) -> None:
        # normalised_heading → list[node_id]
        self._heading_to_nodes: dict[str, list[str]] = defaultdict(list)
        # node_id → normalised_heading (for removal)
        self._node_to_heading: dict[str, str] = {}

    # ── Write operations ───────────────────────────────────────────────────

    def index(self, node_id: str, title: str) -> None:
        """Register *node_id* under the normalised form of *title*."""
        if not title:
            return
        key = _normalise(title)
        if not key:
            return
        # Deduplicate within the list
        bucket = self._heading_to_nodes[key]
        if node_id not in bucket:
            bucket.append(node_id)
        self._node_to_heading[node_id] = key

    def remove(self, node_id: str) -> None:
        """Remove *node_id* from the index."""
        key = self._node_to_heading.pop(node_id, None)
        if key and key in self._heading_to_nodes:
            bucket = self._heading_to_nodes[key]
            if node_id in bucket:
                bucket.remove(node_id)
            if not bucket:
                del self._heading_to_nodes[key]

    def rebuild(self, heading_items: list[tuple[str, str]]) -> None:
        """Rebuild the entire index from *(node_id, title)* pairs."""
        self._heading_to_nodes.clear()
        self._node_to_heading.clear()
        for node_id, title in heading_items:
            self.index(node_id, title)

    # ── Read operations ────────────────────────────────────────────────────

    def search(self, heading_text: str) -> list[str]:
        """Return node_ids whose titles best match *heading_text*, ranked.

        Returns an empty list when no heading meets the minimum overlap threshold.
        """
        if not heading_text:
            return []
        query = _normalise(heading_text)
        if not query:
            return []

        scored: list[tuple[str, float]] = []
        query_tokens = _tokens(query)

        for key, node_ids in self._heading_to_nodes.items():
            score = self._score(query, query_tokens, key)
            if score >= self._MIN_OVERLAP:
                for node_id in node_ids:
                    scored.append((node_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for node_id, _ in scored:
            if node_id not in seen:
                seen.add(node_id)
                result.append(node_id)
        return result

    def __len__(self) -> int:
        return len(self._node_to_heading)

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _score(query: str, query_tokens: frozenset[str], key: str) -> float:
        if query == key:
            return 1.0
        if key.startswith(query) or query.startswith(key):
            return 0.8
        key_tokens = _tokens(key)
        if not query_tokens or not key_tokens:
            return 0.0
        intersection = len(query_tokens & key_tokens)
        union = len(query_tokens | key_tokens)
        return (intersection / union) * 0.6 if union else 0.0

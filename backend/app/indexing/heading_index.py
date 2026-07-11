"""HeadingIndex — in-memory index from normalised heading text to node_id.

Enables direct heading-based lookup without vector similarity.
Ranking uses: exact match → prefix match → token overlap.

Rebuilt from NodeRepository on startup and after full index rebuild.
"""
from __future__ import annotations

import re
from collections import defaultdict

# Pre-compiled regex for performance
_TOKEN_RE = re.compile(r"\w+")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"^[\W_]+|[\W_]+$")


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, and strip punctuation at boundaries."""
    text = text.lower().strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return _NON_WORD_RE.sub("", text)


def _tokens(text: str) -> frozenset[str]:
    """Tokenize text into a set of unique words longer than 1 character."""
    return frozenset(w for w in _TOKEN_RE.findall(text.lower()) if len(w) > 1)


class HeadingIndex:
    """Maps normalised heading text to ``node_id`` values.

    Similarity ranking logic:
    1. Exact normalised match    → score 1.0
    2. Prefix match              → score 0.8
    3. Token overlap (Jaccard)   → score 0.0–0.6
    
    Nodes below a minimum overlap threshold are excluded.

    Attributes:
        _node_to_heading: Maps node_id to the last seen heading.
        _heading_to_nodes: Maps normalised heading to a list of node_ids.
    """

    _MIN_OVERLAP = 0.15

    def __init__(self) -> None:
        self._node_to_heading: dict[str, str] = {}
        self._heading_to_nodes: dict[str, list[str]] = defaultdict(list)

    def index(self, node_id: str, heading: str) -> None:
        """Add or replace a heading mapping for a node.

        Attributes:
            node_id: Identifier of the node being indexed.
            heading: The node's heading/title text.
        """
        self.remove(node_id)
        normalised = _normalise(heading)
        self._node_to_heading[node_id] = heading
        self._heading_to_nodes[normalised].append(node_id)

    def remove(self, node_id: str) -> None:
        """Remove a node from the index, if present.

        Attributes:
            node_id: Identifier of the node to remove.
        """
        heading = self._node_to_heading.pop(node_id, None)
        if heading is None:
            return
        normalised = _normalise(heading)
        bucket = self._heading_to_nodes.get(normalised)
        if bucket:
            self._heading_to_nodes[normalised] = [nid for nid in bucket if nid != node_id]
            if not self._heading_to_nodes[normalised]:
                del self._heading_to_nodes[normalised]

    def rebuild(self, items: list[tuple[str, str]]) -> None:
        """Wipe and rebuild the index from (node_id, heading) pairs."""
        self._node_to_heading.clear()
        self._heading_to_nodes.clear()
        for node_id, heading in items:
            self.index(node_id, heading)

    def search(self, heading_text: str) -> list[str]:
        """Search for nodes matching a heading, ranked by similarity.

        Attributes:
            heading_text: The user-provided heading to search for.
        """
        if not (query := _normalise(heading_text)):
            return []

        scored: list[tuple[str, float]] = []
        query_tokens = _tokens(query)

        for key, node_ids in self._heading_to_nodes.items():
            score = self._score(query, query_tokens, key)
            if score >= self._MIN_OVERLAP:
                for node_id in node_ids:
                    scored.append((node_id, score))

        # Rank by score descending
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
        """Return the number of tracked headings."""
        return len(self._node_to_heading)

    @staticmethod
    def _score(query: str, query_tokens: frozenset[str], key: str) -> float:
        """Calculate similarity score between query and index key."""
        if query == key:
            return 1.0
        if key.startswith(query) or query.startswith(key):
            return 0.8
        
        key_tokens = _tokens(key)
        if not key_tokens or not query_tokens:
            return 0.0
            
        # Jaccard index
        intersection = query_tokens.intersection(key_tokens)
        union = query_tokens.union(key_tokens)
        return len(intersection) / len(union)
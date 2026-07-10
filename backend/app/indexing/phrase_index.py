"""PhraseIndex — exact multi-word phrase lookup.

Indexes phrases extracted from node titles and first sentences.
Enables fast exact-phrase retrieval for quoted queries (e.g., "annual leave").

All operations are pure in-memory string matching — no embedding, no vectors.
"""
from __future__ import annotations

import re
from collections import defaultdict

# Pre-compiled regex for performance
_WORD_RE = re.compile(r"\b\w+\b")
_SENTENCE_RE = re.compile(r"[.!?]")


def _extract_phrases(text: str, max_n: int = 4) -> list[str]:
    """Extract 2-to-max_n word n-grams from text, lower-cased."""
    words = _WORD_RE.findall(text.lower())
    phrases: list[str] = []
    for n in range(2, min(max_n + 1, len(words) + 1)):
        for i in range(len(words) - n + 1):
            phrases.append(" ".join(words[i : i + n]))
    return phrases


def _first_sentence(text: str) -> str:
    """Return the first sentence of text (up to 200 characters)."""
    match = _SENTENCE_RE.search(text[:400])
    return text[: match.start() + 1].strip() if match else text[:200].strip()


class PhraseIndex:
    """Maps multi-word phrases to the nodes that contain them.

    Scoring logic:
    Exact phrase match → 1.0
    Partial overlap    → ~0.5 (weighted)

    Attributes:
        _phrase_map: Maps a phrase string to a list of node_ids.
        _node_phrases: Maps a node_id to a set of its indexed phrases.
    """

    def __init__(self) -> None:
        self._phrase_map: dict[str, list[str]] = defaultdict(list)
        self._node_phrases: dict[str, set[str]] = defaultdict(set)

    def index(self, node_id: str, title: str | None, text: str) -> None:
        """Extract and index phrases from a node.

        Attributes:
            node_id: Identifier for the node.
            title:   Node title (if any).
            text:    Node content.
        """
        self.remove(node_id)
        
        content = f"{title or ''} {_first_sentence(text)}"
        phrases = _extract_phrases(content)
        
        for phrase in phrases:
            self._phrase_map[phrase].append(node_id)
            self._node_phrases[node_id].add(phrase)

    def remove(self, node_id: str) -> None:
        """Remove a node and its indexed phrases from the index.

        Attributes:
            node_id: Identifier for the node.
        """
        phrases = self._node_phrases.pop(node_id, set())
        for phrase in phrases:
            bucket = self._phrase_map.get(phrase)
            if bucket:
                # Remove all occurrences of this node_id
                self._phrase_map[phrase] = [nid for nid in bucket if nid != node_id]
                if not self._phrase_map[phrase]:
                    del self._phrase_map[phrase]

    def rebuild(self, items: list[tuple[str, str | None, str]]) -> None:
        """Wipe and rebuild from list of (node_id, title, text) tuples."""
        self._phrase_map.clear()
        self._node_phrases.clear()
        for node_id, title, text in items:
            self.index(node_id, title, text)

    def search(self, phrase: str) -> list[tuple[str, float]]:
        """Return (node_id, score) pairs for nodes matching the phrase.

        Tries exact match first; falls back to partial word overlap.
        """
        if not phrase:
            return []
            
        key = " ".join(_WORD_RE.findall(phrase.lower()))
        if not key:
            return []

        # 1. Exact match (Score 1.0)
        results: dict[str, float] = {nid: 1.0 for nid in self._phrase_map.get(key, [])}

        # 2. Partial overlap: drop outer words to allow fuzzy matching
        words = key.split()
        if len(words) >= 3:
            for partial in [" ".join(words[1:]), " ".join(words[:-1])]:
                if partial in self._phrase_map:
                    for nid in self._phrase_map[partial]:
                        if nid not in results:
                            results[nid] = 0.5
                            
        return list(results.items())
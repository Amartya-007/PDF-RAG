"""PhraseIndex — exact multi-word phrase lookup.

Indexes phrases extracted from node titles and first sentences.
Enables fast exact-phrase retrieval for quoted queries (e.g. "annual leave").

All operations are pure in-memory string matching — no embedding, no vectors.
"""
from __future__ import annotations

import re
from collections import defaultdict


def _extract_phrases(text: str, max_n: int = 4) -> list[str]:
    """Extract 2-to-max_n word n-grams from *text*, lower-cased."""
    words = re.findall(r"\b\w+\b", text.lower())
    phrases: list[str] = []
    for n in range(2, min(max_n + 1, len(words) + 1)):
        for i in range(len(words) - n + 1):
            phrases.append(" ".join(words[i : i + n]))
    return phrases


def _first_sentence(text: str) -> str:
    """Return the first sentence of *text* (up to 200 characters)."""
    match = re.search(r"[.!?]", text[:400])
    return text[: match.start() + 1].strip() if match else text[:200].strip()


class PhraseIndex:
    """Maps multi-word phrases to the nodes that contain them.

    Scoring
    -------
    Exact phrase match → 1.0
    Partial phrase overlap (one query word missing) → 0.6
    """

    def __init__(self) -> None:
        # phrase → list[(node_id, score)]
        self._phrase_map: dict[str, list[str]] = defaultdict(list)
        # node_id → set of indexed phrases (for removal)
        self._node_phrases: dict[str, set[str]] = defaultdict(set)

    # ── Write operations ───────────────────────────────────────────────────

    def index(self, node_id: str, title: str | None, text: str) -> None:
        """Index phrases from *title* and the first sentence of *text*."""
        sources: list[str] = []
        if title:
            sources.append(title)
        if text:
            sources.append(_first_sentence(text))

        for source in sources:
            for phrase in _extract_phrases(source):
                bucket = self._phrase_map[phrase]
                if node_id not in bucket:
                    bucket.append(node_id)
                self._node_phrases[node_id].add(phrase)

    def remove(self, node_id: str) -> None:
        """Remove all phrase entries for *node_id*."""
        for phrase in self._node_phrases.pop(node_id, set()):
            bucket = self._phrase_map.get(phrase, [])
            if node_id in bucket:
                bucket.remove(node_id)
            if not bucket and phrase in self._phrase_map:
                del self._phrase_map[phrase]

    def rebuild(self, items: list[tuple[str, str | None, str]]) -> None:
        """Rebuild from *(node_id, title, text)* triples."""
        self._phrase_map.clear()
        self._node_phrases.clear()
        for node_id, title, text in items:
            self.index(node_id, title, text)

    # ── Read operations ────────────────────────────────────────────────────

    def search(self, phrase: str) -> list[tuple[str, float]]:
        """Return ``(node_id, score)`` pairs for nodes matching *phrase*.

        Tries exact match first; falls back to partial word overlap.
        """
        if not phrase:
            return []
        key = " ".join(re.findall(r"\b\w+\b", phrase.lower()))
        if not key:
            return []

        # Exact match
        exact_ids = self._phrase_map.get(key, [])
        results: dict[str, float] = {nid: 1.0 for nid in exact_ids}

        # Partial overlap: any sub-phrase (drop one word from either end)
        words = key.split()
        if len(words) >= 3:
            for partial in [" ".join(words[1:]), " ".join(words[:-1])]:
                for nid in self._phrase_map.get(partial, []):
                    if nid not in results:
                        results[nid] = 0.6

        return sorted(results.items(), key=lambda x: x[1], reverse=True)

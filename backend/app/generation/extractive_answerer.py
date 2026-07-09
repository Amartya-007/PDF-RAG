"""ExtractiveAnswerer — instant, no-LLM answer extraction.

Selects the best sentence or passage directly from retrieved DocumentNodes
using pure text-overlap scoring.  Used in two situations:

  1. Fast path: short factual questions (names, dates, numbers, definitions)
     where an extractive sentence is good enough.
  2. Fallback: when Ollama is unavailable or times out.

Scoring
-------
For each sentence in the retrieved nodes the scorer computes:
  score = query_token_recall            (fraction of query tokens found)
        + position_bonus * 0.20         (first sentence of a section)
        + heading_match  * 0.15         (query appears in node title)

The top-scoring sentence is returned as the answer text; the node it came
from becomes the primary citation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.domain.models.node import DocumentNode
from backend.app.models import Answer, Citation


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\b\w{2,}\b")

_FAST_FACT_PATTERNS = [
    re.compile(r"\bwhat\s+is\b", re.I),
    re.compile(r"\bwho\s+is\b", re.I),
    re.compile(r"\bwhen\s+(is|was|did)\b", re.I),
    re.compile(r"\bhow\s+many\b", re.I),
    re.compile(r"\bdefine\b", re.I),
]

_INSUFFICIENT = (
    "I could not find sufficient evidence in the documents to answer this question."
)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w.lower() for w in _WORD_RE.findall(text))


def _recall(query_tokens: frozenset[str], sent_tokens: frozenset[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & sent_tokens) / len(query_tokens)


@dataclass(slots=True)
class _SentenceCandidate:
    text: str
    score: float
    node: DocumentNode
    sentence_index: int


class ExtractiveAnswerer:
    """Extracts answers from ``DocumentNode`` objects without model inference.

    Args:
        min_sentence_words: Discard sentences shorter than this threshold
                            to avoid extracting fragment noise.
    """

    def __init__(self, min_sentence_words: int = 5) -> None:
        self._min_words = min_sentence_words

    # ── Public API ─────────────────────────────────────────────────────────

    def is_fast_fact_question(self, question: str) -> bool:
        """Return True for short factual questions that don't need an LLM."""
        if len(question.split()) > 15:
            return False
        return any(p.search(question) for p in _FAST_FACT_PATTERNS)

    def answer(
        self, question: str, nodes: list[DocumentNode]
    ) -> Answer:
        """Extract the best sentence from *nodes* for *question*.

        Always returns a valid ``Answer``; falls back to
        ``INSUFFICIENT`` when no good sentence is found.
        """
        if not nodes:
            return self._no_evidence(question)

        query_tokens = _tokens(question)
        best = self._find_best_sentence(query_tokens, nodes)

        if best is None or best.score < 0.10:
            return self._no_evidence(question)

        citation = self._node_to_citation(best.node, f"[S{1}]")
        return Answer(
            question=question,
            answer=best.text,
            citations=[citation],
            answerable=True,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _find_best_sentence(
        self, query_tokens: frozenset[str], nodes: list[DocumentNode]
    ) -> "_SentenceCandidate | None":
        best: _SentenceCandidate | None = None

        for node in nodes:
            title_tokens = _tokens(node.title or "")
            heading_match = _recall(query_tokens, title_tokens)
            sentences = _SENTENCE_SPLIT.split(node.text.strip())

            for idx, sent in enumerate(sentences):
                sent = sent.strip()
                if len(sent.split()) < self._min_words:
                    continue
                sent_tokens = _tokens(sent)
                score = (
                    _recall(query_tokens, sent_tokens)
                    + (0.20 if idx == 0 else 0.0)
                    + 0.15 * heading_match
                )
                if best is None or score > best.score:
                    best = _SentenceCandidate(
                        text=sent,
                        score=score,
                        node=node,
                        sentence_index=idx,
                    )
        return best

    @staticmethod
    def _node_to_citation(node: DocumentNode, source_id: str) -> Citation:
        pages = str(node.page_start)
        if node.page_end and node.page_end != node.page_start:
            pages = f"{node.page_start}-{node.page_end}"
        return Citation(
            source_id=source_id,
            chunk_id=node.id,
            filename=node.document_id,
            page_start=node.page_start,
            page_end=node.page_end or node.page_start,
            excerpt=node.text[:300],
        )

    @staticmethod
    def _no_evidence(question: str) -> Answer:
        return Answer(
            question=question,
            answer=_INSUFFICIENT,
            citations=[],
            answerable=False,
        )

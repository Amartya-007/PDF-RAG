"""NodeRanker — reranks DocumentNode candidates without embeddings.

Uses a cascade of pure text signals so no GPU / model call is required.
Designed to run in < 5 ms for ≤ 200 candidates on any hardware.

Scoring signals (weighted sum, normalised to 0–1 range)
---------------------------------------------------------
Signal                    Weight   Notes
──────────────────────────────────────────────────────────
Token overlap (Jaccard)    0.50    Query tokens ∩ node tokens
Heading-text match         0.20    Normalised substring of title in query
Source signal (tree)       0.15    Tree-selected nodes get a bonus
Page position boost        0.10    Earlier pages rank slightly higher
Text density               0.05    Prefer nodes with > 30 words
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from backend.app.domain.models.node import DocumentNode


def _token_set(text: str) -> frozenset[str]:
    return frozenset(w.lower() for w in re.findall(r"\b\w{2,}\b", text))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _recall(query_tokens: frozenset[str], text_tokens: frozenset[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


@dataclass(slots=True)
class RankingResult:
    node: DocumentNode
    score: float
    signal_overlap: float = 0.0
    signal_heading: float = 0.0
    signal_source: float = 0.0
    signal_page: float = 0.0
    signal_density: float = 0.0


class NodeRanker:
    """Scores and sorts ``DocumentNode`` candidates for a given query.

    Args:
        max_page: Used to normalise the page-position signal.
                  Set to the total page count of the current document set.
    """

    _W_OVERLAP  = 0.50
    _W_HEADING  = 0.20
    _W_SOURCE   = 0.15
    _W_PAGE     = 0.10
    _W_DENSITY  = 0.05

    def __init__(self, max_page: int = 500) -> None:
        self._max_page = max(max_page, 1)
        self._last_details: dict[str, dict[str, float]] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def rank(
        self,
        query: str,
        candidates: list[DocumentNode],
        tree_selected_ids: set[str] | None = None,
        top_k: int | None = None,
    ) -> list[RankingResult]:
        """Score and sort *candidates* for *query*.

        Args:
            query:             User question string.
            candidates:        Candidate nodes from LexicalRetriever / TreeNavigator.
            tree_selected_ids: Node IDs chosen by TreeNavigator (source bonus).
            top_k:             Return at most this many results.

        Returns:
            List of ``RankingResult`` sorted by descending score.
        """
        if not candidates:
            return []

        query_tokens = _token_set(query)
        tree_ids = tree_selected_ids or set()
        results = [
            self._score_node(node, query, query_tokens, tree_ids)
            for node in candidates
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        self._remember_results(results)
        return results[:top_k] if top_k is not None else results

    def rerank(
        self,
        query: str,
        candidates: list[DocumentNode],
        tree_selected_ids: set[str] | None = None,
        top_k: int = 8,
    ) -> list[DocumentNode]:
        """Convenience wrapper — returns plain ``DocumentNode`` list."""
        ranked = self.rank(query, candidates, tree_selected_ids, top_k)
        return [r.node for r in ranked]

    def rank_fast_fact(
        self,
        question: str,
        nodes: list[DocumentNode],
        candidate_ids: set[str],
    ) -> list[DocumentNode]:
        """Rank structured-field lookup candidates without embeddings."""
        results = [
            self._score_specialized_node(
                node,
                question,
                candidate_ids,
                mode="fast_fact",
            )
            for node in nodes
        ]
        results.sort(key=lambda item: item[0], reverse=True)
        return [node for _score, node in results]

    def rank_topic(
        self,
        question: str,
        nodes: list[DocumentNode],
        candidate_ids: set[str],
    ) -> list[DocumentNode]:
        """Rank topic/explanation candidates without embeddings."""
        results = [
            self._score_specialized_node(
                node,
                question,
                candidate_ids,
                mode="topic",
            )
            for node in nodes
        ]
        results.sort(key=lambda item: item[0], reverse=True)
        return [node for _score, node in results]

    def score_details(self, node_id: str) -> dict[str, float]:
        return self._last_details.get(node_id, {})

    # ── Private helpers ────────────────────────────────────────────────────

    def _score_node(
        self,
        node: DocumentNode,
        query: str,
        query_tokens: frozenset[str],
        tree_ids: set[str],
    ) -> RankingResult:
        node_tokens = _token_set(node.text)
        overlap  = _jaccard(query_tokens, node_tokens)
        heading  = self._heading_match(query, node.title)
        source   = 1.0 if node.id in tree_ids else 0.0
        page     = self._page_score(node.page_start)
        density  = self._density_score(node.text)

        score = (
            self._W_OVERLAP * overlap
            + self._W_HEADING * heading
            + self._W_SOURCE  * source
            + self._W_PAGE    * page
            + self._W_DENSITY * density
        )

        return RankingResult(
            node=node,
            score=round(score, 4),
            signal_overlap=overlap,
            signal_heading=heading,
            signal_source=source,
            signal_page=page,
            signal_density=density,
        )

    def _score_specialized_node(
        self,
        node: DocumentNode,
        question: str,
        candidate_ids: set[str],
        mode: str,
    ) -> tuple[float, DocumentNode]:
        question_tokens = _token_set(question)
        node_tokens = _token_set(" ".join([node.title or "", node.text]))
        overlap = _recall(question_tokens, node_tokens)
        heading = self._heading_match(question, node.title)
        retrieval_hit = 1.0 if node.id in candidate_ids else 0.0
        page = self._page_score(node.page_start)
        density = self._density_score(node.text)

        fast_fact_score = self._fast_fact_signal(question_tokens, node_tokens)
        topic_score = self._topic_signal(question, question_tokens, node)

        if mode == "fast_fact":
            score = (
                0.35 * overlap
                + 0.25 * fast_fact_score
                + 0.20 * retrieval_hit
                + 0.10 * heading
                + 0.05 * page
                + 0.05 * density
            )
        else:
            score = (
                0.35 * topic_score
                + 0.25 * heading
                + 0.20 * overlap
                + 0.10 * retrieval_hit
                + 0.05 * page
                + 0.05 * density
            )

        self._last_details[node.id] = {
            "score": round(score, 4),
            "overlap": overlap,
            "heading_score": heading,
            "retrieval_hit": retrieval_hit,
            "page_score": page,
            "density_score": density,
            "fast_fact_score": fast_fact_score,
            "topic_score": topic_score,
        }
        return (score, node)

    def _remember_results(self, results: list[RankingResult]) -> None:
        self._last_details = {
            result.node.id: {
                "score": result.score,
                "overlap": result.signal_overlap,
                "heading_score": result.signal_heading,
                "retrieval_hit": result.signal_source,
                "page_score": result.signal_page,
                "density_score": result.signal_density,
                "fast_fact_score": 0.0,
                "topic_score": result.signal_overlap + result.signal_heading,
            }
            for result in results
        }

    @staticmethod
    def _fast_fact_signal(
        question_tokens: frozenset[str],
        node_tokens: frozenset[str],
    ) -> float:
        fast_terms = {
            "name", "college", "collage", "institute", "university", "school",
            "degree", "course", "branch", "program", "cgpa", "gpa", "email",
            "phone", "mobile", "contact",
        }
        matched_query_terms = question_tokens & fast_terms
        if not matched_query_terms:
            return 0.0
        synonym_hits = 0.0
        if {"college", "collage"} & matched_query_terms and {
            "college", "institute", "university", "school",
        } & node_tokens:
            synonym_hits = 1.0
        if "name" in matched_query_terms and len(node_tokens) >= 2:
            synonym_hits = max(synonym_hits, 0.5)
        return min(1.0, _recall(matched_query_terms, node_tokens) + synonym_hits)

    def _topic_signal(
        self,
        question: str,
        question_tokens: frozenset[str],
        node: DocumentNode,
    ) -> float:
        node_tokens = _token_set(" ".join([node.title or "", node.text]))
        return max(
            _recall(question_tokens, node_tokens),
            self._heading_match(question, node.title),
        )

    @staticmethod
    def _heading_match(query: str, title: str | None) -> float:
        if not title:
            return 0.0
        q = query.lower()
        t = title.lower()
        if t in q or q in t:
            return 1.0
        q_tokens = _token_set(q)
        t_tokens = _token_set(t)
        return _jaccard(q_tokens, t_tokens)

    def _page_score(self, page: int) -> float:
        """Earlier pages score higher; decays logarithmically."""
        if page <= 0:
            return 0.5
        return max(0.0, 1.0 - math.log(page + 1) / math.log(self._max_page + 2))

    @staticmethod
    def _density_score(text: str) -> float:
        words = len(text.split())
        if words < 10:
            return 0.0
        if words < 30:
            return 0.5
        return 1.0

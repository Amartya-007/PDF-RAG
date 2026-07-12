"""NodeRanker — reranks DocumentNode candidates without embeddings.

Uses a cascade of pure text signals so no GPU / model call is required.
Designed to run in < 5 ms for ≤ 200 candidates on any hardware.

Scoring signals (weighted sum, normalised to 0–1 range)
---------------------------------------------------------
Signal                    Weight    Notes
──────────────────────────────────────────────────────────
Token overlap (Jaccard)    0.50    Query tokens ∩ node tokens
Heading-text match         0.20    Normalised substring or token overlap of title in query
Source signal (tree)       0.15    Tree-selected nodes get a bonus
Page position boost        0.10    Earlier pages rank slightly higher
Text density               0.05    Prefer robust nodes over empty fragments
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from backend.app.core.nlp import nlp
from backend.app.domain.models.node import DocumentNode


def _token_set(text: str) -> frozenset[str]:
    return frozenset(w.lower() for w in re.findall(r"\b\w{2,}\b", text))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# A "what's the name" style query can miss on pure token overlap entirely --
# a resume header containing "Amartya Vishwakarma" shares zero literal words
# with a query asking for "name". Gates the (comparatively expensive) NER
# check in _score_node so it only runs for queries that actually need it.
_NAME_QUERY_RE = re.compile(r"\bname\b", re.I)


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
    # Extra (not part of the base 1.0 weight budget above): a name-seeking
    # query gets a strong bonus for candidates that actually contain a
    # detectable person name, since such queries often share zero literal
    # words with a resume/profile header ("name" vs. "Amartya Vishwakarma").
    _W_ENTITY   = 0.30

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
        wants_person_name = bool(_NAME_QUERY_RE.search(query))

        results = [
            self._score_node(node, query, query_tokens, tree_ids, wants_person_name)
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
        query: str,
        candidates: list[DocumentNode],
        candidate_ids: set[str] | None = None,
    ) -> list[DocumentNode]:
        """Rank candidates for a fast-fact query (short, specific-entity answer).

        Weighs retrieval-hit membership and token overlap most heavily, since
        fast-fact answers usually live in the node that was actually
        retrieved rather than the one with the most on-topic heading.

        Attributes:
            query:         The user's question.
            candidates:    Candidate nodes to rank.
            candidate_ids: Node IDs that came back from the retrieval step
                           (e.g. lexical search hits); ranked higher.
        """
        return self._rank_by_query_type(query, candidates, candidate_ids, primary="fast_fact")

    def rank_topic(
        self,
        query: str,
        candidates: list[DocumentNode],
        candidate_ids: set[str] | None = None,
    ) -> list[DocumentNode]:
        """Rank candidates for a topic/overview query (broad, heading-led answer).

        Weighs heading-text match most heavily, since topic questions ("tell
        me about X") are best answered starting from the section whose
        title matches X, even if the retrieval hit set is noisy.

        Attributes:
            query:         The user's question.
            candidates:    Candidate nodes to rank.
            candidate_ids: Node IDs that came back from the retrieval step;
                           ranked higher.
        """
        return self._rank_by_query_type(query, candidates, candidate_ids, primary="topic")

    def _rank_by_query_type(
        self,
        query: str,
        candidates: list[DocumentNode],
        candidate_ids: set[str] | None,
        primary: str,
    ) -> list[DocumentNode]:
        if not candidates:
            return []

        candidate_ids = candidate_ids or set()
        query_tokens = _token_set(query)

        scored: list[tuple[DocumentNode, float, dict[str, float]]] = []
        for node in candidates:
            node_text = node.text or ""
            overlap = _jaccard(query_tokens, _token_set(node_text))
            heading = self._heading_match(query, node.title)
            retrieval_hit = 1.0 if node.id in candidate_ids else 0.0
            page = self._page_score(node.page_start)
            density = self._density_score(node_text)

            fast_fact_score = round(
                0.55 * retrieval_hit + 0.30 * overlap + 0.10 * heading + 0.05 * density, 4
            )
            topic_score = round(
                0.45 * heading + 0.30 * overlap + 0.15 * retrieval_hit + 0.10 * page, 4
            )
            primary_score = fast_fact_score if primary == "fast_fact" else topic_score

            scored.append((node, primary_score, {
                "score": primary_score,
                "fast_fact_score": fast_fact_score,
                "topic_score": topic_score,
                "overlap": overlap,
                "heading_score": heading,
                "retrieval_hit": retrieval_hit,
                "page_score": page,
                "density_score": density,
            }))

        scored.sort(key=lambda entry: entry[1], reverse=True)
        self._last_details = {node.id: details for node, _, details in scored}
        return [node for node, _, _ in scored]

    def score_details(self, node_id: str) -> dict[str, float]:
        """Returns scoring breakdowns for debugging retrieval logic."""
        return self._last_details.get(node_id, {})

    # ── Private helpers ────────────────────────────────────────────────────

    def _score_node(
        self,
        node: DocumentNode,
        query: str,
        query_tokens: frozenset[str],
        tree_ids: set[str],
        wants_person_name: bool = False,
    ) -> RankingResult:
        node_text = node.text or ""
        node_tokens = _token_set(node_text)
        
        overlap  = _jaccard(query_tokens, node_tokens)
        heading  = self._heading_match(query, node.title)
        source   = 1.0 if node.id in tree_ids else 0.0
        page     = self._page_score(node.page_start)
        density  = self._density_score(node_text)
        entity   = self._person_entity_score(node_text) if wants_person_name else 0.0

        score = (
            self._W_OVERLAP * overlap
            + self._W_HEADING * heading
            + self._W_SOURCE  * source
            + self._W_PAGE    * page
            + self._W_DENSITY * density
            + self._W_ENTITY  * entity
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

    def _remember_results(self, results: list[RankingResult]) -> None:
        self._last_details = {
            result.node.id: {
                "score": result.score,
                "overlap": result.signal_overlap,
                "heading_score": result.signal_heading,
                "retrieval_hit": result.signal_source,
                "page_score": result.signal_page,
                "density_score": result.signal_density,
            }
            for result in results
        }

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

    @staticmethod
    def _person_entity_score(text: str) -> float:
        """Cheap PERSON-entity presence check via the shared spaCy pipeline.

        Only invoked for name-seeking queries (gated by the caller) so NER
        doesn't run on every candidate for every query -- just the ones
        where token overlap alone is known to be an unreliable signal.

        Requires genuine title-case tokens (e.g. "Amartya Vishwakarma").
        spaCy's small model also mis-tags ALL-CAPS section headers (e.g.
        "EDUCATION COURSEWORK SKILLS...") as PERSON; without this check
        such false positives would tie the score against a real name.
        """
        if not text or len(text) > 500:
            # A name usually lives in a short header/label; skip long
            # paragraphs both for cost and because NER precision drops on
            # dense body text anyway.
            return 0.0
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            tokens = [t for t in ent.text.split() if t.isalpha()]
            if len(tokens) >= 2 and all(t[:1].isupper() and t[1:].islower() for t in tokens):
                return 1.0
        return 0.0
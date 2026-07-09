"""LexicalRetriever — vectorless retrieval over DocumentNodes.

Combines three complementary search signals with Reciprocal Rank Fusion:
  1. FTS5 full-text search        (BM25 via SQLite built-in)
  2. HeadingIndex exact/prefix/overlap matching
  3. PhraseIndex exact/partial phrase matching

Returns (node_id, fused_score) pairs — zero embedding calls, zero vector ops.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from backend.app.indexing.full_text_index import FTS5Index
from backend.app.indexing.heading_index import HeadingIndex
from backend.app.indexing.phrase_index import PhraseIndex
from backend.app.retrieval.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

_QUOTE_RE = re.compile(r'"([^"]+)"')


@dataclass
class ScoreBreakdown:
    """Per-node score breakdown for debug output."""
    node_id: str
    fts5_score: float = 0.0
    heading_score: float = 0.0
    phrase_score: float = 0.0
    fused_score: float = 0.0


class LexicalRetriever:
    """Searches ``DocumentNode`` records using three vectorless signal sources.

    Args:
        fts5:    The FTS5 full-text index.
        heading: The heading text → node_id index.
        phrase:  The multi-word phrase index.
        node_repo: Used to verify the nodes table is non-empty.
    """

    def __init__(
        self,
        fts5: FTS5Index,
        heading: HeadingIndex,
        phrase: PhraseIndex,
    ) -> None:
        self._fts5 = fts5
        self._heading = heading
        self._phrase = phrase
        self._last_breakdowns: dict[str, ScoreBreakdown] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 20,
        session_node_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return fused ``(node_id, score)`` pairs for *query*.

        Args:
            query:            User question or sub-query.
            top_k:            Maximum number of results.
            session_node_ids: When provided, results are filtered to these IDs.
                              Avoids cross-session leakage.

        Returns:
            Empty list (with warning) when the node store is empty.
        """
        if not query.strip():
            return []

        self._last_breakdowns = {}

        # FTS5 signal
        fts5_results = self._fts5.search(query, top_k * 2)
        if not fts5_results:
            logger.warning(
                "LexicalRetriever: FTS5 returned no results — "
                "nodes table may be empty or index inconsistent"
            )

        # Heading signal
        heading_ids = self._heading.search(query)
        heading_results = [(nid, 1.0 - (i * 0.05)) for i, nid in enumerate(heading_ids[:top_k])]

        # Phrase signal (only for quoted or multi-word queries)
        phrase_query = self._extract_phrase_query(query)
        phrase_results: list[tuple[str, float]] = []
        if phrase_query:
            phrase_results = self._phrase.search(phrase_query)[:top_k]

        # Build breakdown map
        for node_id, score in fts5_results:
            self._last_breakdowns.setdefault(node_id, ScoreBreakdown(node_id)).fts5_score = score
        for node_id, score in heading_results:
            self._last_breakdowns.setdefault(node_id, ScoreBreakdown(node_id)).heading_score = score
        for node_id, score in phrase_results:
            self._last_breakdowns.setdefault(node_id, ScoreBreakdown(node_id)).phrase_score = score

        # Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            [fts5_results, heading_results, phrase_results],
            top_k=top_k,
        )

        # Filter to session scope
        if session_node_ids is not None:
            fused = [(nid, score) for nid, score in fused if nid in session_node_ids]

        for node_id, score in fused:
            bd = self._last_breakdowns.setdefault(node_id, ScoreBreakdown(node_id))
            bd.fused_score = score

        return fused[:top_k]

    def score_breakdown(self, node_id: str) -> dict[str, float]:
        """Return per-signal scores for *node_id* from the last ``search`` call."""
        bd = self._last_breakdowns.get(node_id)
        if bd is None:
            return {}
        return {
            "fts5_score": bd.fts5_score,
            "heading_score": bd.heading_score,
            "phrase_score": bd.phrase_score,
            "fused_score": bd.fused_score,
        }

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract_phrase_query(query: str) -> str:
        """Extract quoted phrase or return the full query for multi-word terms."""
        quoted = _QUOTE_RE.findall(query)
        if quoted:
            return quoted[0]
        words = query.split()
        if len(words) >= 2:
            return query
        return ""

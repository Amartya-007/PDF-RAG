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
from dataclasses import dataclass
from typing import Final

from thefuzz import fuzz

from backend.app.database.repositories.node_repository import NodeRepository
from backend.app.domain.exceptions import DocumentNotFoundError
from backend.app.domain.models.node import DocumentNode
from backend.app.indexing.full_text_index import FTS5Index
from backend.app.indexing.heading_index import HeadingIndex
from backend.app.indexing.phrase_index import PhraseIndex
from backend.app.retrieval.fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

# Pre-compiled Regex for Phrase extraction
_QUOTE_RE: Final = re.compile(r'"([^"]+)"')


# Added slots=True. Since this class is heavily instantiated per search result,
# this prevents Python from generating a dynamic `__dict__` for every instance,
# drastically saving memory footprint and improving access speeds.
@dataclass(slots=True)
class ScoreBreakdown:
    """Per-node score breakdown for debug output."""
    node_id: str
    fts5_score: float = 0.0
    heading_score: float = 0.0
    phrase_score: float = 0.0
    keyword_coverage: float = 0.0
    structural_score: float = 0.0
    fused_score: float = 0.0


class LexicalRetriever:
    """Searches ``DocumentNode`` records using three vectorless signal sources.

    Args:
        fts5:      The FTS5 full-text index.
        heading: The heading text → node_id index.
        phrase:  The multi-word phrase index.
        node_repo: Used to verify the nodes table is non-empty.
    """

    def __init__(
        self,
        fts5: FTS5Index,
        heading: HeadingIndex,
        phrase: PhraseIndex,
        node_repo: NodeRepository | None = None,
    ) -> None:
        self._fts5 = fts5
        self._heading = heading
        self._phrase = phrase
        self._node_repo = node_repo
        self._last_breakdowns: dict[str, ScoreBreakdown] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        session_id: str | None = None,
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
        if session_node_ids is None and session_id is not None and self._node_repo is not None:
            session_node_ids = {node.id for node in self._node_repo.list_nodes_for_session(session_id)}
            if not session_node_ids:
                logger.warning("LexicalRetriever: nodes table is empty for session %s", session_id)
                return []

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
        # Optimization: Replaced `.setdefault(k, ScoreBreakdown())` with explicit `if not in` checks.
        # Python evaluates arguments eagerly, so `.setdefault` was instantiating memory objects
        # on every single loop iteration even if they weren't used.
        for node_id, score in fts5_results:
            if node_id not in self._last_breakdowns:
                self._last_breakdowns[node_id] = ScoreBreakdown(node_id)
            self._last_breakdowns[node_id].fts5_score = score
            
        for node_id, score in heading_results:
            if node_id not in self._last_breakdowns:
                self._last_breakdowns[node_id] = ScoreBreakdown(node_id)
            self._last_breakdowns[node_id].heading_score = score
            
        for node_id, score in phrase_results:
            if node_id not in self._last_breakdowns:
                self._last_breakdowns[node_id] = ScoreBreakdown(node_id)
            self._last_breakdowns[node_id].phrase_score = score

        # Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            [fts5_results, heading_results, phrase_results],
            top_k=top_k,
        )

        # Filter to session scope
        if session_node_ids is not None:
            fused = [(nid, score) for nid, score in fused if nid in session_node_ids]

        # Final breakdown resolution
        for node_id, score in fused:
            if node_id not in self._last_breakdowns:
                self._last_breakdowns[node_id] = ScoreBreakdown(node_id)
                
            bd = self._last_breakdowns[node_id]
            bd.fused_score = score
            self._populate_text_breakdown(query, bd)

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
            "keyword_coverage": bd.keyword_coverage,
            "structural_score": bd.structural_score,
            "fused_score": bd.fused_score,
        }

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract_phrase_query(query: str) -> str:
        """Extract quoted phrase or return the full query for multi-word terms."""
        # Optimization: Use the walrus operator
        if quoted := _QUOTE_RE.findall(query):
            return quoted[0]
            
        # Optimization: Counting whitespace is faster in CPython than executing `query.split()`
        # which creates a potentially large temporary array in memory.
        if query.strip().count(" ") > 0:
            return query
        return ""

    def _populate_text_breakdown(self, query: str, breakdown: ScoreBreakdown) -> None:
        node = self._get_node(breakdown.node_id)
        if node is None:
            return
            
        # TheFuzz replacement. We combine the text natively and let TheFuzz compute the overlap ratio.
        text_to_match = f"{node.title or ''} {node.text}"
        
        if query.strip():
            # `token_set_ratio` calculates subset intersections identically to the old `_tokens` loop
            # scaled 0 to 100. We divide by 100 to map it perfectly back to 0.0 - 1.0.
            breakdown.keyword_coverage = fuzz.token_set_ratio(query, text_to_match) / 100.0
            
        breakdown.structural_score = self._structural_score(node)

    def _get_node(self, node_id: str) -> DocumentNode | None:
        if self._node_repo is None:
            return None
        try:
            return self._node_repo.get_node_by_id(node_id)
        except DocumentNotFoundError:
            return None

    @staticmethod
    def _structural_score(node: DocumentNode) -> float:
        if node.node_type in {"chapter", "section", "subsection"} or node.title:
            return 1.0
        if node.heading_path:
            return 0.6
        return 0.2
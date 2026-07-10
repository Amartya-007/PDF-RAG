"""RetrievalService — orchestrates all vectorless retrieval for one query.

Pipeline per query
------------------
1. LexicalRetriever    → (node_id, score) list via FTS5 + Heading + Phrase RRF
2. TreeNavigator       → deterministic parent/sibling/child expansion
3. Merge               → deduplicated in-memory union, tree nodes first
4. NodeRanker          → sort merged candidates by pure text signals
5. ConfidenceGate      → abort with InsufficientEvidenceError if score too low
6. Return              → top-k DocumentNode objects + debug info

The service is intentionally thin — it owns no state other than its
collaborators, so it can be called concurrently from the API layer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backend.app.database.repositories.node_repository import NodeRepository
from backend.app.domain.exceptions import InsufficientEvidenceError
from backend.app.domain.models.node import DocumentNode
from backend.app.retrieval.confidence_gate import ConfidenceGate, GateDecision
from backend.app.retrieval.lexical_retriever import LexicalRetriever
from backend.app.retrieval.node_ranker import NodeRanker, RankingResult
from backend.app.retrieval.tree_navigator import TreeNavigator

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Full output of one retrieval call, including debug information."""
    nodes: list[DocumentNode]
    gate_decision: GateDecision
    lexical_hits: list[tuple[str, float]] = field(default_factory=list)
    tree_hits: list[str]                  = field(default_factory=list)
    ranking_results: list[RankingResult]  = field(default_factory=list)
    error: str | None                     = None


class RetrievalService:
    """Thin orchestrator that sequences all vectorless retrieval components."""

    def __init__(
        self,
        node_repo: NodeRepository,
        lexical: LexicalRetriever,
        navigator: TreeNavigator,
        ranker: NodeRanker,
        gate: ConfidenceGate,
        top_k: int = 8,
        lexical_top_k: int = 40,
    ) -> None:
        self._repo     = node_repo
        self._lexical  = lexical
        self._navigator = navigator
        self._ranker   = ranker
        self._gate     = gate
        self._top_k    = top_k
        self._lexical_top_k = lexical_top_k

    # ── Public API ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        cancelled: list[bool] | None = None,
        include_debug: bool = False,
    ) -> RetrievalResult:
        """Run the full retrieval pipeline for *query*.

        Args:
            query:         User question string.
            session_id:    Scope retrieval to a session (prevents cross-session leakage).
            cancelled:     Mutable list; tree navigation stops when True.
            include_debug: Populate ``RetrievalResult`` debug fields.
        """
        if not query.strip():
            return RetrievalResult(
                nodes=[],
                gate_decision=GateDecision(passed=False, score=0.0, reason="Empty query"),
            )

        # 1. Lexical retrieval
        session_node_ids = self._session_node_ids(session_id)
        lexical_hits = self._lexical.search(
            query,
            session_id=session_id,
            top_k=self._lexical_top_k,
            session_node_ids=session_node_ids,
        )

        # FAST FAIL: If no lexical hits, stop immediately. Do not load the whole DB.
        if not lexical_hits:
            logger.info("RetrievalService: No lexical hits found for query.")
            return RetrievalResult(
                nodes=[],
                gate_decision=GateDecision(passed=False, score=0.0, reason="No initial matches found."),
            )

        lexical_ids = {nid for nid, _ in lexical_hits}

        # Fetch lexical nodes from DB (Only ONE DB call needed for the merge process)
        lexical_nodes = self._repo.get_many(list(lexical_ids))

        # 2. Tree navigation: deterministic context expansion around lexical hits
        # Passed `cancelled` down to allow the navigator to abort early if needed.
        tree_nodes = self._navigator.expand(lexical_nodes, self._repo, cancelled=cancelled)
        tree_ids = {n.id for n in tree_nodes}

        # 3. Merge: tree nodes first, then lexical-only hits
        # (Merged purely in-memory using the lexical_nodes we already fetched)
        id_to_node: dict[str, DocumentNode] = {n.id: n for n in tree_nodes}
        
        for node in lexical_nodes:
            if node.id not in id_to_node:
                id_to_node[node.id] = node

        candidates = list(id_to_node.values())

        # 4. Rank merged candidates
        ranking = self._ranker.rank(
            query, candidates, tree_ids, top_k=self._top_k * 4
        )
        ranked_nodes = [r.node for r in ranking[: self._top_k * 4]]

        # 5. Confidence gate
        gate_decision = self._gate.check(query, ranked_nodes[: self._top_k])

        if not gate_decision.passed:
            logger.info(
                "RetrievalService: gate blocked — %s", gate_decision.reason
            )
            return RetrievalResult(
                nodes=[],
                gate_decision=gate_decision,
                error=f"Insufficient evidence: {gate_decision.reason}",
                lexical_hits=lexical_hits if include_debug else [],
                tree_hits=list(tree_ids) if include_debug else [],
                ranking_results=ranking if include_debug else [],
            )

        final_nodes = ranked_nodes[: self._top_k]
        logger.info(
            "RetrievalService: returning %d nodes (gate score=%.3f)",
            len(final_nodes), gate_decision.score,
        )

        return RetrievalResult(
            nodes=final_nodes,
            gate_decision=gate_decision,
            lexical_hits=lexical_hits if include_debug else [],
            tree_hits=list(tree_ids) if include_debug else [],
            ranking_results=ranking if include_debug else [],
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _session_node_ids(self, session_id: str | None) -> set[str] | None:
        if session_id is None:
            return None
        return {n.id for n in self._repo.list_nodes_for_session(session_id)}
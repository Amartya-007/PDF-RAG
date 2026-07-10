"""ConfidenceGate strategies for vectorless retrieval.

This module keeps the legacy gate scoring API while adding the newer
strategy-selection API used by the answer pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Final

from thefuzz import fuzz

from backend.app.domain.enums import QueryType
from backend.app.domain.exceptions import InsufficientEvidenceError
from backend.app.domain.models.node import DocumentNode

logger = logging.getLogger(__name__)


class AnswerStrategy(str, Enum):
    """Answer path selected after retrieval."""

    EXTRACTIVE = "EXTRACTIVE"
    GENERATE = "GENERATE"
    INSUFFICIENT = "INSUFFICIENT"


# Added slots=True for faster instantiation and lower memory overhead during high-throughput validation
@dataclass(frozen=True, slots=True)
class GateDecision:
    """Result of a confidence check, containing the decision, score, and explanation."""
    passed: bool
    score: float
    reason: str


class ConfidenceGate:
    """Guards answer generation by scoring retrieved evidence.
    
    Evaluates the quality of retrieved DocumentNodes based on fuzzy text overlap,
    document density, and retrieval scores.
    """

    # Using Final for static analyzer hints and slight CPython optimizations
    _SCORE_KEYS: Final = (
        "score",
        "fused_score",
        "fast_fact_score",
        "topic_score",
        "keyword_coverage",
        "structural_score",
        "confidence",
    )

    def __init__(
        self,
        threshold: float = 0.15,
        target_count: int = 3,
        w_overlap: float = 0.60,
        w_count: float = 0.25,
        w_density: float = 0.15,
        minimum_score: float | None = None,
        extractive_threshold: float | None = None,
    ) -> None:
        if not (0 < threshold < 1):
            raise ValueError(f"threshold must be in (0, 1); got {threshold}")
        if target_count <= 0:
            raise ValueError(f"target_count must be positive; got {target_count}")

        self._threshold = threshold
        self._minimum_score = (
            threshold if minimum_score is None else self._validate_score(
                "minimum_score", minimum_score
            )
        )
        self._extractive_threshold = (
            threshold
            if extractive_threshold is None
            else self._validate_score("extractive_threshold", extractive_threshold)
        )
        self._target_count = target_count
        self._w_overlap = w_overlap
        self._w_count = w_count
        self._w_density = w_density

    def evaluate(
        self,
        query_type: QueryType,
        ranked_nodes: list[DocumentNode],
        score_details: Mapping[str, Mapping[str, float]],
        use_ollama: bool,
    ) -> AnswerStrategy:
        """Choose an answer strategy from query type and ranking confidence."""
        if not use_ollama:
            return AnswerStrategy.EXTRACTIVE

        score = self._best_rank_score(ranked_nodes, score_details)
        if score < self._minimum_score:
            return AnswerStrategy.INSUFFICIENT

        if query_type in {QueryType.COMPARISON, QueryType.SUMMARY}:
            return AnswerStrategy.GENERATE

        if (
            query_type in {QueryType.FAST_FACT, QueryType.TOPIC}
            and score > self._extractive_threshold
        ):
            return AnswerStrategy.EXTRACTIVE

        return AnswerStrategy.GENERATE

    def check(self, query: str, nodes: list[DocumentNode]) -> GateDecision:
        """Evaluate whether nodes provide sufficient evidence for query."""
        if not nodes:
            return GateDecision(
                passed=False,
                score=0.0,
                reason="No nodes retrieved",
            )

        # Removed manual regex tokenization; passing raw query directly to TheFuzz
        overlap = self._overlap_signal(query, nodes)
        count = self._count_signal(len(nodes))
        density = self._density_signal(nodes)

        score = (
            self._w_overlap * overlap
            + self._w_count * count
            + self._w_density * density
        )

        passed = score >= self._threshold
        reason = (
            f"score={score:.3f} (overlap={overlap:.2f}, count={count:.2f}, "
            f"density={density:.2f}), threshold={self._threshold}"
        )
        return GateDecision(passed=passed, score=score, reason=reason)

    def check_or_raise(self, query: str, nodes: list[DocumentNode]) -> None:
        """Like check, but raises InsufficientEvidenceError on failure."""
        decision = self.check(query, nodes)
        if not decision.passed:
            raise InsufficientEvidenceError(
                f"ConfidenceGate blocked generation: {decision.reason}"
            )

    @classmethod
    def _best_rank_score(
        cls,
        ranked_nodes: list[DocumentNode],
        score_details: Mapping[str, Mapping[str, float]],
    ) -> float:
        best = 0.0
        # Optimization: Walrus operator avoids an extra nested loop if details are empty
        for node in ranked_nodes:
            if details := score_details.get(node.id):
                for key in cls._SCORE_KEYS:
                    value = details.get(key)
                    # Skip expensive float conversions if the current value isn't greater
                    if isinstance(value, (int, float)) and value > best:
                        best = float(value)
        return best

    @staticmethod
    def _validate_score(name: str, value: float) -> float:
        if not (0 <= value <= 1):
            raise ValueError(f"{name} must be in [0, 1]; got {value}")
        return float(value)

    @staticmethod
    def _overlap_signal(query: str, nodes: list[DocumentNode]) -> float:
        """Calculates token overlap using fuzzy matching."""
        if not query.strip():
            return 1.0
            
        # Optimization: Pre-join all nodes into a single string. 
        # TheFuzz parses this much faster natively in C than Python looping.
        combined_text = " ".join(node.text for node in nodes)
        
        # fuzz.token_set_ratio naturally acts like a Jaccard intersection 
        # (ignoring duplicates and finding subset matches). It returns 0-100, so we scale it back to 0.0-1.0
        return fuzz.token_set_ratio(query, combined_text) / 100.0

    def _count_signal(self, n: int) -> float:
        return min(n / self._target_count, 1.0)

    @staticmethod
    def _density_signal(nodes: list[DocumentNode]) -> float:
        if not nodes:
            return 0.0
            
        # Optimization: Replaced `.split()` with pure C `.count(" ")` to avoid 
        # instantiating thousands of string lists in memory per validation run.
        total_words = sum(n.text.count(" ") + 1 for n in nodes)
        avg = total_words / len(nodes)
        return min(avg / 50.0, 1.0)
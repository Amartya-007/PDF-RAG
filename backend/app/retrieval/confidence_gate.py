"""ConfidenceGate — decides whether retrieved context is sufficient to answer.

A lightweight scoring layer between retrieval and generation.  It prevents
the LLM from hallucinating answers for off-topic queries by checking whether
the top-ranked nodes meet a minimum evidence threshold before a generation
call is made.

Decision logic
--------------
  score = w_overlap * query_node_overlap
        + w_count   * min(node_count / target_count, 1.0)
        + w_density * density_ratio

  score >= threshold  →  PASS  (proceed to Answerer)
  score <  threshold  →  FAIL  (return InsufficientEvidenceError)

All weights and thresholds are configurable; defaults are tuned to avoid
false negatives (rejecting good context) at the cost of occasional false
positives (forwarding weak context to the LLM).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.domain.exceptions import InsufficientEvidenceError
from backend.app.domain.models.node import DocumentNode


def _word_set(text: str) -> frozenset[str]:
    return frozenset(w.lower() for w in re.findall(r"\b\w{3,}\b", text))


def _overlap(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)   # recall: how much of query is covered


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    score: float
    reason: str


class ConfidenceGate:
    """Guards generation by verifying retrieved nodes meet a quality bar.

    Args:
        threshold:     Minimum score to proceed with generation (0–1).
        target_count:  Ideal number of nodes; fewer nodes penalise the score.
        w_overlap:     Weight for query-token / node-token overlap.
        w_count:       Weight for the node count signal.
        w_density:     Weight for average node word density.
    """

    def __init__(
        self,
        threshold: float = 0.15,
        target_count: int = 3,
        w_overlap: float = 0.60,
        w_count: float = 0.25,
        w_density: float = 0.15,
    ) -> None:
        if not (0 < threshold < 1):
            raise ValueError(f"threshold must be in (0, 1); got {threshold}")
        self._threshold = threshold
        self._target_count = target_count
        self._w_overlap = w_overlap
        self._w_count = w_count
        self._w_density = w_density

    # ── Public API ─────────────────────────────────────────────────────────

    def check(self, query: str, nodes: list[DocumentNode]) -> GateDecision:
        """Evaluate whether *nodes* provide sufficient evidence for *query*.

        Returns:
            ``GateDecision`` with ``passed=True`` when the score meets the
            threshold.  Does NOT raise; callers inspect ``.passed``.
        """
        if not nodes:
            return GateDecision(
                passed=False,
                score=0.0,
                reason="No nodes retrieved",
            )

        query_words = _word_set(query)
        overlap  = self._overlap_signal(query_words, nodes)
        count    = self._count_signal(len(nodes))
        density  = self._density_signal(nodes)

        score = (
            self._w_overlap * overlap
            + self._w_count   * count
            + self._w_density * density
        )

        passed = score >= self._threshold
        reason = (
            f"score={score:.3f} (overlap={overlap:.2f}, count={count:.2f}, "
            f"density={density:.2f}), threshold={self._threshold}"
        )
        return GateDecision(passed=passed, score=score, reason=reason)

    def check_or_raise(self, query: str, nodes: list[DocumentNode]) -> None:
        """Like ``check`` but raises ``InsufficientEvidenceError`` on failure."""
        decision = self.check(query, nodes)
        if not decision.passed:
            raise InsufficientEvidenceError(
                f"ConfidenceGate blocked generation: {decision.reason}"
            )

    # ── Private signal helpers ─────────────────────────────────────────────

    @staticmethod
    def _overlap_signal(
        query_words: frozenset[str], nodes: list[DocumentNode]
    ) -> float:
        """Fraction of query words covered across all node texts."""
        if not query_words:
            return 1.0
        combined = frozenset(
            w for node in nodes for w in _word_set(node.text)
        )
        return _overlap(query_words, combined)

    def _count_signal(self, n: int) -> float:
        """Score rises to 1.0 as node count approaches target_count."""
        return min(n / self._target_count, 1.0)

    @staticmethod
    def _density_signal(nodes: list[DocumentNode]) -> float:
        """Average word density across nodes (1.0 if avg > 50 words)."""
        if not nodes:
            return 0.0
        avg = sum(len(n.text.split()) for n in nodes) / len(nodes)
        return min(avg / 50.0, 1.0)

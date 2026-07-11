"""HeadingDetector — classifies LayoutNodes as headings or body text.

Uses a weighted scoring system over visual and textual signals rather than
a fixed rule set, so it degrades gracefully when font metadata is absent
(plain text / Markdown / Docling output).

Signal weights (summed; threshold 0.5 to be classified as heading):
  Numbering pattern        0.9  (strongest signal)
  ALL-CAPS short line      0.7
  Font size > 120% body    0.6
  Bold font flag           0.5
  Short line (< 12 words)  0.2
  Blank line above/below   0.25
  Low indent               0.1
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.ingestion.layout_parser import LayoutNode

# ── Compiled patterns (created once at module load) ────────────────────────
_SENTENCE_END = re.compile(r"[.!?]$")
_NUMBER_PATTERNS = [
    re.compile(r"^\s*\d+\.\d+\.\d+[\s\.]"),
    re.compile(r"^\s*\d+\.\d+[\s\.]"),
    re.compile(r"^\s*\d+[\s\.]"),
    re.compile(r"^\s*(chapter|section|part)\s+\w+", re.IGNORECASE),
]
_NUMBER_DEPTHS = [3, 2, 1, 1]

# ── Scoring Weights ────────────────────────────────────────────────────────
_THRESHOLD = 0.5
_W_NUMBERING = 0.9
_W_ALL_CAPS = 0.7
_W_FONT_SIZE = 0.6
_W_BOLD = 0.5
_W_SHORT_LINE = 0.2
_W_SPACING = 0.25
_W_LOW_INDENT = 0.1
_W_LONG_PARA_PENALTY = -0.4


@dataclass
class HeadingResult:
    """The outcome of a layout node classification.

    Attributes:
        is_heading: Boolean classification.
        depth:      Inferred hierarchy level (1–3).
        score:      Raw confidence score from the weighted system.
    """
    is_heading: bool
    depth: int
    score: float


class HeadingDetector:
    """Classifies LayoutNodes as headings or body text.

    Accepts an optional pre-known body font size baseline (useful when the
    caller already knows the document's dominant font size); otherwise the
    baseline is estimated from whichever batch of nodes is passed to
    ``detect()``.
    """

    def __init__(
        self,
        body_font_size_baseline: float | None = None,
        threshold: float = _THRESHOLD,
    ) -> None:
        self._HEADING_THRESHOLD = threshold
        self._provided_baseline = body_font_size_baseline

    def detect(self, nodes: list[LayoutNode]) -> list[HeadingResult]:
        """Classify a batch of layout nodes as headings or body text.

        Attributes:
            nodes: The layout nodes to inspect, in document order. The body
                   font-size baseline is estimated from this batch unless one
                   was supplied to the constructor.
        """
        baseline = (
            self._provided_baseline
            if self._provided_baseline is not None
            else self.estimate_baseline(nodes)
        )
        return [self._classify(node, baseline) for node in nodes]

    def _classify(self, node: LayoutNode, baseline_size: float) -> HeadingResult:
        """Score a single node and classify it as a heading or body text.

        Attributes:
            node:          The layout node to inspect.
            baseline_size: The median font size of the document (for comparison).
        """
        score = 0.0
        depth = self._numbering_depth(node.text)

        # 1. Numbering pattern (Strongest signal)
        if depth > 0:
            score += _W_NUMBERING

        # 2. ALL-CAPS short line
        text = node.text.strip()
        if text.isupper() and len(text.split()) <= 12:
            score += _W_ALL_CAPS

        # 3. Relative font size
        if node.font_size and baseline_size > 0:
            if node.font_size > (baseline_size * 1.2):
                score += _W_FONT_SIZE

        # 4. Bold flag
        if node.is_bold:
            score += _W_BOLD

        # 5. Short line (< 12 words) and no terminal punctuation
        words = text.split()
        if len(words) <= 12 and not _SENTENCE_END.search(text):
            score += _W_SHORT_LINE

        # 6. Spacing check
        if node.line_spacing >= 18.0:
            score += _W_SPACING

        # 7. Length penalty (Long paragraphs are rarely headings)
        if len(words) > 20:
            score += _W_LONG_PARA_PENALTY

        # 8. Low indent
        if node.indent < 10.0:
            score += _W_LOW_INDENT

        is_heading = score >= self._HEADING_THRESHOLD
        return HeadingResult(
            is_heading=is_heading,
            depth=depth if is_heading else 0,
            score=score,
        )

    @staticmethod
    def _numbering_depth(text: str) -> int:
        """Return 1–3 when text starts with a section-number pattern, else 0."""
        for pattern, depth in zip(_NUMBER_PATTERNS, _NUMBER_DEPTHS):
            if pattern.match(text):
                return depth
        return 0

    @staticmethod
    def estimate_baseline(nodes: list[LayoutNode]) -> float:
        """Calculate median font size for nodes that have metadata."""
        sizes = sorted(n.font_size for n in nodes if n.font_size and n.font_size > 0)
        return float(sizes[len(sizes) // 2]) if sizes else 12.0
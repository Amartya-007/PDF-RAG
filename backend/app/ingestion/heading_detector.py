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
  Blank line above/below   0.15
  Low indent               0.1
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.ingestion.layout_parser import LayoutNode


# ── Compiled patterns (created once at module load) ────────────────────────

_NUMBER_PATTERNS = [
    re.compile(r"^\s*\d+\.\d+\.\d+[\s\.]"),        # 1.2.3  → depth 3
    re.compile(r"^\s*\d+\.\d+[\s\.]"),             # 1.2    → depth 2
    re.compile(r"^\s*\d+[\s\.]"),                  # 1.     → depth 1
    re.compile(r"^\s*(chapter|section|part)\s+\w+", re.IGNORECASE),  # depth 1
    re.compile(r"^\s*[IVXLCDM]+[\s\.]"),           # Roman  → depth 1
]
_NUMBER_DEPTHS = [3, 2, 1, 1, 1]  # depth per pattern above

_CAPS_RE = re.compile(r"^[A-Z\d\s\W]{3,60}$")  # ALL-CAPS short line
_SENTENCE_END = re.compile(r"[.!?,;]$")


@dataclass(slots=True)
class HeadingResult:
    """Classification result for a single LayoutNode."""
    is_heading: bool
    depth: int       # 1 = chapter, 2 = section, 3 = subsection; 0 = body
    score: float     # raw signal score (useful for debugging)


class HeadingDetector:
    """Classifies ``LayoutNode`` objects as headings or body text.

    Args:
        body_font_size_baseline: Estimated body font size used for relative
            size comparisons.  When ``None``, the detector auto-estimates
            it from the median font size of the provided nodes.
    """

    _HEADING_THRESHOLD = 0.50
    _SIZE_RATIO = 1.20   # font must be 20% larger than body to score

    def __init__(self, body_font_size_baseline: float | None = None) -> None:
        self._body_baseline = body_font_size_baseline

    def detect(self, nodes: list[LayoutNode]) -> list[HeadingResult]:
        """Classify all *nodes*, returning one ``HeadingResult`` per node."""
        baseline = self._body_baseline or self._estimate_baseline(nodes)
        return [self._classify(node, baseline) for node in nodes]

    # ── Private helpers ────────────────────────────────────────────────────

    def _classify(self, node: LayoutNode, baseline: float) -> HeadingResult:
        text = node.text.strip()
        if not text:
            return HeadingResult(is_heading=False, depth=0, score=0.0)

        score = 0.0
        depth = 1  # default depth when heading signals fire

        # 1. Numbering pattern (highest weight)
        numbering_depth = self._numbering_depth(text)
        if numbering_depth > 0:
            score += 0.90
            depth = numbering_depth

        # 2. ALL-CAPS short line
        if _CAPS_RE.match(text) and len(text.split()) <= 8:
            score += 0.70

        # 3. Font size > 120% body baseline
        if node.font_size and baseline and node.font_size >= baseline * self._SIZE_RATIO:
            score += 0.60
            # Larger font → likely higher-level heading
            ratio = node.font_size / baseline
            if ratio >= 1.6:
                depth = min(depth, 1)
            elif ratio >= 1.35:
                depth = min(depth, 2)

        # 4. Bold font
        if node.is_bold:
            score += 0.50

        # 5. Short line (< 12 words, does not end in sentence-terminal punctuation)
        words = text.split()
        if len(words) <= 12 and not _SENTENCE_END.search(text):
            score += 0.20

        # 6. Subtract for long paragraphs (unlikely to be headings)
        if len(words) > 20:
            score -= 0.40

        # 7. Low indent (top-level sections usually start at left margin)
        if node.indent < 10.0:
            score += 0.10

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
    def _estimate_baseline(nodes: list[LayoutNode]) -> float:
        """Median font size across nodes that have font metadata."""
        sizes = sorted(
            n.font_size for n in nodes if n.font_size and n.font_size > 0
        )
        if not sizes:
            return 12.0  # typical default body font size
        mid = len(sizes) // 2
        return sizes[mid]

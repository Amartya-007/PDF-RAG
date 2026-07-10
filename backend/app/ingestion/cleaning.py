"""Utilities for cleaning extracted text content.

Handles noise reduction (headers/footers) to improve retrieval quality.
"""
from __future__ import annotations

from collections import Counter
from typing import Final

from backend.app.core.text import normalize_space
from backend.app.models import PageText

# Configuration constants
MIN_PAGES_FOR_CLEANING: Final = 3
REPEATED_THRESHOLD_RATIO: Final = 0.6
MAX_LINE_LENGTH: Final = 120


def remove_repeated_headers_footers(pages: list[PageText]) -> list[PageText]:
    """Remove lines that appear as headers/footers across the document.

    Identifies lines appearing in more than 60% of pages (top/bottom 3)
    and strips them to prevent retrieval pollution.

    Attributes:
        pages: List of extracted page text objects.
    """
    if len(pages) < MIN_PAGES_FOR_CLEANING:
        return pages

    line_counts: Counter[str] = Counter()
    page_lines: list[list[str]] = []

    # First pass: normalize and identify candidates
    for page in pages:
        lines = [
            norm for line in page.text.splitlines() 
            if (norm := normalize_space(line))
        ]
        page_lines.append(lines)
        
        # We only count lines that are likely to be headers or footers 
        # (top 3 and bottom 3 lines of the page)
        candidates = set(lines[:3] + lines[-3:])
        line_counts.update(candidates)

    # Calculate threshold based on document length
    threshold = max(MIN_PAGES_FOR_CLEANING, int(len(pages) * REPEATED_THRESHOLD_RATIO))
    
    # Identify lines that meet the frequency and length criteria
    repeated = {
        line for line, count in line_counts.items() 
        if count >= threshold and len(line) < MAX_LINE_LENGTH
    }

    # Second pass: Reconstruct pages omitting the noisy lines
    cleaned: list[PageText] = []
    for page, lines in zip(pages, page_lines):
        kept = [line for line in lines if line not in repeated]
        cleaned.append(
            PageText(
                page_number=page.page_number,
                text="\n".join(kept),
                section_path=page.section_path,
                ocr_confidence=page.ocr_confidence,
            )
        )
    return cleaned
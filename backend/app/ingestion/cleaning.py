from __future__ import annotations

from collections import Counter

from backend.app.core.text import normalize_space
from backend.app.models import PageText


def remove_repeated_headers_footers(pages: list[PageText]) -> list[PageText]:
    if len(pages) < 3:
        return pages

    line_counts: Counter[str] = Counter()
    page_lines: list[list[str]] = []
    for page in pages:
        lines = [normalize_space(line) for line in page.text.splitlines() if normalize_space(line)]
        page_lines.append(lines)
        candidates = lines[:3] + lines[-3:]
        line_counts.update(set(candidates))

    threshold = max(3, int(len(pages) * 0.6))
    repeated = {line for line, count in line_counts.items() if count >= threshold and len(line) < 120}

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

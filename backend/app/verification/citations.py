from __future__ import annotations

import re

from backend.app.models import Citation


def has_supported_citation(answer: str, citations: list[Citation]) -> bool:
    if not citations:
        return False
    source_ids = {citation.source_id for citation in citations}
    cited = set(re.findall(r"\[(S\d+)\]", answer))
    return bool(cited & source_ids)


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))


def unsupported_numbers(answer: str, citations: list[Citation]) -> set[str]:
    evidence_numbers: set[str] = set()
    for citation in citations:
        evidence_numbers |= extract_numbers(citation.excerpt)
    return extract_numbers(answer) - evidence_numbers

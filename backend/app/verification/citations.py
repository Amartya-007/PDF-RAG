"""Citation verification and extraction utilities."""
from __future__ import annotations

import re
from typing import Final

from backend.app.models import Citation

# Pre-compiled regex for efficiency
_SOURCE_RE: Final = re.compile(r"\[(S\d+)\]")
_NUMBER_RE: Final = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def has_supported_citation(answer: str, citations: list[Citation]) -> bool:
    """Verifies that at least one citation used in the answer is valid."""
    if not citations:
        return False
        
    source_ids = {c.source_id for c in citations}
    # Extract unique [Sx] patterns from answer
    cited = set(_SOURCE_RE.findall(answer))
    return bool(cited & source_ids)


def extract_numbers(text: str) -> set[str]:
    """Extracts all numbers, percentages, and decimals from text."""
    return set(_NUMBER_RE.findall(text))


def unsupported_numbers(answer: str, citations: list[Citation]) -> set[str]:
    """Identifies numbers in the answer that do not appear in any citation excerpt."""
    # Build set of all numbers present in the provided evidence
    evidence_numbers = {
        num for citation in citations 
        for num in _NUMBER_RE.findall(citation.excerpt)
    }
    
    # Return numbers found in answer not present in evidence
    return extract_numbers(answer) - evidence_numbers
"""Citation validation for generated answers."""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.models import Citation
from backend.app.verification.citations import extract_numbers, has_supported_citation

# Regex patterns pre-compiled for performance
_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9'-]{2,}\b")
_SOURCE_RE = re.compile(r"\[(S\d+)\]")
_PROPER_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_DATE_RE = re.compile(
    r"\b(?:\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"[a-z]*\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)

_DIRECTION_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"increase", "increased", "increases", "increasing", "rise", "rose", "up"}),
    frozenset({"decrease", "decreased", "decreases", "decreasing", "fall", "fell", "down"}),
    frozenset({"before", "prior", "earlier"}),
    frozenset({"after", "following", "later"}),
    frozenset({"above", "over", "greater"}),
    frozenset({"below", "under", "less"}),
)

_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "into", "onto",
        "was", "were", "are", "has", "have", "had", "its", "their", "there",
        "source", "according", "citation",
    }
)


@dataclass(frozen=True)
class CitationValidator:
    """Validates that an answer is supported by its cited excerpts."""

    min_keyword_overlap: float = 0.35

    def validate(self, answer_text: str, citations: list[Citation]) -> bool:
        """Checks if the answer is supported by evidence and meets overlap thresholds."""
        if not answer_text.strip() or not citations or not has_supported_citation(answer_text, citations):
            return False

        cited = self._cited_citations(answer_text, citations)
        if not cited:
            return False

        evidence = " ".join(c.excerpt for c in cited)
        
        # Verify content support across different dimensions
        return (
            self._keyword_overlap_supported(answer_text, evidence)
            and self._numbers_supported(answer_text, evidence)
            and self._dates_supported(answer_text, evidence)
            and self._proper_names_supported(answer_text, evidence)
            and self._directions_supported(answer_text, evidence)
        )

    @staticmethod
    def _cited_citations(answer_text: str, citations: list[Citation]) -> list[Citation]:
        """Maps found citation IDs in the text to their corresponding objects."""
        source_ids = set(_SOURCE_RE.findall(answer_text))
        return [c for c in citations if c.source_id in source_ids]

    def _keyword_overlap_supported(self, answer_text: str, evidence: str) -> bool:
        """Checks if key terminology in the answer appears in the evidence."""
        answer_terms = self._terms(_SOURCE_RE.sub("", answer_text))
        if not answer_terms:
            return True
            
        evidence_terms = self._terms(evidence)
        overlap = len(answer_terms & evidence_terms) / len(answer_terms)
        return overlap >= self.min_keyword_overlap

    @staticmethod
    def _numbers_supported(answer_text: str, evidence: str) -> bool:
        """Verifies that all numerical claims are present in the evidence."""
        return extract_numbers(answer_text) <= extract_numbers(evidence)

    @staticmethod
    def _dates_supported(answer_text: str, evidence: str) -> bool:
        """Verifies that all dates cited are present in the evidence."""
        answer_dates = {m.group(0).lower() for m in _DATE_RE.finditer(answer_text)}
        evidence_dates = {m.group(0).lower() for m in _DATE_RE.finditer(evidence)}
        return answer_dates <= evidence_dates

    @staticmethod
    def _proper_names_supported(answer_text: str, evidence: str) -> bool:
        """Verifies that all named entities cited are present in the evidence."""
        clean_text = _SOURCE_RE.sub("", answer_text)
        answer_names = {m.group(0).lower() for m in _PROPER_NAME_RE.finditer(clean_text)}
        evidence_names = {m.group(0).lower() for m in _PROPER_NAME_RE.finditer(evidence)}
        return answer_names <= evidence_names

    @staticmethod
    def _directions_supported(answer_text: str, evidence: str) -> bool:
        """Ensures directional trends in the answer are reflected in the evidence."""
        answer_terms = CitationValidator._terms(answer_text)
        evidence_terms = CitationValidator._terms(evidence)
        
        for group in _DIRECTION_GROUPS:
            if (answer_terms & group) and not (evidence_terms & group):
                return False
        return True

    @staticmethod
    def _terms(text: str) -> frozenset[str]:
        """Extracts significant terms, excluding stopwords."""
        return frozenset(
            word.lower()
            for word in _WORD_RE.findall(text)
            if word.lower() not in _STOPWORDS
        )
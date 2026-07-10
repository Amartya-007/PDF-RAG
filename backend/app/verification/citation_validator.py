"""Citation validation for generated answers."""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.models import Citation
from backend.app.verification.citations import (
    extract_numbers,
    has_supported_citation,
)


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
        """Return True when cited evidence supports the answer text."""
        if not answer_text.strip() or not citations:
            return False
        if not has_supported_citation(answer_text, citations):
            return False

        cited = self._cited_citations(answer_text, citations)
        if not cited:
            return False

        evidence = " ".join(citation.excerpt for citation in cited)
        return (
            self._keyword_overlap_supported(answer_text, evidence)
            and self._numbers_supported(answer_text, evidence)
            and self._dates_supported(answer_text, evidence)
            and self._proper_names_supported(answer_text, evidence)
            and self._directions_supported(answer_text, evidence)
        )

    @staticmethod
    def _cited_citations(
        answer_text: str,
        citations: list[Citation],
    ) -> list[Citation]:
        source_ids = set(_SOURCE_RE.findall(answer_text))
        by_source = {citation.source_id: citation for citation in citations}
        return [
            by_source[source_id]
            for source_id in source_ids
            if source_id in by_source
        ]

    def _keyword_overlap_supported(self, answer_text: str, evidence: str) -> bool:
        answer_terms = self._terms(_SOURCE_RE.sub("", answer_text))
        if not answer_terms:
            return True
        evidence_terms = self._terms(evidence)
        overlap = len(answer_terms & evidence_terms) / len(answer_terms)
        return overlap >= self.min_keyword_overlap

    @staticmethod
    def _numbers_supported(answer_text: str, evidence: str) -> bool:
        return extract_numbers(answer_text) <= extract_numbers(evidence)

    @staticmethod
    def _dates_supported(answer_text: str, evidence: str) -> bool:
        answer_dates = {
            match.group(0).lower()
            for match in _DATE_RE.finditer(answer_text)
        }
        evidence_dates = {
            match.group(0).lower()
            for match in _DATE_RE.finditer(evidence)
        }
        return answer_dates <= evidence_dates

    @staticmethod
    def _proper_names_supported(answer_text: str, evidence: str) -> bool:
        answer_names = {
            match.group(0).lower()
            for match in _PROPER_NAME_RE.finditer(_SOURCE_RE.sub("", answer_text))
        }
        evidence_names = {
            match.group(0).lower()
            for match in _PROPER_NAME_RE.finditer(evidence)
        }
        return answer_names <= evidence_names

    @staticmethod
    def _directions_supported(answer_text: str, evidence: str) -> bool:
        answer_terms = CitationValidator._terms(answer_text)
        evidence_terms = CitationValidator._terms(evidence)
        for group in _DIRECTION_GROUPS:
            answer_hits = answer_terms & group
            if answer_hits and not (evidence_terms & group):
                return False
        return True

    @staticmethod
    def _terms(text: str) -> frozenset[str]:
        return frozenset(
            word.lower()
            for word in _WORD_RE.findall(text)
            if word.lower() not in _STOPWORDS
        )


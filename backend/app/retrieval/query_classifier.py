"""Query classification — single authoritative location.

Previously split between ``retrieval/query_analysis.py`` (6 regex branches)
and two inline methods in ``RagService`` (``_is_fast_fact_query``,
``_is_topic_query``).  Having them in three places meant identical logic
drifting out of sync.

``QueryClassifier.classify`` returns a ``QueryType`` enum value that the
retrieval service uses to choose the right search strategy.

Complexity: O(Q) where Q = number of regex patterns × question length.
All patterns are compiled once at class creation.
"""
from __future__ import annotations

import re
from functools import lru_cache

from backend.app.domain.enums import QueryType


# ── Compiled pattern sets (compiled once, reused every call) ──────────────

# Fast-fact: resume-style structured fields that BM25 resolves without embedding
_FAST_FACT_TERMS: frozenset[str] = frozenset({
    "full name", "person name", "candidate name", "user name", "student name",
    "name", "college", "collage", "university", "institute", "school",
    "degree", "course", "branch", "program",
    "cgpa", "gpa", "email", "mail", "phone", "mobile", "contact",
})

# Topic: "explain / describe / tell me everything about X"
_TOPIC_PATTERN = re.compile(
    r"\b(what is|what are|define|explain|describe|"
    r"tell me(?: everything| all)? about|everything about)\b",
    re.IGNORECASE,
)
_TOPIC_DETAIL_WORDS: frozenset[str] = frozenset({
    "in detail", "details about",
})

# Other intent patterns
_COMPARISON_RE = re.compile(r"\b(compare|difference|versus|vs\.?)\b", re.IGNORECASE)
_SUMMARY_RE = re.compile(r"\b(summarize|summary|overview)\b", re.IGNORECASE)
_NUMERIC_RE = re.compile(
    r"\b(table|highest|lowest|total|amount|number|date|when)\b", re.IGNORECASE
)
_IDENTIFIER_RE = re.compile(r"\b(id|order|clause|section|code|no\.)\b", re.IGNORECASE)

# Normalisation corrections (typos common in Indian-English academic context)
_NORMALISE_MAP: list[tuple[str, str]] = [
    ("collage", "college"),
    ("Collage", "College"),
    ("transection", "transaction"),
    ("Transection", "Transaction"),
    ("whats", "what is"),
    ("Whats", "What is"),
]


class QueryClassifier:
    """Classifies a user question into a ``QueryType``.

    Design
    ------
    - All regex patterns compiled at class instantiation (not per call).
    - ``normalise`` is a pure function so it can be cached at the call site.
    - ``classify`` is deterministic and side-effect free.
    """

    def normalise(self, question: str) -> str:
        """Apply spelling corrections common in the target user base."""
        result = question
        for wrong, right in _NORMALISE_MAP:
            result = result.replace(wrong, right)
        return result

    def classify(self, question: str) -> QueryType:
        """Return the most specific QueryType for *question*.

        Priority order (most specific → most general):
          1. FastFact  – resume / structured-data field lookup
          2. Topic     – definition / explanation request
          3. Comparison
          4. Summary
          5. NumericOrTable
          6. ExactIdentifier
          7. FollowUpOrShort
          8. DirectFactual (default)
        """
        normalised = self.normalise(question).lower()

        if self._is_fast_fact(normalised):
            return QueryType.FAST_FACT

        if self._is_topic(normalised):
            return QueryType.TOPIC

        if _COMPARISON_RE.search(normalised):
            return QueryType.COMPARISON

        if _SUMMARY_RE.search(normalised):
            return QueryType.SUMMARY

        if _NUMERIC_RE.search(normalised):
            return QueryType.NUMERIC_OR_TABLE

        if _IDENTIFIER_RE.search(normalised):
            return QueryType.EXACT_IDENTIFIER

        if len(question.split()) <= 4:
            return QueryType.FOLLOW_UP_OR_SHORT

        return QueryType.DIRECT_FACTUAL

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_fast_fact(self, normalised: str) -> bool:
        """True when the question is asking for a specific structured field."""
        return any(term in normalised for term in _FAST_FACT_TERMS)

    def _is_topic(self, normalised: str) -> bool:
        """True when the question is asking for a definition or explanation."""
        if self._is_fast_fact(normalised):
            return False  # fast-fact takes priority
        if _TOPIC_PATTERN.search(normalised):
            return True
        return any(detail in normalised for detail in _TOPIC_DETAIL_WORDS)

    def is_fast_fact(self, question: str) -> bool:
        """Public convenience method for callers that only need fast-fact check."""
        return self._is_fast_fact(self.normalise(question).lower())

    def is_topic(self, question: str) -> bool:
        """Public convenience method for callers that only need topic check."""
        normalised = self.normalise(question).lower()
        return self._is_topic(normalised)

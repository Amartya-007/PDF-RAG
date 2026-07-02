"""Backward-compatibility shim.

All classification logic now lives in ``query_classifier.QueryClassifier``.
This module re-exports ``classify_query`` so existing callers keep working
without changes.
"""
from __future__ import annotations

from backend.app.retrieval.query_classifier import QueryClassifier

_classifier = QueryClassifier()


def classify_query(question: str) -> str:
    """Return the query type string for *question*.

    Deprecated: prefer ``QueryClassifier.classify()`` which returns a
    ``QueryType`` enum.  This function is kept for backward compatibility.
    """
    return _classifier.classify(question).value

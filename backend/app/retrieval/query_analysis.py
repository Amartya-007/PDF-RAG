"""Backward-compatibility shim.

All classification logic now lives in ``query_classifier.QueryClassifier``.
This module re-exports ``classify_query`` so existing callers keep working
without changes.
"""
from __future__ import annotations

import warnings
from backend.app.retrieval.query_classifier import QueryClassifier

# Instantiate once at module load (zero per-call overhead)
_classifier = QueryClassifier()


def classify_query(question: str) -> str:
    """Return the query type string for *question*.

    Deprecated: prefer ``QueryClassifier.classify()`` which returns a
    ``QueryType`` enum. This function is kept for backward compatibility.
    """
    warnings.warn(
        "classify_query is deprecated and will be removed in a future update. "
        "Please migrate to QueryClassifier.classify() which returns a QueryType enum.",
        DeprecationWarning,
        stacklevel=2,  # Points the warning to the file that called this function
    )
    return _classifier.classify(question).value
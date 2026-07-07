"""Query domain model.

``Query`` wraps a classified, normalised user question together with its
``QueryType`` so that the retrieval and answer pipelines can operate on a
single typed value instead of raw strings.

``QueryType`` is re-exported here from ``domain.enums`` so callers only need
to import from one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Re-export QueryType so importers can do:
#   from backend.app.domain.models.query import Query, QueryType
from backend.app.domain.enums import QueryType  # noqa: F401


@dataclass(frozen=True)
class Query:
    """A classified, normalised user question ready for retrieval.

    Attributes:
        question:    The normalised question text.
        query_type:  Classification of the question (FAST_FACT, TOPIC, …).
        session_id:  The chat session this query belongs to.
        raw:         The original, unmodified question text before normalisation.
        metadata:    Optional bag of extra data (e.g. extracted entities).
    """

    question: str
    query_type: QueryType
    session_id: str = "default"
    raw: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

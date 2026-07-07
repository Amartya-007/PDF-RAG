"""Domain model definitions for the Local PDF RAG system.

Models are organised by concern:
  node.py      — DocumentNode (hierarchical tree unit)
  document.py  — Document, ChatSession
  answer.py    — Answer, Citation
  query.py     — Query, QueryType
  job.py       — IngestionJob
"""
from __future__ import annotations

from backend.app.domain.models.node import DocumentNode, stable_id
from backend.app.domain.models.document import Document, ChatSession
from backend.app.domain.models.answer import Answer, Citation
from backend.app.domain.models.query import Query, QueryType
from backend.app.domain.models.job import IngestionJob

__all__ = [
    "DocumentNode",
    "stable_id",
    "Document",
    "ChatSession",
    "Answer",
    "Citation",
    "Query",
    "QueryType",
    "IngestionJob",
]

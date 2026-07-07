"""Repository classes for database access.

Each repository owns the SQL for one domain entity.
``MetadataStore`` delegates to these classes internally while keeping
its existing public interface unchanged.
"""
from __future__ import annotations

from backend.app.database.repositories.answer_repository import AnswerRepository
from backend.app.database.repositories.document_repository import DocumentRepository
from backend.app.database.repositories.job_repository import JobRepository
from backend.app.database.repositories.node_repository import NodeRepository
from backend.app.database.repositories.session_repository import SessionRepository

__all__ = [
    "AnswerRepository",
    "DocumentRepository",
    "JobRepository",
    "NodeRepository",
    "SessionRepository",
]

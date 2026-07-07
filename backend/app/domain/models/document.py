"""Document and ChatSession domain models.

These are the authoritative definitions.  ``backend.app.models`` re-exports
them for backward compatibility during migration.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatSession:
    """A workspace that groups documents and conversation messages."""

    session_id: str
    title: str
    created_at: str = ""


@dataclass(frozen=True)
class Document:
    """Metadata record for an ingested source file.

    Attributes:
        document_id: Stable SHA-256-derived identifier.
        filename:    Original filename as uploaded.
        sha256:      SHA-256 hex digest of the file bytes.
        path:        Absolute path to the managed copy in documents_dir.
        status:      Current lifecycle state (pending → processing → ready|failed).
        session_id:  Chat session this document belongs to.
    """

    document_id: str
    filename: str
    sha256: str
    path: str
    status: str = "ready"     # kept as str for SQLite compat; validated at boundaries
    session_id: str = "default"

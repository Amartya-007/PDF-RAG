"""Domain-specific exceptions for the Local PDF RAG system.

Every subsystem raises one of these instead of bare RuntimeError or
ValueError so callers can catch exactly what they need to.
"""
from __future__ import annotations


class RagError(Exception):
    """Base class for all RAG domain errors."""


# ── Ingestion ──────────────────────────────────────────────────────────────


class IngestionError(RagError):
    """Raised when a document cannot be ingested successfully."""


class ParseError(IngestionError):
    """Raised when PDF / text extraction produces no usable content."""


class DuplicateDocumentError(IngestionError):
    """Raised when an identical document is already indexed in this session.

    Attributes:
        document_id: The ID of the existing document.
        filename:    The filename of the document.
    """

    def __init__(self, document_id: str, filename: str) -> None:
        self.document_id = document_id
        self.filename = filename
        super().__init__(
            f"Document '{filename}' (id={document_id}) is already indexed in this session."
        )


class UnsupportedFileTypeError(IngestionError):
    """Raised when the uploaded file type is not supported.

    Attributes:
        suffix: The file extension that was rejected.
    """

    def __init__(self, suffix: str) -> None:
        self.suffix = suffix
        super().__init__(
            f"Unsupported file type: '{suffix}'. Only .pdf, .txt, and .md are accepted."
        )


class EmptyDocumentError(IngestionError):
    """Raised when a document produces zero text chunks after parsing."""


# ── Embedding ──────────────────────────────────────────────────────────────


class EmbeddingError(RagError):
    """Raised when the embedding provider fails or returns unexpected output."""


# ── Retrieval ──────────────────────────────────────────────────────────────


class RetrievalError(RagError):
    """Raised when the retrieval pipeline encounters an unrecoverable error."""


# ── Generation ─────────────────────────────────────────────────────────────


class GenerationError(RagError):
    """Raised when the LLM provider fails to generate a response."""


# ── Storage ────────────────────────────────────────────────────────────────


class StorageError(RagError):
    """Raised when a database or index operation fails."""


class DocumentNotFoundError(StorageError):
    """Raised when a requested document does not exist in the store.

    Attributes:
        document_id: The missing document's identifier.
    """

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(f"Document '{document_id}' not found.")


class SessionNotFoundError(StorageError):
    """Raised when a requested chat session does not exist.

    Attributes:
        session_id: The missing session's identifier.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session '{session_id}' not found.")


# ── Index ──────────────────────────────────────────────────────────────────


class IndexCorruptionError(RagError):
    """Raised when a persisted index file is unreadable or structurally invalid."""


# ── Answer generation ──────────────────────────────────────────────────────


class AnswerGenerationError(RagError):
    """Raised when the LLM answerer fails to produce a valid response."""


class InsufficientEvidenceError(RetrievalError):
    """Raised by ConfidenceGate when retrieved nodes don't meet the quality bar."""
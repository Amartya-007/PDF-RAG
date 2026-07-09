"""Backward-compatible import for the vectorless RAG coordinator."""
from __future__ import annotations

from backend.app.services.rag_service_v2 import RagServiceV2


RagService = RagServiceV2

__all__ = ["RagService", "RagServiceV2"]

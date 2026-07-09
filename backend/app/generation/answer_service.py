"""AnswerService — single entry point for all answer generation.

Routes each query through the correct answerer and handles the
fast-path / streaming / fallback cascade transparently.

Routing logic
-------------
1. No retrieved nodes → InsufficientEvidenceError (raised or in Answer)
2. Fast-fact question  → ExtractiveAnswerer (instant, no Ollama)
3. Ollama available    → OllamaAnswerer (streaming or blocking)
4. Ollama unavailable  → ExtractiveAnswerer (fallback)

The service is intentionally stateless between calls; all mutable
state lives in the answerer collaborators.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

from backend.app.domain.exceptions import AnswerGenerationError, InsufficientEvidenceError
from backend.app.domain.models.node import DocumentNode
from backend.app.generation.extractive_answerer import ExtractiveAnswerer
from backend.app.generation.ollama_answerer import OllamaAnswerer
from backend.app.models import Answer

logger = logging.getLogger(__name__)

_INSUFFICIENT = (
    "I could not find sufficient evidence in the documents to answer this question."
)


class AnswerService:
    """Routes question + nodes to the appropriate answerer.

    Args:
        extractive:      Pure text sentence extractor (always available).
        ollama:          LLM-backed answerer; may be ``None`` when Ollama
                         is disabled or unreachable.
    """

    def __init__(
        self,
        extractive: ExtractiveAnswerer,
        ollama: OllamaAnswerer | None,
    ) -> None:
        self._extractive = extractive
        self._ollama = ollama

    # ── Blocking ───────────────────────────────────────────────────────────

    def answer(self, question: str, nodes: list[DocumentNode]) -> Answer:
        """Generate a complete answer for *question* given retrieved *nodes*.

        Returns a valid ``Answer`` in all cases (never raises to callers).
        """
        if not nodes:
            return Answer(
                question=question, answer=_INSUFFICIENT,
                citations=[], answerable=False,
            )

        # 1. Fast-fact extractive path
        if self._extractive.is_fast_fact_question(question):
            logger.debug("AnswerService: fast-fact path for %r", question[:60])
            result = self._extractive.answer(question, nodes)
            if result.answerable:
                return result

        # 2. Ollama path
        if self._ollama is not None:
            try:
                return self._ollama.answer(question, nodes)
            except (AnswerGenerationError, Exception) as exc:
                logger.warning("AnswerService: Ollama failed, falling back: %s", exc)

        # 3. Extractive fallback
        return self._extractive.answer(question, nodes)

    # ── Streaming ──────────────────────────────────────────────────────────

    def answer_stream(
        self,
        question: str,
        nodes: list[DocumentNode],
        cancelled: list[bool] | None = None,
    ) -> Iterator[str | Answer]:
        """Yield text fragments then the final ``Answer``.

        Fast-fact and fallback paths yield the full text as one fragment.
        Ollama path yields individual token bursts.
        """
        if not nodes:
            msg = _INSUFFICIENT
            yield msg
            yield Answer(question=question, answer=msg, citations=[], answerable=False)
            return

        # 1. Fast-fact extractive path
        if self._extractive.is_fast_fact_question(question):
            result = self._extractive.answer(question, nodes)
            if result.answerable:
                yield result.answer
                yield result
                return

        # 2. Ollama streaming path
        if self._ollama is not None:
            try:
                yield from self._ollama.answer_stream(question, nodes, cancelled)
                return
            except (AnswerGenerationError, Exception) as exc:
                logger.warning("AnswerService stream: Ollama failed, falling back: %s", exc)

        # 3. Extractive fallback
        result = self._extractive.answer(question, nodes)
        yield result.answer
        yield result

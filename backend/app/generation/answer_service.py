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
from typing import Final

from backend.app.domain.exceptions import AnswerGenerationError
from backend.app.domain.models.node import DocumentNode
from backend.app.generation.extractive_answerer import ExtractiveAnswerer
from backend.app.generation.ollama_answerer import OllamaAnswerer
from backend.app.models import Answer

logger = logging.getLogger(__name__)

# --- Class-level Constants ---
_INSUFFICIENT: Final[str] = (
    "I could not find sufficient evidence in the documents to answer this question."
)


class AnswerService:
    """Routes question + nodes to the appropriate answerer orchestration component.

    This service implements the cascade fallback strategy. It ensures that quick,
    factual queries skip language model inference overhead entirely, and provides
    resiliency by falling back to text-overlap extraction if downstream LLM
    providers time out or encounter runtime exceptions.

    Attributes:
        _extractive (ExtractiveAnswerer): Pure text sentence extraction engine.
        _ollama (OllamaAnswerer | None): Optional LLM-backed model orchestrator.
    """

    def __init__(
        self,
        extractive: ExtractiveAnswerer,
        ollama: OllamaAnswerer | None,
    ) -> None:
        """Initializes the AnswerService component.

        Args:
            extractive: Pure text sentence extractor (always available).
            ollama: LLM-backed answerer; may be ``None`` when Ollama
                is disabled or unreachable.
        """
        self._extractive: ExtractiveAnswerer = extractive
        self._ollama: OllamaAnswerer | None = ollama

    # ── Blocking ───────────────────────────────────────────────────────────

    def answer(self, question: str, nodes: list[DocumentNode]) -> Answer:
        """Generate a complete answer for *question* given retrieved *nodes*.

        Evaluates the input query string to determine if it can be resolved via
        the fast-fact fallback track. If not, it requests inference from the LLM provider,
        gracefully sliding back to extractive keyword fallback if exceptions arise.

        Args:
            question: The incoming raw natural language question from the client.
            nodes: A list of contextually retrieved document chunks.

        Returns:
            A valid fully hydrated ``Answer`` model instance (never raises exceptions to callers).
        """
        # Fast exit strategy: no documents provided implies no validation source available
        if not nodes:
            return Answer(
                question=question,
                answer=_INSUFFICIENT,
                citations=[],
                answerable=False,
            )

        # 1. Fast-fact extractive path (skips LLM overhead for predictable factual answers)
        if self._extractive.is_fast_fact_question(question):
            logger.debug("AnswerService: fast-fact path triggered for %r", question[:60])
            result = self._extractive.answer(question, nodes)
            if result.answerable:
                return result

        # 2. Ollama generative path
        if self._ollama is not None:
            try:
                return self._ollama.answer(question, nodes)
            except (AnswerGenerationError, Exception) as exc:
                # Capture complete traceback logs silently to monitor service reliability degration
                logger.warning(
                    "AnswerService: Ollama execution failed, executing cascade fallback mechanism: %s",
                    exc,
                    exc_info=True,
                )

        # 3. Extractive fallback path (executed if LLM fails or is un-configured)
        logger.debug("AnswerService: Falling back to heuristic text-overlap extraction for %r", question[:60])
        return self._extractive.answer(question, nodes)

    # ── Streaming ──────────────────────────────────────────────────────────

    def answer_stream(
        self,
        question: str,
        nodes: list[DocumentNode],
        cancelled: list[bool] | None = None,
    ) -> Iterator[str | Answer]:
        """Yield text fragments iteratively followed by the final verified ``Answer``.

        Fast-fact and fallback paths yield the full text as one contiguous fragment.
        Ollama streaming paths yield individual raw token bursts as they arrive.

        Args:
            question: The incoming raw natural language question from the client.
            nodes: A list of contextually retrieved document chunks.
            cancelled: Optional shared state reference hook to cancel long-running generator chunks.

        Yields:
            Individual text chunk slices (`str`) followed by the fully instantiated `Answer` metadata model.
        """
        # Fast exit strategy for empty document arrays
        if not nodes:
            msg = _INSUFFICIENT
            yield msg
            yield Answer(question=question, answer=msg, citations=[], answerable=False)
            return

        # 1. Fast-fact extractive path
        if self._extractive.is_fast_fact_question(question):
            result = self._extractive.answer(question, nodes)
            if result.answerable:
                # Cache local string lookups to minimize micro-allocation lookups across generator boundaries
                answer_text = result.answer
                yield answer_text
                yield result
                return

        # 2. Ollama streaming inference path
        if self._ollama is not None:
            try:
                yield from self._ollama.answer_stream(question, nodes, cancelled)
                return
            except (AnswerGenerationError, Exception) as exc:
                logger.warning(
                    "AnswerService stream: Ollama generator failed, falling back to extractive pass: %s",
                    exc,
                    exc_info=True,
                )

        # 3. Extractive fallback generation path
        result = self._extractive.answer(question, nodes)
        yield result.answer
        yield result
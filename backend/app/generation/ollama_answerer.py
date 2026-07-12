"""OllamaAnswerer — streaming LLM generation via Ollama.

Wraps OllamaClient to provide both a blocking ``answer()`` and an
incremental ``answer_stream()`` that yields text fragments for the web
token-streaming pipeline.

Context assembly
----------------
The answerer receives pre-ranked ``DocumentNode`` objects from
``RetrievalService``.  It formats them into an evidence block:

  [S1] filename, page N
  <text of node 1>

  [S2] filename, page M
  <text of node 2>
  ...

Then passes the block + user question to ``build_answer_prompt()`` and
streams the response token by token.

Citation validation
-------------------
After generation completes, ``CitationValidator`` verifies that the cited
text supports the generated answer. Invalid generated answers fall back to
pure extractive answering.

Error handling
--------------
All ``GenerationError`` exceptions from ``OllamaClient`` are caught and
re-raised as ``AnswerGenerationError`` so callers have a single exception
type to handle.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

from backend.app.domain.exceptions import AnswerGenerationError
from backend.app.domain.models.node import DocumentNode
from backend.app.generation.extractive_answerer import ExtractiveAnswerer
from backend.app.generation.ollama_client import GenerationError, OllamaClient
from backend.app.generation.prompts import build_answer_prompt
from backend.app.models import Answer, Citation
from backend.app.verification.citation_validator import CitationValidator

logger = logging.getLogger(__name__)

_MAX_NODE_WORDS = 200   # truncate each evidence block to keep prompt manageable


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")


def _build_evidence_block(nodes: list[DocumentNode]) -> tuple[str, list[tuple[str, DocumentNode]]]:
    """Build evidence block string + a [(source_id, node)] reference list."""
    lines: list[str] = []
    references: list[tuple[str, DocumentNode]] = []
    for i, node in enumerate(nodes, start=1):
        source_id = f"S{i}"
        page_info = f"page {node.page_start}"
        if node.page_end and node.page_end != node.page_start:
            page_info = f"pages {node.page_start}–{node.page_end}"
        lines.append(f"[{source_id}] {node.document_id}, {page_info}")
        lines.append(_truncate(node.text, _MAX_NODE_WORDS))
        lines.append("")
        references.append((source_id, node))
    return "\n".join(lines), references


def _nodes_to_citations(
    generated_text: str,
    references: list[tuple[str, DocumentNode]],
) -> list[Citation]:
    """Extract citations that are actually mentioned in *generated_text*."""
    citations: list[Citation] = []
    seen: set[str] = set()
    for source_id, node in references:
        if f"[{source_id}]" in generated_text and source_id not in seen:
            seen.add(source_id)
            citations.append(Citation(
                source_id=source_id,
                document_id=node.document_id,
                chunk_id=node.id,
                filename=node.document_id,
                page_start=node.page_start,
                page_end=node.page_end or node.page_start,
                excerpt=node.text[:300],
                heading_path=list(node.heading_path),
            ))
    return citations


class OllamaAnswerer:
    """Generates answers from DocumentNodes using OllamaClient.

    Args:
        client: Configured ``OllamaClient`` instance.
    """

    def __init__(
        self,
        client: OllamaClient,
        extractive: ExtractiveAnswerer | None = None,
        validator: CitationValidator | None = None,
    ) -> None:
        self._client = client
        self._extractive = extractive or ExtractiveAnswerer()
        self._validator = validator or CitationValidator()

    # ── Blocking ───────────────────────────────────────────────────────────

    def answer(self, question: str, nodes: list[DocumentNode]) -> Answer:
        """Generate a complete answer synchronously.

        Raises:
            AnswerGenerationError: On any Ollama connection or generation failure.
        """
        evidence, references = _build_evidence_block(nodes)
        prompt = build_answer_prompt(question, evidence)
        try:
            text = self._client.generate(prompt).strip()
        except GenerationError as exc:
            raise AnswerGenerationError(str(exc)) from exc

        citations = _nodes_to_citations(text, references)
        if not self._validator.validate(text, citations):
            return self._extractive.answer(question, nodes)

        return Answer(
            question=question,
            answer=text,
            citations=citations,
            answerable=bool(text),
        )

    # ── Streaming ──────────────────────────────────────────────────────────

    def answer_stream(
        self,
        question: str,
        nodes: list[DocumentNode],
        cancelled: list[bool] | None = None,
    ) -> Iterator[str | Answer]:
        """Yield text fragments then the final Answer.

        Protocol: every ``str`` is a fragment to append to the UI bubble;
        the single ``Answer`` at the end carries the full text + citations.

        Raises:
            AnswerGenerationError: When the stream fails before producing
                                   any text.
        """
        evidence, references = _build_evidence_block(nodes)
        prompt = build_answer_prompt(question, evidence)

        accumulated: list[str] = []
        try:
            for fragment in self._client.generate_stream(prompt):
                if cancelled and cancelled[0]:
                    break
                accumulated.append(fragment)
                yield fragment
        except GenerationError as exc:
            if not accumulated:
                raise AnswerGenerationError(str(exc)) from exc
            logger.warning("OllamaAnswerer stream interrupted after %d chars: %s",
                           sum(len(f) for f in accumulated), exc)

        full_text = "".join(accumulated).strip()
        citations = _nodes_to_citations(full_text, references)
        if full_text and not self._validator.validate(full_text, citations):
            fallback = self._extractive.answer(question, nodes)
            yield fallback
            return

        yield Answer(
            question=question,
            answer=full_text or "Generation was interrupted before completion.",
            citations=citations,
            answerable=bool(full_text),
        )

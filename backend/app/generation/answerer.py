"""Answerer — handles logic for extracting or generating answers.

Coordinates the RAG response pipeline: Extractive Fallback -> LLM Generation -> Static Fallback.
"""
from __future__ import annotations

import re
from typing import Any

from nltk.tokenize import sent_tokenize

from backend.app.core.config import Settings
from backend.app.core.text import truncate_words
from backend.app.domain.exceptions import GenerationError
from backend.app.domain.models.node import DocumentNode
from backend.app.generation.extractive_answerer import ExtractiveAnswerer
from backend.app.generation.ollama_client import OllamaClient
from backend.app.generation.prompts import build_answer_prompt
from backend.app.models import Answer, Citation, Chunk
from backend.app.retrieval.context_builder import build_evidence_block
from backend.app.verification.citations import has_supported_citation

INSUFFICIENT_EVIDENCE = (
    "I could not find sufficient evidence in the uploaded documents to answer this question."
)

# Pre-compiled generic regex for performance
ALPHANUMERIC_PATTERN = re.compile(r"[a-z0-9]+")
# Generic stopwords to ignore when matching query to sentences
STOPWORDS = re.compile(
    r"\b(what|is|are|the|a|an|tell|me|about|of|in|how|why|when|where|can|you)\b"
)


class Answerer:
    """Handles extracting or generating answers to user questions based on provided document chunks.

    Attributes:
        settings: Application configuration settings.
        ollama:   The initialized Ollama client for generative synthesis.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ollama = OllamaClient(settings)
        self._extractive = ExtractiveAnswerer()

    def answer(
        self, question: str, chunks: list[Chunk], debug: dict[str, Any] | None = None
    ) -> Answer:
        """Derives an answer to the provided question based solely on the provided chunks.

        Attributes:
            question: The user's query.
            chunks:   List of document chunks retrieved from the knowledge base.
            debug:    Optional debug dictionary for tracing.
        """
        if not chunks:
            return Answer(
                question=question,
                answer=INSUFFICIENT_EVIDENCE,
                citations=[],
                answerable=False,
            )

        evidence, citations = build_evidence_block(chunks)
        debug_info = debug or {}

        # 1. Rich extractive matching (names, CGPA, institutions, degrees,
        #    phone numbers, and detailed topic passages) via ExtractiveAnswerer.
        pseudo_nodes = [self._chunk_to_pseudo_node(chunk) for chunk in chunks]
        extractive = self._extractive.answer(question, pseudo_nodes)
        if extractive.answerable:
            return Answer(
                question=question,
                answer=extractive.answer,
                citations=citations,
                answerable=True,
                debug=debug_info,
            )

        # 2. Fast-path generic extraction (Keyword overlap)
        if extractive_answer := self._generic_extractive_answer(question, chunks, citations):
            return Answer(
                question=question,
                answer=extractive_answer,
                citations=citations,
                answerable=True,
                debug=debug_info,
            )

        # 3. Ollama generation — for synthesis, reasoning, or complex queries
        if self.settings.use_ollama:
            try:
                generated = self.ollama.generate(build_answer_prompt(question, evidence))
                if generated and has_supported_citation(generated, citations):
                    return Answer(
                        question=question,
                        answer=generated,
                        citations=citations,
                        answerable=True,
                        debug=debug_info,
                    )
            except GenerationError:
                pass  # Fallback to extractive on generation failure

        # 4. Final fallback — always return the best extracted sentence from top chunk
        fallback = self._fallback_answer(chunks, citations)
        return Answer(
            question=question,
            answer=fallback,
            citations=citations,
            answerable=True,
            debug=debug_info,
        )

    @staticmethod
    def _chunk_to_pseudo_node(chunk: Chunk) -> DocumentNode:
        """Wrap a Chunk as a minimal DocumentNode so ExtractiveAnswerer (which
        operates on the newer node-based domain model) can be reused here
        without duplicating its extraction logic."""
        return DocumentNode(
            id=chunk.chunk_id,
            document_id=chunk.document_id,
            parent_id=None,
            node_type="paragraph",
            title=None,
            text=chunk.text,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            depth=0,
            position=0,
            heading_path=list(chunk.section_path),
        )

    def _generic_extractive_answer(
        self, question: str, chunks: list[Chunk], citations: list[Citation]
    ) -> str | None:
        """Attempts to find a highly relevant sentence by matching query keywords.

        Attributes:
            question:  The user's query.
            chunks:    Context chunks to search.
            citations: Corresponding citations for chunks.
        """
        cleaned_q = STOPWORDS.sub(" ", question.lower())
        query_terms = [t for t in ALPHANUMERIC_PATTERN.findall(cleaned_q) if len(t) > 2]

        if not query_terms:
            return None

        best_match: tuple[int, str, Citation] | None = None

        for citation, chunk in zip(citations, chunks):
            for sentence in sent_tokenize(chunk.text):
                sentence_terms = set(ALPHANUMERIC_PATTERN.findall(sentence.lower()))
                score = sum(1 for term in query_terms if term in sentence_terms)

                if score > 0 and (best_match is None or score > best_match[0]):
                    best_match = (score, sentence, citation)

        # Threshold: ensure at least 2 matches for long queries, 1 for short queries
        threshold = 2 if len(query_terms) > 1 else 1

        if best_match and best_match[0] >= threshold:
            _, sentence, citation = best_match
            sentence = truncate_words(sentence, 70).strip(" -")
            return f"{sentence} [{citation.source_id}]"

        return None

    def _fallback_answer(self, chunks: list[Chunk], citations: list[Citation]) -> str:
        """Final fallback: returns the first valid sentence from the top evidence chunk.

        Attributes:
            chunks:    Context chunks to search.
            citations: Corresponding citations for chunks.
        """
        if not chunks or not citations:
            return INSUFFICIENT_EVIDENCE

        top_chunk_text = chunks[0].text
        top_citation = citations[0]

        sentences = sent_tokenize(top_chunk_text)
        if sentences:
            return f"{truncate_words(sentences[0], 45)} [{top_citation.source_id}]"

        return f"{truncate_words(top_chunk_text, 45)} [{top_citation.source_id}]"
from __future__ import annotations

from backend.app.core.config import Settings
from backend.app.core.text import truncate_words
from backend.app.generation.ollama_client import GenerationError, OllamaClient
from backend.app.generation.prompts import build_answer_prompt
from backend.app.models import Answer, Citation, Chunk
from backend.app.retrieval.context_builder import build_evidence_block
from backend.app.verification.citations import has_supported_citation


INSUFFICIENT_EVIDENCE = (
    "I could not find sufficient evidence in the uploaded documents to answer this question."
)


class Answerer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ollama = OllamaClient(settings)

    def answer(self, question: str, chunks: list[Chunk], debug: dict[str, object] | None = None) -> Answer:
        if not chunks:
            return Answer(question=question, answer=INSUFFICIENT_EVIDENCE, citations=[], answerable=False)

        evidence, citations = build_evidence_block(chunks)
        if self.settings.use_ollama:
            try:
                generated = self.ollama.generate(build_answer_prompt(question, evidence))
                if generated and has_supported_citation(generated, citations):
                    return Answer(
                        question=question,
                        answer=generated,
                        citations=citations,
                        answerable=True,
                        debug=debug or {},
                    )
            except GenerationError:
                pass

        fallback = self._extractive_answer(chunks, citations)
        return Answer(
            question=question,
            answer=fallback,
            citations=citations,
            answerable=True,
            debug=debug or {},
        )

    def _extractive_answer(self, chunks: list[Chunk], citations: list[Citation]) -> str:
        parts: list[str] = []
        for citation, chunk in zip(citations[:3], chunks[:3]):
            parts.append(f"{truncate_words(chunk.text, 70)} [{citation.source_id}]")
        return "\n\n".join(parts) if parts else INSUFFICIENT_EVIDENCE

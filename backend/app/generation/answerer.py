from __future__ import annotations

import re

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
        fact_answer = self._extractive_fact_answer(question, chunks, citations)
        if fact_answer:
            return Answer(
                question=question,
                answer=fact_answer,
                citations=citations,
                answerable=True,
                debug=debug or {},
            )

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

    def _extractive_fact_answer(
        self,
        question: str,
        chunks: list[Chunk],
        citations: list[Citation],
    ) -> str | None:
        normalized = question.lower()
        if any(term in normalized for term in ["full name", "person name", "user name", "candidate name"]):
            return self._answer_name(question, chunks, citations)
        if "name" in normalized and any(term in normalized for term in ["resume", "cv", "person", "candidate"]):
            return self._answer_name(question, chunks, citations)
        if "cgpa" in normalized or "gpa" in normalized:
            return self._answer_pattern(
                chunks,
                citations,
                r"\bC?GPA\b\s*(?:[:=\-]|is|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                "The CGPA is {value}.",
            )
        if "email" in normalized or "mail" in normalized:
            return self._answer_pattern(
                chunks,
                citations,
                r"\b([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})\b",
                "The email address is {value}.",
                flags=re.IGNORECASE,
            )
        if "phone" in normalized or "mobile" in normalized or "contact" in normalized:
            return self._answer_pattern(
                chunks,
                citations,
                r"(\+?\d[\d\s().-]{7,}\d)",
                "The phone number is {value}.",
            )
        return None

    def _answer_name(self, question: str, chunks: list[Chunk], citations: list[Citation]) -> str | None:
        target_names = [
            token
            for token in re.findall(r"[a-z][a-z]+", question.lower())
            if token not in {"what", "whats", "name", "full", "person", "user", "candidate", "resume"}
        ]
        for citation, chunk in zip(citations, chunks):
            text = chunk.text
            for target in target_names:
                match = re.search(
                    rf"\b({re.escape(target.title())}\s+[A-Z][A-Za-z]+)\b",
                    text,
                    flags=re.IGNORECASE,
                )
                if match:
                    return f"The full name is {self._title_case_name(match.group(1))}. [{citation.source_id}]"

            label_match = re.search(
                r"(?:^|\n|\b)(?:name|full name|candidate name|student name)\s*[:\-]\s*"
                r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})",
                text,
            )
            if label_match:
                return f"The full name is {self._title_case_name(label_match.group(1))}. [{citation.source_id}]"

            first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
            line_match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", first_line)
            if line_match:
                return f"The full name is {self._title_case_name(line_match.group(1))}. [{citation.source_id}]"
        return None

    def _answer_pattern(
        self,
        chunks: list[Chunk],
        citations: list[Citation],
        pattern: str,
        template: str,
        flags: int = 0,
    ) -> str | None:
        for citation, chunk in zip(citations, chunks):
            match = re.search(pattern, chunk.text, flags=flags)
            if match:
                value = match.group(1).strip(" .,:;-")
                return f"{template.format(value=value)} [{citation.source_id}]"
        return None

    def _extractive_answer(self, chunks: list[Chunk], citations: list[Citation]) -> str:
        for citation, chunk in zip(citations, chunks):
            sentence = self._best_sentence(chunk.text, citation)
            if sentence:
                return sentence
        if citations and chunks:
            return f"{truncate_words(chunks[0].text, 45)} [{citations[0].source_id}]"
        return INSUFFICIENT_EVIDENCE

    def _best_sentence(self, text: str, citation: Citation) -> str | None:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
            if sentence.strip()
        ]
        if not sentences:
            return None
        return f"{truncate_words(sentences[0], 45)} [{citation.source_id}]"

    def _title_case_name(self, value: str) -> str:
        return " ".join(part[:1].upper() + part[1:].lower() for part in value.split())

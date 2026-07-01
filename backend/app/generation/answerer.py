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

        definition_answer = self._extractive_definition_answer(question, chunks, citations)
        if definition_answer:
            return Answer(
                question=question,
                answer=definition_answer,
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
        if any(
            term in normalized
            for term in ["full name", "person name", "user name", "candidate name", "student name"]
        ):
            return self._answer_name(question, chunks, citations)
        if "name" in normalized and any(term in normalized for term in ["resume", "cv", "person", "candidate"]):
            return self._answer_name(question, chunks, citations)
        if normalized.strip() in {"name", "what is name", "what is the name"}:
            return self._answer_name(question, chunks, citations)
        if "cgpa" in normalized or "gpa" in normalized:
            return self._answer_pattern(
                chunks,
                citations,
                r"\bC?GPA\b\s*(?:[:=\-]|is|of)?\s*([0-9]+(?:\.[0-9]+)?)",
                "The CGPA is {value}.",
            )
        if any(term in normalized for term in ["college", "collage", "university", "institute", "school"]):
            return self._answer_institution(chunks, citations)
        if any(term in normalized for term in ["degree", "course", "branch", "program"]):
            return self._answer_degree(chunks, citations)
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

    def _extractive_definition_answer(
        self,
        question: str,
        chunks: list[Chunk],
        citations: list[Citation],
    ) -> str | None:
        normalized = self._normalize_question(question)
        if not self._is_definition_query(normalized):
            return None

        query_terms = self._definition_terms(normalized)
        if not query_terms:
            return None

        detailed = self._is_detailed_query(normalized)
        phrase = self._definition_phrase(normalized)
        if detailed:
            passage = self._best_topic_passage(chunks, citations, query_terms, phrase)
            if passage:
                return passage

        best: tuple[int, str, Citation] | None = None
        for citation, chunk in zip(citations, chunks):
            sentences = self._sentences(chunk.text)
            if not sentences:
                continue
            for sentence in sentences:
                sentence_terms = set(re.findall(r"[a-z0-9]+", self._normalize_question(sentence)))
                score = sum(1 for term in query_terms if term in sentence_terms)
                if score and (best is None or score > best[0]):
                    best = (score, sentence, citation)

        if best is None:
            return None

        _score, sentence, citation = best
        sentence = truncate_words(sentence, 70).strip(" -")
        return f"{sentence} [{citation.source_id}]"

    def _is_definition_query(self, normalized_question: str) -> bool:
        return bool(
            re.search(
                r"\b(what is|what are|define|explain|describe|"
                r"tell me(?: everything| all)? about|everything about)\b",
                normalized_question,
            )
            or "in detail" in normalized_question
            or "details about" in normalized_question
        )

    def _is_detailed_query(self, normalized_question: str) -> bool:
        return any(
            term in normalized_question
            for term in ["everything", "all about", "in detail", "details about"]
        )

    def _best_topic_passage(
        self,
        chunks: list[Chunk],
        citations: list[Citation],
        query_terms: list[str],
        phrase: str,
    ) -> str | None:
        best: tuple[float, str, Citation] | None = None
        for citation, chunk in zip(citations, chunks):
            excerpt = self._topic_excerpt(chunk.text, phrase)
            if not excerpt:
                continue
            score = self._passage_score(excerpt, query_terms, phrase)
            if score and (best is None or score > best[0]):
                best = (score, excerpt, citation)

        if best is None:
            return None

        _score, excerpt, citation = best
        excerpt = truncate_words(excerpt, 170).strip(" -")
        return f"{excerpt} [{citation.source_id}]"

    def _topic_excerpt(self, text: str, phrase: str) -> str:
        normalized_text = self._normalize_question(text)
        position = normalized_text.find(phrase) if phrase else -1
        if position < 0:
            sentences = self._sentences(text)
            return " ".join(sentences[:4])

        word_starts = [match.start() for match in re.finditer(r"\S+", text)]
        start_word = 0
        for index, start in enumerate(word_starts):
            if start >= position:
                start_word = index
                break
        words = text.split()
        start_word = max(0, start_word - 4)
        return " ".join(words[start_word : start_word + 190])

    def _passage_score(self, passage: str, query_terms: list[str], phrase: str) -> float:
        normalized = self._normalize_question(passage)
        score = 0.0
        if phrase and phrase in normalized:
            score += 20.0
        passage_terms = set(re.findall(r"[a-z0-9]+", normalized))
        score += sum(2.0 for term in query_terms if term in passage_terms)
        if "transaction" in query_terms and "states" in query_terms:
            state_terms = ["active", "partially", "failed", "aborted", "committed", "terminated"]
            score += sum(3.0 for term in state_terms if term in normalized)
        return score

    def _answer_name(self, question: str, chunks: list[Chunk], citations: list[Citation]) -> str | None:
        target_names = [
            token
            for token in re.findall(r"[a-z][a-z]+", question.lower())
            if token
            not in {"what", "whats", "the", "is", "name", "full", "person", "user", "candidate", "student", "resume"}
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
            line_match = re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", first_line)
            if line_match:
                return f"The full name is {self._title_case_name(line_match.group(1))}. [{citation.source_id}]"
        return None

    def _answer_institution(self, chunks: list[Chunk], citations: list[Citation]) -> str | None:
        for citation, chunk in zip(citations, chunks):
            for segment in self._candidate_segments(chunk.text):
                institution = self._extract_institution(segment)
                if institution:
                    return f"The college name is {institution}. [{citation.source_id}]"
        return None

    def _answer_degree(self, chunks: list[Chunk], citations: list[Citation]) -> str | None:
        degree_pattern = (
            r"\b("
            r"B\.?\s?Tech|BTECH|Bachelor of Technology|"
            r"M\.?\s?Tech|MTECH|Master of Technology|"
            r"B\.?\s?E\.?|BCA|MCA|BSc|MSc|MBA"
            r")\b\s*(?:\(?\s*([A-Z][A-Z.&\s]{1,20})\s*\)?)?"
        )
        for citation, chunk in zip(citations, chunks):
            match = re.search(degree_pattern, chunk.text, flags=re.IGNORECASE)
            if not match:
                continue
            degree = match.group(1).replace(" ", "")
            branch = (match.group(2) or "").strip(" .,:;-()")
            value = self._normalize_degree(degree)
            if branch:
                value = f"{value} ({branch})"
            return f"The degree is {value}. [{citation.source_id}]"
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
        sentences = self._sentences(text)
        if not sentences:
            return None
        return f"{truncate_words(sentences[0], 45)} [{citation.source_id}]"

    def _title_case_name(self, value: str) -> str:
        return " ".join(part[:1].upper() + part[1:].lower() for part in value.split())

    def _candidate_segments(self, text: str) -> list[str]:
        segments = re.split(r"[\n|•]+", text)
        return [segment.strip() for segment in segments if segment.strip()] + [text]

    def _extract_institution(self, text: str) -> str | None:
        pattern = (
            r"\b([A-Z][A-Za-z.'-]*"
            r"(?:\s+(?:[A-Z][A-Za-z.'-]*|of|and|&)){0,12}"
            r"\s+(?:Institute|University|College|School)"
            r"(?:\s+(?:[A-Z][A-Za-z.'-]*|of|and|&)){0,12})\b"
        )
        match = re.search(pattern, text)
        if not match:
            return None
        value = match.group(1)
        value = re.split(
            r"\b(?:CGPA|GPA|BTECH|B\.Tech|MTECH|M\.Tech|Percentage|20\d{2}|19\d{2})\b",
            value,
        )[0]
        return " ".join(value.strip(" .,:;-()").split())

    def _normalize_degree(self, value: str) -> str:
        normalized = value.lower().replace(".", "").replace(" ", "")
        mapping = {
            "btech": "B.Tech",
            "bacheloroftechnology": "B.Tech",
            "mtech": "M.Tech",
            "masteroftechnology": "M.Tech",
            "be": "B.E.",
            "bca": "BCA",
            "mca": "MCA",
            "bsc": "BSc",
            "msc": "MSc",
            "mba": "MBA",
        }
        return mapping.get(normalized, value)

    def _normalize_question(self, value: str) -> str:
        normalized = value.lower()
        return normalized.replace("transection", "transaction").replace("collage", "college")

    def _definition_terms(self, normalized_question: str) -> list[str]:
        cleaned = re.sub(
            r"\b(what|is|are|the|a|an|define|explain|describe|tell|me|"
            r"everything|all|about|of|in|detail|details|please|also)\b",
            " ",
            normalized_question,
        )
        terms = [term for term in re.findall(r"[a-z0-9]+", cleaned) if len(term) > 2]
        if "transaction" in terms:
            terms.extend(["transactions", "transactional"])
        return terms

    def _definition_phrase(self, normalized_question: str) -> str:
        cleaned = re.sub(
            r"\b(what|is|are|the|a|an|define|explain|describe|tell|me|"
            r"everything|all|about|of|in|detail|details|please|also)\b",
            " ",
            normalized_question,
        )
        return " ".join(re.findall(r"[a-z0-9]+", cleaned))

    def _sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+|\n+|(?<=:)\s+", text)
        return [" ".join(part.split()) for part in parts if part.strip()]

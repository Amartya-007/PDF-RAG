"""ExtractiveAnswerer — instant, no-LLM answer extraction using NLP packages.

Selects the best sentence or passage directly from retrieved DocumentNodes
using spaCy for entity recognition/segmentation and TheFuzz for text overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

import spacy
import phonenumbers
from thefuzz import fuzz

from backend.app.core.text import truncate_words
from backend.app.domain.models.node import DocumentNode
from backend.app.models import Answer, Citation

# Load the spaCy English model globally so it only initializes once
nlp = spacy.load("en_core_web_sm")

_FAST_FACT_PATTERNS = [
    re.compile(r"\bwhat['s|s| ]+(is|are|was|were)\b", re.I),
    re.compile(r"\bdefine\b", re.I),
    re.compile(r"\bmeaning\s+of\b", re.I),
    re.compile(r"\bwhat\s+does\s+.*\s+mean\b", re.I),
    re.compile(r"\bwho\s+(is|was|were|are|did|wrote|directed|built)\b", re.I),
    re.compile(r"\bcreator\s+of\b", re.I),
    re.compile(r"\bwhen\s+(is|was|did|were|are|happened|occurred)\b", re.I),
    re.compile(r"\b(date|year)\s+of\b", re.I),
    re.compile(r"\bwhere\s+(is|was|are|were|located|situated|find)\b", re.I),
    re.compile(r"\bheadquarters\s+of\b", re.I),
    re.compile(r"\bhow\s+(many|much|tall|old|far|long|heavy|fast)\b", re.I),
    re.compile(r"\bpopulation\s+of\b", re.I),
    re.compile(r"\bage\s+of\b", re.I),
    re.compile(r"\b(list\s+of|name\s+all|give\s+me\s+a\s+list)\b", re.I),
    re.compile(r"\b(name|college|collage|institute|university|school|academy|campus|dept|department)\b", re.I),
]

# We keep specific regexes for highly formatted academic data where NER struggles
_DEGREE_RE = re.compile(
    r"\b(B\.?\s?Tech|BTECH|Bachelor of Technology|M\.?\s?Tech|MTECH|Master of Technology|B\.?\s?E\.?|BCA|MCA|BSc|MSc|MBA)\b\s*(?:\(?\s*([A-Z][A-Z.&\s]{1,20})\s*\)?)?",
    re.I
)
_CGPA_RE = re.compile(r"\bC?GPA\b\s*(?:[:=\-]|is|of)?\s*([0-9]+(?:\.[0-9]+)?)", re.I)
_EMAIL_RE = re.compile(r"\b([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})\b", re.I)

_DEFINITION_QUERY_RE = re.compile(
    r"\b(what is|what are|define|explain|describe|tell me(?: everything| all)? about|everything about)\b"
)

_INSUFFICIENT = "I could not find sufficient evidence in the documents to answer this question."


@dataclass(slots=True)
class _SentenceCandidate:
    text: str
    score: float
    node: DocumentNode
    sentence_index: int


class ExtractiveAnswerer:
    def __init__(self, min_sentence_words: int = 5) -> None:
        self._min_words = min_sentence_words

    def is_fast_fact_question(self, question: str) -> bool:
        if len(question.split()) > 15:
            return False
        return any(p.search(question) for p in _FAST_FACT_PATTERNS)

    def answer(self, question: str, nodes: list[DocumentNode]) -> Answer:
        if not nodes:
            return self._no_evidence(question)

        citations = [
            self._node_to_citation(node, f"S{index}")
            for index, node in enumerate(nodes, start=1)
        ]
        
        fact_answer = self._extractive_fact_answer(question, nodes, citations)
        if fact_answer:
            return self._answer_with_selected_citation(question, fact_answer, citations)

        definition_answer = self._extractive_definition_answer(question, nodes, citations)
        if definition_answer:
            return self._answer_with_selected_citation(question, definition_answer, citations)

        best = self._find_best_sentence(question, nodes)

        # Adjusted score threshold for fuzzy matching scale (0.0 to 1.0)
        if best is None or best.score < 0.40:
            return self._no_evidence(question)

        citation = self._node_to_citation(best.node, "S1")
        return Answer(
            question=question,
            answer=self._with_source_marker(best.text, citation.source_id),
            citations=[citation],
            answerable=True,
        )

    def _find_best_sentence(self, question: str, nodes: list[DocumentNode]) -> _SentenceCandidate | None:
        best: _SentenceCandidate | None = None

        for node in nodes:
            # TheFuzz handles tokenization, lowercasing, and overlap automatically
            heading_match_score = fuzz.token_set_ratio(question, node.title or "") / 100.0
            
            # spaCy replaces manual regex sentence splitting
            doc = nlp(node.text)
            sentences = [sent.text.strip() for sent in doc.sents]

            for idx, sent in enumerate(sentences):
                if sent.count(' ') + 1 < self._min_words:
                    continue
                
                base_score = fuzz.token_set_ratio(question, sent) / 100.0
                score = base_score + (0.20 if idx == 0 else 0.0) + (0.15 * heading_match_score)
                
                if best is None or score > best.score:
                    best = _SentenceCandidate(
                        text=sent,
                        score=score,
                        node=node,
                        sentence_index=idx,
                    )
        return best

    def _extractive_fact_answer(self, question: str, nodes: list[DocumentNode], citations: list[Citation]) -> str | None:
        normalized = question.lower()
        words_set = set(normalized.split())

        # Check for Names using spaCy NER
        if "name" in words_set or any(term in words_set for term in ["resume", "cv", "person", "candidate"]):
            return self._answer_name(nodes, citations)

        # Check for Grades
        if "cgpa" in words_set or "gpa" in words_set:
            return self._answer_pattern(nodes, citations, _CGPA_RE, "The CGPA is {value}.")

        # Check for Institutions using spaCy NER
        if words_set & {"college", "collage", "university", "institute", "school"}:
            return self._answer_institution(nodes, citations)

        # Check for Degrees
        if words_set & {"degree", "course", "branch", "program"}:
            return self._answer_degree(nodes, citations)

        # Check for Contacts
        if "email" in words_set or "mail" in words_set:
            return self._answer_pattern(nodes, citations, _EMAIL_RE, "The email address is {value}.")
            
        # Use phonenumbers package instead of custom Regex
        if words_set & {"phone", "mobile", "contact", "number"}:
            return self._answer_phone(nodes, citations)

        return None

    def _answer_name(self, nodes: list[DocumentNode], citations: list[Citation]) -> str | None:
        # Replaces massive custom regex list. spaCy finds PERSON entities natively.
        for citation, node in zip(citations, nodes):
            doc = nlp(node.text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # Clean up common NER trailing characters
                    clean_name = ent.text.strip(" .\n:")
                    if len(clean_name.split()) >= 2: # Prefer full names
                        return f"The name is {clean_name}. [{citation.source_id}]"
        return None

    def _answer_institution(self, nodes: list[DocumentNode], citations: list[Citation]) -> str | None:
        # Replaces complex _INSTITUTION_RE. spaCy finds ORG entities.
        valid_suffixes = {"University", "College", "Institute", "School", "Academy"}
        for citation, node in zip(citations, nodes):
            doc = nlp(node.text)
            for ent in doc.ents:
                if ent.label_ == "ORG" and any(suffix in ent.text for suffix in valid_suffixes):
                    clean_org = ent.text.replace('\n', ' ').strip(" .,:;")
                    return f"The institution name is {clean_org}. [{citation.source_id}]"
        return None
        
    def _answer_phone(self, nodes: list[DocumentNode], citations: list[Citation]) -> str | None:
        # Uses the phonenumbers package for robust international parsing
        # "IN" is set as the default region hint, falling back to international parsing
        for citation, node in zip(citations, nodes):
            for match in phonenumbers.PhoneNumberMatcher(node.text, "IN"):
                formatted = phonenumbers.format_number(
                    match.number, phonenumbers.PhoneNumberFormat.INTERNATIONAL
                )
                return f"The phone number is {formatted}. [{citation.source_id}]"
        return None

    def _extractive_definition_answer(self, question: str, nodes: list[DocumentNode], citations: list[Citation]) -> str | None:
        normalized = question.lower()
        if not _DEFINITION_QUERY_RE.search(normalized) and "in detail" not in normalized and "details about" not in normalized:
            return None

        detailed = any(term in normalized for term in ["everything", "all about", "in detail", "details about"])
        
        # Strip query words to find the core topic phrase
        core_topic = re.sub(r"\b(what|is|are|the|a|an|define|explain|describe|tell|me|everything|all|about|of|in|detail|details|please|also)\b", "", normalized).strip()

        best: tuple[float, str, Citation] | None = None
        for citation, node in zip(citations, nodes):
            doc = nlp(node.text)
            sentences = [sent.text.strip() for sent in doc.sents]
            
            for sentence in sentences:
                # TheFuzz replaces custom token overlap scoring
                score = fuzz.token_set_ratio(core_topic, sentence)
                if score > 50 and (best is None or score > best[0]):
                    best = (score, sentence, citation)

        if best is not None:
            _, sentence, citation = best
            limit = 170 if detailed else 70
            sentence = truncate_words(sentence, limit).strip(" -")
            return f"{sentence} [{citation.source_id}]"

        if nodes and citations:
            excerpt = truncate_words(nodes[0].text, 60).strip(" -")
            return f"{excerpt} [{citations[0].source_id}]"

        return None

    def _answer_degree(self, nodes: list[DocumentNode], citations: list[Citation]) -> str | None:
        for citation, node in zip(citations, nodes):
            match = _DEGREE_RE.search(node.text)
            if not match:
                continue
            degree = match.group(1).replace(" ", "")
            branch = (match.group(2) or "").strip(" .,:;-()")
            value = self._normalize_degree(degree)
            if branch:
                value = f"{value} ({branch})"
            return f"The degree is {value}. [{citation.source_id}]"
        return None

    def _answer_pattern(self, nodes: list[DocumentNode], citations: list[Citation], pattern: re.Pattern, template: str) -> str | None:
        for citation, node in zip(citations, nodes):
            match = pattern.search(node.text)
            if match:
                value = match.group(1).strip(" .,:;-")
                return f"{template.format(value=value)} [{citation.source_id}]"
        return None

    @staticmethod
    def _node_to_citation(node: DocumentNode, source_id: str) -> Citation:
        return Citation(
            source_id=source_id,
            document_id=node.document_id,
            chunk_id=node.id,
            filename=node.document_id,
            page_start=node.page_start,
            page_end=node.page_end or node.page_start,
            excerpt=node.text[:300],
        )

    @staticmethod
    def _with_source_marker(text: str, source_id: str) -> str:
        marker = f"[{source_id}]"
        return text if marker in text else f"{text} {marker}"

    @staticmethod
    def _answer_with_selected_citation(question: str, answer_text: str, citations: list[Citation]) -> Answer:
        source_match = re.search(r"\[(S\d+)\]", answer_text)
        if source_match:
            selected_source = source_match.group(1)
            selected = next((c for c in citations if c.source_id == selected_source), citations[0])
            answer_text = re.sub(r"\[S\d+\]", "[S1]", answer_text)
        else:
            selected = citations[0]
            answer_text = f"{answer_text} [S1]"

        return Answer(
            question=question,
            answer=answer_text,
            citations=[replace(selected, source_id="S1")],
            answerable=True,
        )

    @staticmethod
    def _normalize_degree(value: str) -> str:
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

    @staticmethod
    def _no_evidence(question: str) -> Answer:
        return Answer(question=question, answer=_INSUFFICIENT, citations=[], answerable=False)
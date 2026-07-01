from __future__ import annotations

import os
import unittest

from backend.app.core.config import get_settings
from backend.app.generation.answerer import Answerer
from backend.app.models import Chunk


class AnswererTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["RAG_USE_OLLAMA"] = "0"

    def test_full_name_question_returns_short_fact(self) -> None:
        answerer = Answerer(get_settings())
        answer = answerer.answer(
            "what is amartya full name?",
            [
                Chunk(
                    chunk_id="chunk_1",
                    document_id="doc_1",
                    filename="resume.pdf",
                    page_start=1,
                    page_end=1,
                    section_path=(),
                    text=(
                        "Amartya Vishwakarma Adhartal, Jabalpur, MP | +91 9329251802 "
                        "Professional Summary .NET backend developer."
                    ),
                )
            ],
        )

        self.assertEqual(answer.answer, "The full name is Amartya Vishwakarma. [S1]")
        self.assertLess(len(answer.answer.split()), 10)

    def test_student_name_question_returns_name(self) -> None:
        answerer = Answerer(get_settings())
        answer = answerer.answer(
            "whats student name",
            [
                Chunk(
                    chunk_id="chunk_1",
                    document_id="doc_1",
                    filename="resume.pdf",
                    page_start=1,
                    page_end=1,
                    section_path=(),
                    text=(
                        "Amartya Vishwakarma\n"
                        "Education Coursework Skills Projects Developed an AI-driven web app."
                    ),
                )
            ],
        )

        self.assertEqual(answer.answer, "The full name is Amartya Vishwakarma. [S1]")

    def test_cgpa_question_returns_short_fact(self) -> None:
        answerer = Answerer(get_settings())
        answer = answerer.answer(
            "what is his cgpa in college?",
            [
                Chunk(
                    chunk_id="chunk_1",
                    document_id="doc_1",
                    filename="resume.pdf",
                    page_start=1,
                    page_end=1,
                    section_path=(),
                    text="Education: BTECH CSE, CGPA - 8.13, 2021-2025.",
                )
            ],
        )

        self.assertEqual(answer.answer, "The CGPA is 8.13. [S1]")

    def test_college_question_returns_short_fact(self) -> None:
        answerer = Answerer(get_settings())
        answer = answerer.answer(
            "whats amartya collage name?",
            [
                Chunk(
                    chunk_id="chunk_1",
                    document_id="doc_1",
                    filename="resume.pdf",
                    page_start=1,
                    page_end=1,
                    section_path=(),
                    text=(
                        "EDUCATION Amartya Vishwakarma BTECH(CSE), CGPA: 8.09 "
                        "2021 - 2025 Shri Ram Institute of Science & Technology Jabalpur, MP"
                    ),
                )
            ],
        )

        self.assertIn("Shri Ram Institute of Science & Technology", answer.answer)
        self.assertIn("[S1]", answer.answer)

    def test_definition_question_returns_extract_without_ollama(self) -> None:
        answerer = Answerer(get_settings())
        answer = answerer.answer(
            "what is transection?",
            [
                Chunk(
                    chunk_id="chunk_1",
                    document_id="doc_1",
                    filename="dbms.pdf",
                    page_start=1,
                    page_end=1,
                    section_path=(),
                    text=(
                        "A transaction is a logical unit of database work. "
                        "Transactions must preserve consistency and isolation."
                    ),
                )
            ],
        )

        self.assertIn("transaction is a logical unit", answer.answer.lower())
        self.assertIn("[S1]", answer.answer)

    def test_detailed_topic_question_returns_focused_passage_without_ollama(self) -> None:
        answerer = Answerer(get_settings())
        answer = answerer.answer(
            "tell me everything about Transaction States.",
            [
                Chunk(
                    chunk_id="chunk_1",
                    document_id="doc_1",
                    filename="dbms.pdf",
                    page_start=4,
                    page_end=4,
                    section_path=(),
                    text=(
                        "Transaction States\n"
                        "There are the following six states in which a transaction may exist: "
                        "Active: The initial state when the transaction has just started execution. "
                        "Partially Committed: The transaction is going towards its commit point. "
                        "Failed: The transaction fails for some reason. "
                        "Aborted: The rollback operation is over. "
                        "Committed: No failure occurs and the transaction reaches the commit point. "
                        "Terminated: Either committed or aborted, the transaction finally reaches this state."
                    ),
                )
            ],
        )

        self.assertIn("Transaction States", answer.answer)
        self.assertIn("Active", answer.answer)
        self.assertIn("Partially Committed", answer.answer)
        self.assertIn("Failed", answer.answer)
        self.assertIn("Aborted", answer.answer)
        self.assertIn("Committed", answer.answer)
        self.assertIn("Terminated", answer.answer)
        self.assertIn("[S1]", answer.answer)


if __name__ == "__main__":
    unittest.main()

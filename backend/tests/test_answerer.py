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


if __name__ == "__main__":
    unittest.main()

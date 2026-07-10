from __future__ import annotations

from backend.app.domain.models.node import DocumentNode
from backend.app.generation.extractive_answerer import ExtractiveAnswerer


def make_node(text: str, title: str | None = None) -> DocumentNode:
    return DocumentNode(
        id="node1",
        document_id="doc1",
        parent_id=None,
        node_type="paragraph",
        title=title,
        text=text,
        page_start=1,
        page_end=1,
        depth=1,
        position=0,
        heading_path=[],
    )


def test_cgpa_question_returns_short_structured_fact_from_document_node() -> None:
    answer = ExtractiveAnswerer().answer(
        "what is his cgpa in college?",
        [make_node("Education: BTECH CSE, CGPA - 8.13, 2021-2025.")],
    )

    assert answer.answer == "The CGPA is 8.13. [S1]"
    assert answer.citations[0].chunk_id == "node1"


def test_college_question_extracts_institution_name_from_document_node() -> None:
    answer = ExtractiveAnswerer().answer(
        "whats amartya collage name?",
        [
            make_node(
                "EDUCATION Amartya Vishwakarma BTECH(CSE), CGPA: 8.09 "
                "2021 - 2025 Shri Ram Institute of Science & Technology Jabalpur, MP"
            )
        ],
    )

    assert "Shri Ram Institute of Science & Technology" in answer.answer
    assert "[S1]" in answer.answer


def test_detailed_topic_question_returns_focused_passage_from_document_node() -> None:
    answer = ExtractiveAnswerer().answer(
        "tell me everything about Transaction States.",
        [
            make_node(
                "Transaction States\n"
                "There are the following six states in which a transaction may exist: "
                "Active: The initial state when the transaction has just started execution. "
                "Partially Committed: The transaction is going towards its commit point. "
                "Failed: The transaction fails for some reason. "
                "Aborted: The rollback operation is over. "
                "Committed: No failure occurs and the transaction reaches the commit point. "
                "Terminated: Either committed or aborted, the transaction finally reaches this state."
            )
        ],
    )

    assert "Transaction States" in answer.answer
    assert "Active" in answer.answer
    assert "Partially Committed" in answer.answer
    assert "Failed" in answer.answer
    assert "Aborted" in answer.answer
    assert "Committed" in answer.answer
    assert "Terminated" in answer.answer
    assert "[S1]" in answer.answer

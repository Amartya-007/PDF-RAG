from __future__ import annotations

from backend.app.domain.models.node import DocumentNode
from backend.app.generation.extractive_answerer import ExtractiveAnswerer
from backend.app.generation.ollama_answerer import OllamaAnswerer


class FakeOllamaClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.text

    def generate_stream(self, prompt: str):
        self.prompts.append(prompt)
        for part in self.text.split(" "):
            yield part + " "


def make_node(text: str) -> DocumentNode:
    return DocumentNode(
        id="node1",
        document_id="doc1",
        parent_id=None,
        node_type="paragraph",
        title="Revenue",
        text=text,
        page_start=3,
        page_end=3,
        depth=1,
        position=0,
        heading_path=["Finance"],
    )


def test_answer_accepts_generated_text_only_when_citation_validates() -> None:
    answerer = OllamaAnswerer(
        FakeOllamaClient("Revenue increased by 12% in 2024 [S1]."),
        extractive=ExtractiveAnswerer(),
    )

    answer = answerer.answer(
        "summarize revenue",
        [make_node("Revenue increased by 12% in 2024 after refunds settled.")],
    )

    assert answer.answer == "Revenue increased by 12% in 2024 [S1]."
    assert answer.citations[0].source_id == "S1"
    assert answer.citations[0].document_id == "doc1"
    assert answer.citations[0].chunk_id == "node1"


def test_answer_falls_back_to_extractive_when_validation_fails() -> None:
    answerer = OllamaAnswerer(
        FakeOllamaClient("Revenue increased by 99% in 2024 [S1]."),
        extractive=ExtractiveAnswerer(),
    )

    answer = answerer.answer(
        "summarize revenue",
        [make_node("Revenue increased by 12% in 2024 after refunds settled.")],
    )

    assert answer.answer != "Revenue increased by 99% in 2024 [S1]."
    assert "Revenue increased by 12% in 2024" in answer.answer
    assert answer.answerable is True

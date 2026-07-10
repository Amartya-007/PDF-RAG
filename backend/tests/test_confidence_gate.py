from __future__ import annotations

from backend.app.domain.enums import QueryType
from backend.app.domain.models.node import DocumentNode
from backend.app.retrieval.confidence import AnswerStrategy, ConfidenceGate


def make_node(node_id: str = "node1") -> DocumentNode:
    return DocumentNode(
        id=node_id,
        document_id="doc1",
        parent_id=None,
        node_type="paragraph",
        title="Policy",
        text="Revenue includes settled payments and refunds.",
        page_start=1,
        page_end=1,
        depth=1,
        position=0,
        heading_path=[],
    )


def test_evaluate_use_ollama_false_always_extracts() -> None:
    gate = ConfidenceGate(minimum_score=0.4, extractive_threshold=0.7)

    strategy = gate.evaluate(
        QueryType.COMPARISON,
        [],
        {},
        use_ollama=False,
    )

    assert strategy == AnswerStrategy.EXTRACTIVE


def test_evaluate_low_score_is_insufficient_when_ollama_enabled() -> None:
    gate = ConfidenceGate(minimum_score=0.4, extractive_threshold=0.7)

    strategy = gate.evaluate(
        QueryType.FAST_FACT,
        [make_node()],
        {"node1": {"score": 0.39}},
        use_ollama=True,
    )

    assert strategy == AnswerStrategy.INSUFFICIENT


def test_evaluate_comparison_and_summary_generate_when_supported() -> None:
    gate = ConfidenceGate(minimum_score=0.4, extractive_threshold=0.7)

    for query_type in [QueryType.COMPARISON, QueryType.SUMMARY]:
        assert gate.evaluate(
            query_type,
            [make_node()],
            {"node1": {"score": 0.8}},
            use_ollama=True,
        ) == AnswerStrategy.GENERATE


def test_evaluate_fast_fact_topic_threshold_boundary() -> None:
    gate = ConfidenceGate(minimum_score=0.4, extractive_threshold=0.7)

    assert gate.evaluate(
        QueryType.FAST_FACT,
        [make_node()],
        {"node1": {"score": 0.71}},
        use_ollama=True,
    ) == AnswerStrategy.EXTRACTIVE
    assert gate.evaluate(
        QueryType.TOPIC,
        [make_node()],
        {"node1": {"score": 0.7}},
        use_ollama=True,
    ) == AnswerStrategy.GENERATE


def test_evaluate_is_deterministic() -> None:
    gate = ConfidenceGate(minimum_score=0.4, extractive_threshold=0.7)
    args = (
        QueryType.TOPIC,
        [make_node()],
        {"node1": {"score": 0.9}},
        True,
    )

    assert gate.evaluate(*args) == gate.evaluate(*args)

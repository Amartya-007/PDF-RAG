from __future__ import annotations

from backend.app.domain.models.node import DocumentNode
from backend.app.retrieval.node_ranker import NodeRanker


def make_node(
    node_id: str,
    text: str,
    *,
    title: str | None = None,
    page_start: int = 1,
) -> DocumentNode:
    return DocumentNode(
        id=node_id,
        document_id="doc1",
        parent_id=None,
        node_type="paragraph",
        title=title,
        text=text,
        page_start=page_start,
        page_end=page_start,
        depth=1,
        position=0,
        heading_path=[],
    )


def test_rank_fast_fact_prioritizes_field_answer_and_records_details() -> None:
    institute = make_node(
        "institute",
        "Amartya Vishwakarma BTECH CSE Shri Ram Institute of Science and Technology.",
    )
    project = make_node(
        "project",
        "Education coursework skills projects include a web application.",
    )
    ranker = NodeRanker()

    ranked = ranker.rank_fast_fact(
        "whats amartya college name?",
        [project, institute],
        candidate_ids={"institute"},
    )

    assert [node.id for node in ranked] == ["institute", "project"]
    details = ranker.score_details("institute")
    assert details["fast_fact_score"] > details["topic_score"]
    assert details["retrieval_hit"] == 1.0


def test_rank_topic_prioritizes_exact_heading_and_records_details() -> None:
    log = make_node(
        "log",
        "Log records contain old values and new values.",
        title="Log Records",
        page_start=10,
    )
    states = make_node(
        "states",
        "Active, Partially Committed, Failed, Aborted, Committed, and Terminated.",
        title="Transaction States",
        page_start=2,
    )
    ranker = NodeRanker()

    ranked = ranker.rank_topic(
        "tell me everything about Transaction States",
        [log, states],
        candidate_ids={"log", "states"},
    )

    assert [node.id for node in ranked] == ["states", "log"]
    details = ranker.score_details("states")
    assert details["topic_score"] > details["fast_fact_score"]
    assert details["heading_score"] == 1.0

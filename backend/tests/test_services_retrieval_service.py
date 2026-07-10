from __future__ import annotations

from backend.app.domain.enums import QueryType
from backend.app.domain.models.node import DocumentNode
from backend.app.services.retrieval_service import RetrievalService


def make_node(
    node_id: str,
    text: str,
    *,
    title: str | None = None,
    parent_id: str | None = None,
    node_type: str = "paragraph",
    position: int = 0,
) -> DocumentNode:
    return DocumentNode(
        id=node_id,
        document_id="doc1",
        parent_id=parent_id,
        node_type=node_type,
        title=title,
        text=text,
        page_start=1,
        page_end=1,
        depth=1,
        position=position,
        heading_path=[],
    )


class FakeNodeRepository:
    def __init__(self, nodes: list[DocumentNode]) -> None:
        self._nodes = {node.id: node for node in nodes}

    def get_many(self, node_ids: list[str]) -> list[DocumentNode]:
        return [self._nodes[node_id] for node_id in node_ids if node_id in self._nodes]

    def list_nodes_for_session(self, session_id: str) -> list[DocumentNode]:
        return list(self._nodes.values())

    def list_nodes_for_document(self, document_id: str) -> list[DocumentNode]:
        return [
            node for node in self._nodes.values() if node.document_id == document_id
        ]


class FakeLexicalRetriever:
    def __init__(self, results: dict[str, list[tuple[str, float]]]) -> None:
        self.results = results
        self.calls: list[str] = []

    def search(self, query: str, **kwargs) -> list[tuple[str, float]]:
        self.calls.append(query)
        return self.results.get(query, [])

    def score_breakdown(self, node_id: str) -> dict[str, float]:
        return {"fused_score": 0.7, "keyword_coverage": 1.0}


class FakeTreeNavigator:
    def __init__(self, expanded: list[DocumentNode] | None = None) -> None:
        self.expanded = expanded or []
        self.calls: list[list[str]] = []

    def expand(self, matched_nodes, node_repo, expand_depth=2, include_siblings=True):
        self.calls.append([node.id for node in matched_nodes])
        return self.expanded or list(matched_nodes)


class FakeNodeRanker:
    def __init__(self) -> None:
        self.fast_fact_calls = 0
        self.topic_calls = 0

    def rank_fast_fact(self, question, nodes, candidate_ids):
        self.fast_fact_calls += 1
        return list(reversed(nodes))

    def rank_topic(self, question, nodes, candidate_ids):
        self.topic_calls += 1
        return nodes

    def rank(self, question, candidates, tree_selected_ids=None, top_k=None):
        return candidates[:top_k]

    def score_details(self, node_id: str) -> dict[str, float]:
        return {"score": 0.8}


def test_retrieve_fast_fact_uses_specialized_ranker_and_debug() -> None:
    first = make_node("first", "University: Example State")
    second = make_node("second", "College: Vectorless Institute")
    ranker = FakeNodeRanker()
    service = RetrievalService(
        node_repo=FakeNodeRepository([first, second]),
        lexical=FakeLexicalRetriever(
            {"what is the college": [("first", 0.4), ("second", 0.9)]}
        ),
        navigator=FakeTreeNavigator(),
        ranker=ranker,
    )

    nodes, debug = service.retrieve(
        "what is the college",
        session_id="s1",
        top_k=1,
        include_debug=True,
    )

    assert [node.id for node in nodes] == ["second"]
    assert ranker.fast_fact_calls == 1
    assert debug["query_type"] == QueryType.FAST_FACT.value
    assert debug["selected_node_ids"] == ["second"]
    assert debug["score_breakdown"]["second"]["score"] == 0.8


def test_retrieve_comparison_searches_each_entity_independently() -> None:
    revenue = make_node("revenue", "Revenue is settled income.")
    refunds = make_node("refunds", "Refunds reverse settled income.")
    lexical = FakeLexicalRetriever(
        {
            "revenue": [("revenue", 1.0)],
            "refunds": [("refunds", 1.0)],
        }
    )
    service = RetrievalService(
        node_repo=FakeNodeRepository([revenue, refunds]),
        lexical=lexical,
        navigator=FakeTreeNavigator(),
        ranker=FakeNodeRanker(),
    )

    nodes, debug = service.retrieve(
        "compare revenue vs refunds",
        session_id="s1",
        top_k=10,
        include_debug=True,
    )

    assert lexical.calls == ["revenue", "refunds"]
    assert [node.id for node in nodes] == ["revenue", "refunds"]
    assert debug["query_type"] == QueryType.COMPARISON.value

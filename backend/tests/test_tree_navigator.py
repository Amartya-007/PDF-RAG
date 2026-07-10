from __future__ import annotations

from backend.app.domain.models.node import DocumentNode
from backend.app.retrieval.tree_navigator import TreeNavigator


class FakeNodeRepository:
    def __init__(self, nodes: list[DocumentNode]) -> None:
        self._nodes = nodes

    def list_nodes_for_document(self, document_id: str) -> list[DocumentNode]:
        return [node for node in self._nodes if node.document_id == document_id]


def make_node(
    node_id: str,
    *,
    parent_id: str | None,
    node_type: str,
    title: str | None = None,
    text: str | None = None,
    depth: int = 0,
    position: int = 0,
) -> DocumentNode:
    return DocumentNode(
        id=node_id,
        document_id="doc1",
        parent_id=parent_id,
        node_type=node_type,
        title=title,
        text=text or title or node_id,
        page_start=1,
        page_end=1,
        depth=depth,
        position=position,
        heading_path=[],
    )


def test_expand_adds_parent_section_and_adjacent_siblings() -> None:
    section = make_node("section", parent_id="root", node_type="section", title="Policy", depth=1)
    before = make_node("before", parent_id="section", node_type="paragraph", depth=2, position=0)
    matched = make_node("matched", parent_id="section", node_type="paragraph", depth=2, position=1)
    after = make_node("after", parent_id="section", node_type="paragraph", depth=2, position=2)
    repo = FakeNodeRepository(
        [
            make_node("root", parent_id=None, node_type="document", title="Document"),
            section,
            before,
            matched,
            after,
        ]
    )

    expanded = TreeNavigator().expand([matched], repo, expand_depth=1, include_siblings=True)

    assert [node.id for node in expanded] == ["matched", "section", "before", "after"]


def test_expand_adds_descendants_without_duplicates() -> None:
    section = make_node("section", parent_id="root", node_type="section", title="Policy", depth=1)
    child = make_node("child", parent_id="section", node_type="paragraph", depth=2, position=0)
    grandchild = make_node("grandchild", parent_id="child", node_type="paragraph", depth=3, position=0)
    repo = FakeNodeRepository(
        [
            make_node("root", parent_id=None, node_type="document", title="Document"),
            section,
            child,
            grandchild,
        ]
    )

    expanded = TreeNavigator().expand([section, child], repo, expand_depth=2, include_siblings=False)

    assert [node.id for node in expanded] == ["section", "child", "grandchild"]

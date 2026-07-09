from __future__ import annotations

from backend.app.database.store import MetadataStore
from backend.app.domain.models.node import DocumentNode
from backend.app.indexing.full_text_index import FTS5Index
from backend.app.models import Document


def make_store() -> MetadataStore:
    store = MetadataStore(":memory:")
    store.init()
    store.create_session("s1", "Session One")
    store.upsert_document(
        Document(
            document_id="doc1",
            filename="policy.pdf",
            sha256="sha",
            path="/tmp/policy.pdf",
            status="ready",
            session_id="s1",
        )
    )
    return store


def make_node(node_id: str, text: str, title: str | None = None) -> DocumentNode:
    return DocumentNode(
        id=node_id,
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


def test_fts5_index_upsert_searches_node_text_and_title() -> None:
    store = make_store()
    node = make_node(
        "node1",
        "Employees may carry forward thirty days of earned leave.",
        title="Leave Policy",
    )
    store.nodes.upsert_node(node)
    index = FTS5Index(store.connect)

    index.upsert(node)

    assert index.search("earned leave", top_k=10)[0][0] == "node1"
    assert index.search("policy", top_k=10)[0][0] == "node1"
    assert index.is_consistent()


def test_fts5_index_delete_removes_node_from_search_results() -> None:
    store = make_store()
    node = make_node("node1", "Alpha beta gamma.", title="Greek Terms")
    store.nodes.upsert_node(node)
    index = FTS5Index(store.connect)
    index.upsert(node)

    index.delete("node1")

    assert index.search("gamma", top_k=10) == []


def test_fts5_index_uses_like_fallback_when_inconsistent() -> None:
    store = make_store()
    node = make_node("node1", "Fallback keyword lives in the node table.")
    store.nodes.upsert_node(node)
    index = FTS5Index(store.connect)
    index._consistent = False

    assert index.search("keyword", top_k=10) == [("node1", 1.0)]


def test_fts5_index_rebuild_replaces_existing_index_contents() -> None:
    store = make_store()
    old_node = make_node("node1", "Obsolete keyword remains in storage.")
    new_node = make_node("node2", "Current keyword is searchable.")
    store.nodes.upsert_node(old_node)
    store.nodes.upsert_node(new_node)
    index = FTS5Index(store.connect)
    index.upsert(old_node)

    index.rebuild([new_node])

    assert index.search("obsolete", top_k=10) == []
    assert index.search("current", top_k=10)[0][0] == "node2"
    assert index.is_consistent()

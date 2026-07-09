from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from backend.app.database.store import MetadataStore
from backend.app.domain.models.node import DocumentNode
from backend.app.indexing.full_text_index import FTS5Index
from backend.app.indexing.heading_index import HeadingIndex
from backend.app.indexing.index_manager import IndexManager
from backend.app.indexing.metadata_index import MetadataIndex
from backend.app.indexing.phrase_index import PhraseIndex
from backend.app.indexing.sparse import BM25Index
from backend.app.models import Document


def make_store() -> MetadataStore:
    store = MetadataStore(":memory:")
    store.init()
    store.create_session("s1", "Session One")
    store.upsert_document(
        Document(
            document_id="doc1",
            filename="source.pdf",
            sha256="sha1",
            path="/tmp/source.pdf",
            status="ready",
            session_id="s1",
        )
    )
    return store


def make_node(node_id: str, document_id: str = "doc1", title: str | None = None) -> DocumentNode:
    return DocumentNode(
        id=node_id,
        document_id=document_id,
        parent_id=None,
        node_type="paragraph",
        title=title,
        text=f"{node_id} transaction state keyword",
        page_start=1,
        page_end=1,
        depth=1,
        position=0,
        heading_path=[],
    )


def make_manager(store: MetadataStore) -> IndexManager:
    index_path = Path.cwd() / "backend" / ".test-tmp" / f"bm25-{uuid4().hex}.json"
    return IndexManager(
        FTS5Index(store.connect),
        HeadingIndex(),
        PhraseIndex(),
        BM25Index(index_path),
        MetadataIndex(),
        store.nodes,
    )


def test_index_manager_add_and_remove_document_updates_all_indexes() -> None:
    store = make_store()
    node = make_node("node1", title="Transaction States")
    store.nodes.upsert_node(node)
    manager = make_manager(store)

    manager.add_document_nodes([node])

    assert manager.fts5.search("keyword", 10)[0][0] == "node1"
    assert manager.heading.search("Transaction States") == ["node1"]
    assert manager.phrase.search("transaction state") == [("node1", 1.0)]
    assert manager.metadata.node_ids_for_document("doc1") == ["node1"]

    manager.remove_document("doc1")

    assert manager.fts5.search("keyword", 10) == []
    assert manager.heading.search("Transaction States") == []
    assert manager.phrase.search("transaction state") == []
    assert manager.metadata.node_ids_for_document("doc1") == []


def test_index_manager_rebuild_all_uses_node_repository() -> None:
    store = make_store()
    node = make_node("node1", title="Transaction States")
    store.nodes.upsert_node(node)
    manager = make_manager(store)

    manager.rebuild_all()

    assert manager.fts5.search("keyword", 10)[0][0] == "node1"
    assert manager.heading.search("Transaction States") == ["node1"]
    assert manager.phrase.search("transaction state") == [("node1", 1.0)]
    assert manager.metadata.node_ids_for_document("doc1") == ["node1"]

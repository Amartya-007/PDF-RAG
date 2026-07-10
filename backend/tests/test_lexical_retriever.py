from __future__ import annotations

from backend.app.database.store import MetadataStore
from backend.app.domain.models.node import DocumentNode
from backend.app.indexing.full_text_index import FTS5Index
from backend.app.indexing.heading_index import HeadingIndex
from backend.app.indexing.phrase_index import PhraseIndex
from backend.app.models import Document
from backend.app.retrieval.lexical_retriever import LexicalRetriever


def make_node(node_id: str, document_id: str, title: str, text: str) -> DocumentNode:
    return DocumentNode(
        id=node_id,
        document_id=document_id,
        parent_id=None,
        node_type="section",
        title=title,
        text=text,
        page_start=1,
        page_end=1,
        depth=1,
        position=0,
        heading_path=[],
    )


def test_lexical_retriever_scopes_by_session_and_exposes_full_breakdown() -> None:
    store = MetadataStore(":memory:")
    store.init()
    store.create_session("s1", "First")
    store.create_session("s2", "Second")
    store.upsert_document(
        Document("doc1", "one.txt", "sha1", "/tmp/one.txt", "ready", "s1")
    )
    store.upsert_document(
        Document("doc2", "two.txt", "sha2", "/tmp/two.txt", "ready", "s2")
    )
    keep = make_node(
        "keep",
        "doc1",
        "Revenue Policy",
        "Revenue includes settled payments and refunds.",
    )
    filtered = make_node(
        "filtered",
        "doc2",
        "Revenue Policy",
        "Revenue in another session must stay isolated.",
    )
    store.nodes.upsert_many([keep, filtered])
    fts5 = FTS5Index(store.connect)
    heading = HeadingIndex()
    phrase = PhraseIndex()
    for node in [keep, filtered]:
        fts5.upsert(node)
        heading.index(node.id, node.title or "")
        phrase.index(node.id, node.title, node.text)
    retriever = LexicalRetriever(fts5, heading, phrase, store.nodes)

    results = retriever.search("Revenue Policy", session_id="s1", top_k=10)

    assert [node_id for node_id, _score in results] == ["keep"]
    breakdown = retriever.score_breakdown("keep")
    assert set(breakdown) == {
        "fts5_score",
        "heading_score",
        "phrase_score",
        "keyword_coverage",
        "structural_score",
        "fused_score",
    }
    assert breakdown["keyword_coverage"] > 0
    assert breakdown["structural_score"] > 0

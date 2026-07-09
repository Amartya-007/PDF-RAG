from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from backend.app.core.hashing import sha256_file
from backend.app.database.store import MetadataStore
from backend.app.domain.exceptions import EmptyDocumentError
from backend.app.indexing.full_text_index import FTS5Index
from backend.app.indexing.heading_index import HeadingIndex
from backend.app.indexing.index_manager import IndexManager
from backend.app.indexing.metadata_index import MetadataIndex
from backend.app.indexing.phrase_index import PhraseIndex
from backend.app.indexing.sparse import BM25Index
from backend.app.ingestion.layout_parser import LayoutNode
from backend.app.ingestion.structure_builder import StructureBuilder
from backend.app.services.ingestion_service import IngestionService


class StubLayoutParser:
    def __init__(self, nodes: list[LayoutNode]) -> None:
        self.nodes = nodes
        self.calls = 0

    def parse(self, path: Path) -> list[LayoutNode]:
        self.calls += 1
        return self.nodes


def make_test_path(name: str) -> Path:
    root = Path.cwd() / "backend" / ".test-tmp" / "ingestion"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{uuid4().hex}-{name}"


def make_store() -> MetadataStore:
    store = MetadataStore(":memory:")
    store.init()
    store.create_session("s1", "Session One")
    return store


def make_manager(store: MetadataStore) -> IndexManager:
    index_path = Path.cwd() / "backend" / ".test-tmp" / f"ingestion-bm25-{uuid4().hex}.json"
    return IndexManager(
        FTS5Index(store.connect),
        HeadingIndex(),
        PhraseIndex(),
        BM25Index(index_path),
        MetadataIndex(),
        store.nodes,
    )


def make_service(store: MetadataStore, layout: StubLayoutParser) -> IngestionService:
    return IngestionService(
        store=store,
        node_repo=store.nodes,
        job_repo=store.jobs,
        index_mgr=make_manager(store),
        layout=layout,
        builder=StructureBuilder(),
    )


def write_source(text: str) -> Path:
    path = make_test_path("source.txt")
    path.write_text(text, encoding="utf-8")
    return path


def test_ingestion_service_is_idempotent_when_force_is_false() -> None:
    store = make_store()
    layout = StubLayoutParser(
        [
            LayoutNode(text="1. Overview", page_number=1),
            LayoutNode(text="This policy explains the transaction state.", page_number=1),
        ]
    )
    service = make_service(store, layout)
    source = write_source("same bytes")

    first = service.ingest(source, session_id="s1")
    first_nodes = store.nodes.list_nodes_for_document(first.document_id)

    second = service.ingest(source, session_id="s1", force=False)

    assert second.document_id == first.document_id
    assert store.nodes.list_nodes_for_document(first.document_id) == first_nodes
    assert layout.calls == 1


def test_ingestion_service_raises_empty_document_and_marks_failed() -> None:
    store = make_store()
    service = make_service(store, StubLayoutParser([]))
    source = write_source("empty-ish")

    with pytest.raises(EmptyDocumentError):
        service.ingest(source, session_id="s1")

    document = store.find_document_by_hash(sha256_file(source), "s1")
    assert document is not None
    assert document.status == "failed"
    assert store.jobs.list_jobs_for_document(document.document_id)[0].status == "failed"


def test_ingestion_service_records_required_job_statuses_and_progress() -> None:
    store = make_store()
    layout = StubLayoutParser(
        [
            LayoutNode(text="1. Overview", page_number=1),
            LayoutNode(text="This policy explains the transaction state.", page_number=1),
        ]
    )
    service = make_service(store, layout)
    source = write_source("progress bytes")
    progress: list[tuple[int, int, str]] = []

    document = service.ingest(
        source,
        session_id="s1",
        progress=lambda done, total, message: progress.append((done, total, message)),
    )
    job = store.jobs.list_jobs_for_document(document.document_id)[0]

    assert job.status == "completed"
    assert [message for _, _, message in progress] == [
        "Queued",
        "Parsing layout",
        "Cleaning text",
        "Detecting structure",
        "Building nodes",
        "Indexing full text",
        "Indexing headings",
        "Completed",
    ]
    assert store.count_chunks_for_document(document.document_id) > 0

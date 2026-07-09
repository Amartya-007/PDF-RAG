from __future__ import annotations

from backend.app.database.repositories.answer_repository import AnswerRepository
from backend.app.database.repositories.document_repository import DocumentRepository
from backend.app.database.repositories.job_repository import JobRepository
from backend.app.database.repositories.node_repository import NodeRepository
from backend.app.database.repositories.session_repository import SessionRepository
from backend.app.database.store import MetadataStore
from backend.app.models import Answer, Citation, Document, IngestionJob


def make_store() -> MetadataStore:
    store = MetadataStore(":memory:")
    store.init()
    return store


def test_metadata_store_exposes_all_repositories() -> None:
    store = make_store()

    assert isinstance(store.documents, DocumentRepository)
    assert isinstance(store.sessions, SessionRepository)
    assert isinstance(store.nodes, NodeRepository)
    assert isinstance(store.jobs, JobRepository)
    assert isinstance(store.answers, AnswerRepository)


def test_metadata_store_document_and_session_methods_delegate_to_repositories() -> None:
    store = make_store()

    session = store.create_session("s1", "Session One")
    document = Document(
        document_id="doc1",
        filename="source.pdf",
        sha256="abc123",
        path="/tmp/source.pdf",
        status="ready",
        session_id=session.session_id,
    )

    store.upsert_document(document)

    assert store.find_document_by_hash("abc123", "s1") == document
    assert store.count_documents_for_session("s1") == 1
    assert store.list_documents("s1") == [document]


def test_job_repository_exposes_task_named_methods() -> None:
    store = make_store()
    store.create_session("s1", "Session One")
    store.upsert_document(
        Document(
            document_id="doc1",
            filename="source.pdf",
            sha256="abc123",
            path="/tmp/source.pdf",
            status="processing",
            session_id="s1",
        )
    )
    job = IngestionJob(
        job_id="job1",
        document_id="doc1",
        session_id="s1",
        status="queued",
        progress_message="Queued",
    )

    store.jobs.create_job(job)
    store.jobs.update_job_status("job1", "completed", "Done")

    assert store.jobs.get_job("job1").status == "completed"
    assert [item.job_id for item in store.jobs.list_jobs_for_document("doc1")] == ["job1"]


def test_answer_repository_persists_answer_and_citations() -> None:
    store = make_store()
    answer = Answer(
        question="What is cited?",
        answer="The policy is cited.",
        answerable=True,
        citations=[
            Citation(
                source_id="S1",
                document_id="doc1",
                filename="policy.pdf",
                page_start=1,
                page_end=2,
                chunk_id="node1",
                excerpt="Policy excerpt",
            )
        ],
        debug={"strategy": "EXTRACTIVE"},
    )

    store.answers.save_answer("answer1", "s1", answer)

    assert store.answers.get_answer("answer1") == answer

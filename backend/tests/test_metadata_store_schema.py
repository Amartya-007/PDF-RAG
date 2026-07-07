"""
Tests for task 1.3: nodes, nodes_fts, and ingestion_jobs table additions to MetadataStore.

Validates Requirements 3.1-3.4, 29.2.
"""
from __future__ import annotations

import sqlite3

import pytest

from backend.app.database.store import MetadataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_store() -> MetadataStore:
    """Create and initialise an in-memory MetadataStore."""
    store = MetadataStore(":memory:")
    store.init()
    return store


def column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"pragma table_info({table})").fetchall()
    return {row["name"] for row in rows}


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "select count(*) from sqlite_master where type in ('table','shadow') and name = ?",
        (name,),
    ).fetchone()
    return bool(row[0])


def virtual_table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "select count(*) from sqlite_master where type = 'table' and name = ?",
        (name,),
    ).fetchone()
    return bool(row[0])


def index_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "select count(*) from sqlite_master where type = 'index' and name = ?",
        (name,),
    ).fetchone()
    return bool(row[0])


def insert_document(conn: sqlite3.Connection, document_id: str, session_id: str = "s1") -> None:
    conn.execute(
        """insert into documents(document_id, filename, sha256, path, status, session_id)
           values(?, 'test.pdf', 'abc', '/tmp/test.pdf', 'ready', ?)""",
        (document_id, session_id),
    )


def insert_session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute(
        "insert or ignore into sessions(session_id, title) values(?, 'Test')",
        (session_id,),
    )


def insert_node(conn: sqlite3.Connection, node_id: str, document_id: str) -> None:
    conn.execute(
        """insert into nodes(node_id, document_id, parent_id, node_type, title, text,
               page_start, page_end, depth, position, heading_path)
           values(?, ?, null, 'paragraph', null, 'hello world', 1, 1, 1, 0, '[]')""",
        (node_id, document_id),
    )


# ---------------------------------------------------------------------------
# Requirement 3.1: nodes table columns
# ---------------------------------------------------------------------------

class TestNodesTableSchema:
    """Requirement 3.1 — nodes table has all required columns."""

    def test_nodes_table_exists(self) -> None:
        store = make_store()
        with store.connect() as conn:
            assert table_exists(conn, "nodes")

    def test_nodes_columns_present(self) -> None:
        store = make_store()
        with store.connect() as conn:
            cols = column_names(conn, "nodes")
        required = {
            "node_id", "document_id", "parent_id", "node_type",
            "title", "text", "page_start", "page_end",
            "depth", "position", "heading_path",
        }
        assert required.issubset(cols), f"Missing columns: {required - cols}"

    def test_nodes_indexes_exist(self) -> None:
        store = make_store()
        with store.connect() as conn:
            assert index_exists(conn, "idx_nodes_document")
            assert index_exists(conn, "idx_nodes_parent")
            assert index_exists(conn, "idx_nodes_depth")

    def test_nodes_fk_rejects_unknown_document(self) -> None:
        """FK from nodes(document_id) → documents(document_id) must be enforced."""
        store = make_store()
        with store.connect() as conn:
            conn.execute("pragma foreign_keys = ON")
            with pytest.raises(sqlite3.IntegrityError):
                insert_node(conn, "n1", "nonexistent-doc")


# ---------------------------------------------------------------------------
# Requirement 3.2 / design: nodes_fts FTS5 virtual table
# ---------------------------------------------------------------------------

class TestNodesFtsTable:
    """FTS5 virtual table nodes_fts is created with correct columns."""

    def test_nodes_fts_exists(self) -> None:
        store = make_store()
        with store.connect() as conn:
            assert virtual_table_exists(conn, "nodes_fts")

    def test_nodes_fts_accepts_insert(self) -> None:
        store = make_store()
        with store.connect() as conn:
            insert_session(conn, "s1")
            insert_document(conn, "doc1")
            insert_node(conn, "n1", "doc1")
            # Manually populate FTS (triggers would normally do this)
            conn.execute(
                "insert into nodes_fts(node_id, text, title) values(?, ?, ?)",
                ("n1", "hello world", ""),
            )
            rows = conn.execute(
                "select node_id from nodes_fts where nodes_fts match 'hello'"
            ).fetchall()
        assert any(row["node_id"] == "n1" for row in rows)


# ---------------------------------------------------------------------------
# Requirement 29.2 / design: ingestion_jobs table
# ---------------------------------------------------------------------------

class TestIngestionJobsTable:
    """ingestion_jobs table DDL and check constraint."""

    def test_ingestion_jobs_table_exists(self) -> None:
        store = make_store()
        with store.connect() as conn:
            assert table_exists(conn, "ingestion_jobs")

    def test_ingestion_jobs_columns_present(self) -> None:
        store = make_store()
        with store.connect() as conn:
            cols = column_names(conn, "ingestion_jobs")
        required = {
            "job_id", "document_id", "session_id",
            "status", "progress_message", "created_at", "updated_at",
        }
        assert required.issubset(cols), f"Missing columns: {required - cols}"

    def test_ingestion_jobs_indexes_exist(self) -> None:
        store = make_store()
        with store.connect() as conn:
            assert index_exists(conn, "idx_jobs_document")
            assert index_exists(conn, "idx_jobs_status")

    def test_progress_message_check_constraint_allows_500_chars(self) -> None:
        store = make_store()
        with store.connect() as conn:
            insert_session(conn, "s1")
            insert_document(conn, "doc1")
            msg_500 = "x" * 500
            conn.execute(
                """insert into ingestion_jobs(job_id, document_id, session_id, progress_message)
                   values('j1', 'doc1', 's1', ?)""",
                (msg_500,),
            )

    def test_progress_message_check_constraint_rejects_501_chars(self) -> None:
        store = make_store()
        with store.connect() as conn:
            insert_session(conn, "s1")
            insert_document(conn, "doc1")
            msg_501 = "x" * 501
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """insert into ingestion_jobs(job_id, document_id, session_id, progress_message)
                       values('j2', 'doc1', 's1', ?)""",
                    (msg_501,),
                )


# ---------------------------------------------------------------------------
# Requirement 3.3: init() is additive — calling it twice is safe
# ---------------------------------------------------------------------------

class TestInitIsAdditive:
    def test_double_init_does_not_raise(self) -> None:
        store = make_store()
        store.init()  # second call must not fail or drop anything

    def test_existing_data_survives_reinit(self) -> None:
        store = make_store()
        with store.connect() as conn:
            insert_session(conn, "s1")
            insert_document(conn, "doc1")
            insert_node(conn, "n1", "doc1")
        store.init()  # re-init
        with store.connect() as conn:
            rows = conn.execute("select node_id from nodes where node_id = 'n1'").fetchall()
        assert len(rows) == 1, "Re-init must not delete existing rows"


# ---------------------------------------------------------------------------
# Requirement 3.4: cascade-delete nodes when document is deleted
# ---------------------------------------------------------------------------

class TestCascadeDelete:
    """delete_document() removes nodes before the document row (Req 3.4)."""

    def test_delete_document_removes_nodes(self) -> None:
        store = make_store()
        with store.connect() as conn:
            insert_session(conn, "s1")
            insert_document(conn, "doc1")
            insert_node(conn, "n1", "doc1")
            insert_node(conn, "n2", "doc1")

        store.delete_document("doc1")

        with store.connect() as conn:
            rows = conn.execute(
                "select node_id from nodes where document_id = 'doc1'"
            ).fetchall()
        assert rows == [], "All nodes for deleted document must be removed"

    def test_delete_session_removes_nodes(self) -> None:
        """delete_session() must also cascade-delete nodes for all session documents."""
        store = make_store()
        with store.connect() as conn:
            insert_session(conn, "s1")
            insert_document(conn, "doc1", "s1")
            insert_document(conn, "doc2", "s1")
            insert_node(conn, "n1", "doc1")
            insert_node(conn, "n2", "doc2")

        store.delete_session("s1")

        with store.connect() as conn:
            rows = conn.execute("select node_id from nodes").fetchall()
        assert rows == [], "All nodes for session documents must be removed on session delete"

    def test_delete_document_preserves_nodes_for_other_documents(self) -> None:
        store = make_store()
        with store.connect() as conn:
            insert_session(conn, "s1")
            insert_document(conn, "doc1", "s1")
            insert_document(conn, "doc2", "s1")
            insert_node(conn, "n1", "doc1")
            insert_node(conn, "n2", "doc2")

        store.delete_document("doc1")

        with store.connect() as conn:
            rows = conn.execute("select node_id from nodes").fetchall()
        remaining = {row["node_id"] for row in rows}
        assert remaining == {"n2"}, "Nodes for unrelated documents must not be deleted"

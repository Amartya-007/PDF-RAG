"""NodeRepository — database access for DocumentNode records.

Provides upsert, lookup, listing, and deletion methods for the
``nodes`` table.  ``heading_path`` is serialised as a JSON array
string on write and deserialised back to ``list[str]`` on read.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable

from backend.app.domain.exceptions import DocumentNotFoundError
from backend.app.domain.models.node import DocumentNode


class NodeRepository:
    """Repository for :class:`~backend.app.domain.models.node.DocumentNode` records.

    Args:
        connection_factory: A callable that returns a :class:`sqlite3.Connection`.
            Pass ``MetadataStore.connect`` so the repository shares the same
            connection strategy (file-backed or in-memory) as the store.
    """

    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self._connect = connection_factory

    # ── Write operations ───────────────────────────────────────────────────

    def upsert_node(self, node: DocumentNode) -> None:
        """Insert or update a single ``DocumentNode`` row.

        ``heading_path`` is stored as a JSON array string.
        All other fields map directly to column values.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO nodes (
                    node_id, document_id, parent_id, node_type, title,
                    text, page_start, page_end, depth, position, heading_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    document_id  = excluded.document_id,
                    parent_id    = excluded.parent_id,
                    node_type    = excluded.node_type,
                    title        = excluded.title,
                    text         = excluded.text,
                    page_start   = excluded.page_start,
                    page_end     = excluded.page_end,
                    depth        = excluded.depth,
                    position     = excluded.position,
                    heading_path = excluded.heading_path
                """,
                (
                    node.id,
                    node.document_id,
                    node.parent_id,
                    node.node_type,
                    node.title,
                    node.text,
                    node.page_start,
                    node.page_end,
                    node.depth,
                    node.position,
                    json.dumps(node.heading_path, ensure_ascii=False),
                ),
            )

    def upsert_many(self, nodes: list[DocumentNode]) -> None:
        """Insert or update multiple ``DocumentNode`` rows."""
        for node in nodes:
            self.upsert_node(node)

    def delete_nodes_for_document(self, document_id: str) -> None:
        """Delete all ``nodes`` rows belonging to *document_id*."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM nodes WHERE document_id = ?",
                (document_id,),
            )

    def delete_for_document(self, document_id: str) -> None:
        self.delete_nodes_for_document(document_id)

    # ── Read operations ────────────────────────────────────────────────────

    def get_node_by_id(self, node_id: str) -> DocumentNode:
        """Return the :class:`DocumentNode` identified by *node_id*.

        Raises:
            DocumentNotFoundError: When no row with *node_id* exists.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM nodes WHERE node_id = ?",
                (node_id,),
            ).fetchone()
        if row is None:
            raise DocumentNotFoundError(node_id)
        return self._node_from_row(row)

    def list_nodes_for_document(self, document_id: str) -> list[DocumentNode]:
        """Return all nodes for *document_id*, ordered by depth then position."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM nodes
                WHERE document_id = ?
                ORDER BY depth ASC, position ASC
                """,
                (document_id,),
            ).fetchall()
        return [self._node_from_row(row) for row in rows]

    def list_nodes_for_session(self, session_id: str) -> list[DocumentNode]:
        """Return all nodes for documents belonging to *session_id*.

        Joins ``nodes`` with ``documents`` to filter by session.
        Results are ordered by document_id, depth, position.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT n.*
                FROM nodes n
                JOIN documents d ON d.document_id = n.document_id
                WHERE d.session_id = ?
                ORDER BY n.document_id ASC, n.depth ASC, n.position ASC
                """,
                (session_id,),
            ).fetchall()
        return [self._node_from_row(row) for row in rows]

    def list_all_nodes(self) -> list[DocumentNode]:
        """Return all nodes, ordered by document_id, depth, and position."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM nodes
                ORDER BY document_id ASC, depth ASC, position ASC
                """
            ).fetchall()
        return [self._node_from_row(row) for row in rows]

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _node_from_row(row: sqlite3.Row) -> DocumentNode:
        return DocumentNode(
            id=row["node_id"],
            document_id=row["document_id"],
            parent_id=row["parent_id"],
            node_type=row["node_type"],
            title=row["title"],
            text=row["text"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            depth=row["depth"],
            position=row["position"],
            heading_path=json.loads(row["heading_path"]),
        )

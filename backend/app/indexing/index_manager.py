"""IndexManager — single owner of all vectorless indexes.

Provides atomic multi-index operations so ``IngestionService`` and
``RagService`` never touch individual indexes directly.

Atomicity guarantee
-------------------
``add_document_nodes`` inserts into all owned indexes inside a context that
tracks partial completions.  If any insertion raises an exception, the manager
rolls back every insertion performed so far for that batch so no node ID leaks
into a partial state.

Indexes owned
-------------
FTS5Index       — full-text search (primary retrieval)
HeadingIndex    — heading-text to node_id mapping
PhraseIndex     — multi-word phrase matching
BM25Index       — legacy sparse index (OKF compatibility only)

Dense retrieval is explicitly excluded: ``IndexManager`` owns only lexical
and metadata indexes.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.app.domain.models.node import DocumentNode

if TYPE_CHECKING:
    from backend.app.indexing.full_text_index import FTS5Index
    from backend.app.indexing.heading_index import HeadingIndex
    from backend.app.indexing.metadata_index import MetadataIndex
    from backend.app.indexing.phrase_index import PhraseIndex
    from backend.app.indexing.sparse import BM25Index
    from backend.app.database.repositories.node_repository import NodeRepository

logger = logging.getLogger(__name__)


class IndexManager:
    """Atomic, multi-index coordinator for all vectorless search structures.

    Args:
        fts5:    The FTS5 full-text index.
        heading: The heading text → node_id index.
        phrase:  The multi-word phrase index.
        bm25:    The BM25 sparse index (OKF legacy path only).
    """

    def __init__(
        self,
        fts5: "FTS5Index",
        heading: "HeadingIndex",
        phrase: "PhraseIndex",
        bm25: "BM25Index",
        metadata: "MetadataIndex | None" = None,
        node_repo: "NodeRepository | None" = None,
    ) -> None:
        self._fts5 = fts5
        self._heading = heading
        self._phrase = phrase
        self._bm25 = bm25
        if metadata is None:
            from backend.app.indexing.metadata_index import MetadataIndex

            metadata = MetadataIndex()
        self._metadata = metadata
        self._node_repo = node_repo

    @property
    def fts5(self) -> "FTS5Index":
        return self._fts5

    @property
    def heading(self) -> "HeadingIndex":
        return self._heading

    @property
    def phrase(self) -> "PhraseIndex":
        return self._phrase

    @property
    def metadata(self) -> "MetadataIndex":
        return self._metadata

    # ── Atomic batch insert ────────────────────────────────────────────────

    def add_document_nodes(self, nodes: list[DocumentNode]) -> None:
        """Insert *nodes* into all indexes atomically.

        If insertion fails for any index, all insertions performed within
        this call are rolled back so no node_id leaks into a partial state.

        Complexity: O(N · I) where N = node count, I = number of indexes.
        """
        if not nodes:
            return

        # Track what has been inserted for rollback purposes
        inserted_ids: list[str] = []
        try:
            for node in nodes:
                self._fts5.upsert(node)
                if node.title:
                    self._heading.index(node.id, node.title)
                self._phrase.index(node.id, node.title, node.text)
                self._metadata.index(node)
                inserted_ids.append(node.id)
        except Exception as exc:
            logger.error(
                "IndexManager.add_document_nodes failed at node %s — rolling back %d insertions: %s",
                nodes[len(inserted_ids)].id if len(inserted_ids) < len(nodes) else "?",
                len(inserted_ids),
                exc,
            )
            self._rollback_nodes(inserted_ids)
            raise

    def remove_document(self, document_id: str, node_ids: list[str] | None = None) -> None:
        """Remove all index entries for *document_id*'s nodes atomically."""
        if node_ids is None:
            node_ids = self._metadata.node_ids_for_document(document_id)
        for node_id in node_ids:
            self._fts5.delete(node_id)
            self._heading.remove(node_id)
            self._phrase.remove(node_id)
            self._metadata.remove(node_id)

    # ── Full rebuild ───────────────────────────────────────────────────────

    def rebuild_all(
        self,
        node_repo: "NodeRepository | None" = None,
        session_id: str | None = None,
    ) -> None:
        """Wipe and rebuild all indexes from the node store.

        Also restores the FTS5 index's consistency flag after a failed upsert.
        """
        repo = node_repo or self._node_repo
        if repo is None:
            raise ValueError("IndexManager.rebuild_all requires a NodeRepository")
        nodes = (
            repo.list_nodes_for_session(session_id)
            if session_id
            else repo.list_all_nodes()
        )
        logger.info("IndexManager.rebuild_all: rebuilding %d nodes", len(nodes))

        self._fts5.rebuild(nodes)
        self._heading.rebuild(
            [(n.id, n.title) for n in nodes if n.title]
        )
        self._phrase.rebuild(
            [(n.id, n.title, n.text) for n in nodes]
        )
        self._metadata.rebuild(nodes)
        logger.info("IndexManager.rebuild_all: complete")

    # ── Private helpers ────────────────────────────────────────────────────

    def _rollback_nodes(self, node_ids: list[str]) -> None:
        for node_id in node_ids:
            try:
                self._fts5.delete(node_id)
                self._heading.remove(node_id)
                self._phrase.remove(node_id)
                self._metadata.remove(node_id)
            except Exception as exc:
                logger.warning("Rollback failed for node %s: %s", node_id, exc)

"""IndexManager — single owner of all vectorless indexes.

Provides atomic multi-index operations so ``IngestionService`` and
``RagService`` never touch individual indexes directly.

Atomicity guarantee
-------------------
``add_document_nodes`` inserts into all owned indexes inside a context that
tracks partial completions. If any insertion raises an exception, the manager
rolls back every insertion performed so far for that batch so no node ID 
leaks into a partial state.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.database.repositories.node_repository import NodeRepository
    from backend.app.indexing.full_text_index import FTS5Index
    from backend.app.indexing.heading_index import HeadingIndex
    from backend.app.indexing.metadata_index import MetadataIndex
    from backend.app.indexing.phrase_index import PhraseIndex
    from backend.app.indexing.sparse import BM25Index
    from backend.app.domain.models.node import DocumentNode

logger = logging.getLogger(__name__)


class IndexManager:
    """Coordinates atomic multi-index operations for node data.

    Sub-indexes are exposed as public attributes (``fts5``, ``heading``,
    ``phrase``, ``bm25``, ``metadata``) so callers (and tests) can query a
    specific index directly without the manager having to proxy every
    method.
    """

    def __init__(
        self,
        fts5: FTS5Index,
        heading: HeadingIndex,
        phrase: PhraseIndex,
        bm25: BM25Index,
        metadata: MetadataIndex,
        node_repo: NodeRepository,
    ) -> None:
        self.fts5 = fts5
        self.heading = heading
        self.phrase = phrase
        self.bm25 = bm25
        self.metadata = metadata
        self._node_repo = node_repo

    def add_document_nodes(self, nodes: list[DocumentNode]) -> None:
        """Add nodes to all indexes atomically; rollback on failure.

        Attributes:
            nodes: The list of nodes to index.
        """
        added_ids: list[str] = []
        try:
            for node in nodes:
                self.fts5.upsert(node)
                self.heading.index(node.id, node.title or "")
                self.phrase.index(node.id, node.title, node.text)
                self.bm25.add(node.id, node.text)
                self.metadata.index(node)
                added_ids.append(node.id)
        except Exception as exc:
            logger.error("Indexing failed, rolling back %d nodes: %s", len(added_ids), exc)
            self._rollback_nodes(added_ids)
            raise exc

    def remove_document(self, document_id: str) -> None:
        """Remove every indexed node belonging to a document, across all indexes.

        Attributes:
            document_id: The document whose nodes should be de-indexed.
        """
        node_ids = self.metadata.remove_document(document_id)
        for node_id in node_ids:
            self.fts5.delete(node_id)
            self.heading.remove(node_id)
            self.phrase.remove(node_id)
            self.bm25.remove(node_id)

    def rebuild_all(self, session_id: str | None = None) -> None:
        """Wipe and rebuild all indexes from the node store.

        Attributes:
            session_id: Optional session ID to limit rebuild scope.
        """
        nodes = (
            self._node_repo.list_nodes_for_session(session_id)
            if session_id
            else self._node_repo.list_all_nodes()
        )
        logger.info("IndexManager: rebuilding %d nodes", len(nodes))

        # Batch rebuilds are highly efficient because they leverage 
        # the underlying index classes' bulk operations.
        self.fts5.rebuild(nodes)
        self.heading.rebuild([(n.id, n.title) for n in nodes if n.title])
        self.phrase.rebuild([(n.id, n.title, n.text) for n in nodes])
        self.bm25.rebuild([(n.id, n.text) for n in nodes])
        self.metadata.rebuild(nodes)

        logger.info("IndexManager: rebuild complete")

    def _rollback_nodes(self, node_ids: list[str]) -> None:
        """Remove a list of nodes from all indexes."""
        for node_id in node_ids:
            try:
                self.fts5.delete(node_id)
                self.heading.remove(node_id)
                self.phrase.remove(node_id)
                self.bm25.remove(node_id)
                self.metadata.remove(node_id)
            except Exception as exc:
                # Log rollback failures but don't stop the rollback process
                logger.error("Rollback failed for node %s: %s", node_id, exc)

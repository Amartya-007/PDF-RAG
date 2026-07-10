"""MetadataIndex — lightweight node metadata lookup for vectorless indexes."""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.domain.models.node import DocumentNode


class MetadataIndex:
    """Tracks node membership by document for index removal and rebuilds.

    Attributes:
        _document_to_nodes: Maps document_id to a set of node_ids.
        _node_to_document:  Maps node_id to document_id for reverse lookup.
    """

    def __init__(self) -> None:
        # OPTIMIZATION: Using set() makes removal and lookup O(1)
        self._document_to_nodes: dict[str, set[str]] = defaultdict(set)
        self._node_to_document: dict[str, str] = {}

    def index(self, node: DocumentNode) -> None:
        """Register a node in the index.

        Attributes:
            node: The DocumentNode to index.
        """
        # Ensure we don't have stale mappings
        self.remove(node.id)
        
        self._document_to_nodes[node.document_id].add(node.id)
        self._node_to_document[node.id] = node.document_id

    def remove(self, node_id: str) -> None:
        """Remove a node from the index.

        Attributes:
            node_id: The node identifier to remove.
        """
        document_id = self._node_to_document.pop(node_id, None)
        if document_id is None:
            return
            
        bucket = self._document_to_nodes.get(document_id)
        if bucket:
            bucket.discard(node_id)
            if not bucket:
                del self._document_to_nodes[document_id]

    def remove_document(self, document_id: str) -> list[str]:
        """Remove all nodes associated with a document.

        Attributes:
            document_id: The document identifier.
        
        Returns:
            A list of node_ids that were removed.
        """
        node_ids = self._document_to_nodes.pop(document_id, set())
        for node_id in node_ids:
            self._node_to_document.pop(node_id, None)
        return list(node_ids)

    def node_ids_for_document(self, document_id: str) -> list[str]:
        """Get all node identifiers belonging to a document.

        Attributes:
            document_id: The document identifier.
        """
        return list(self._document_to_nodes.get(document_id, set()))
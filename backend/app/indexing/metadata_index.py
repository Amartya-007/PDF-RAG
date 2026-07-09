"""MetadataIndex — lightweight node metadata lookup for vectorless indexes."""
from __future__ import annotations

from collections import defaultdict

from backend.app.domain.models.node import DocumentNode


class MetadataIndex:
    """Tracks node membership by document for index removal and rebuilds."""

    def __init__(self) -> None:
        self._document_to_nodes: dict[str, list[str]] = defaultdict(list)
        self._node_to_document: dict[str, str] = {}

    def index(self, node: DocumentNode) -> None:
        self.remove(node.id)
        bucket = self._document_to_nodes[node.document_id]
        if node.id not in bucket:
            bucket.append(node.id)
        self._node_to_document[node.id] = node.document_id

    def remove(self, node_id: str) -> None:
        document_id = self._node_to_document.pop(node_id, None)
        if document_id is None:
            return
        bucket = self._document_to_nodes.get(document_id, [])
        if node_id in bucket:
            bucket.remove(node_id)
        if not bucket and document_id in self._document_to_nodes:
            del self._document_to_nodes[document_id]

    def remove_document(self, document_id: str) -> list[str]:
        node_ids = self._document_to_nodes.pop(document_id, [])
        for node_id in node_ids:
            self._node_to_document.pop(node_id, None)
        return list(node_ids)

    def node_ids_for_document(self, document_id: str) -> list[str]:
        return list(self._document_to_nodes.get(document_id, []))

    def rebuild(self, nodes: list[DocumentNode]) -> None:
        self._document_to_nodes.clear()
        self._node_to_document.clear()
        for node in nodes:
            self.index(node)


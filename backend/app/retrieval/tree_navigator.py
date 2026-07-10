"""Deterministic tree expansion for DocumentNode retrieval.

Tree navigation is deliberately model-free. It uses persisted
``DocumentNode`` relationships to add nearby context around lexical hits:
the parent section, adjacent siblings, and descendants up to a bounded depth.
"""
from __future__ import annotations

from typing import Protocol

from backend.app.domain.models.node import DocumentNode


class NodeLookup(Protocol):
    def list_nodes_for_document(self, document_id: str) -> list[DocumentNode]: ...


class TreeNavigator:
    """Expands matched nodes using parent, sibling, and child relationships."""

    def expand(
        self,
        matched_nodes: list[DocumentNode],
        node_repo: NodeLookup,
        expand_depth: int = 2,
        include_siblings: bool = True,
        cancelled: list[bool] | None = None,
    ) -> list[DocumentNode]:
        """Deterministically adds structural context around lexical hits.
        
        Args:
            matched_nodes: Initial set of nodes to expand around.
            node_repo: Database protocol to fetch document nodes.
            expand_depth: How many levels of children to traverse downwards.
            include_siblings: Whether to fetch immediately adjacent siblings.
            cancelled: Mutable boolean list to trigger an early abort.
        """
        if not matched_nodes:
            return []

        # 1. Fetch all required context in bulk
        nodes_by_document = self._load_documents(matched_nodes, node_repo)
        
        # 2. Pre-compute structural maps ONCE per document to avoid O(N*M) loop overhead
        document_maps: dict[str, tuple[dict[str, DocumentNode], dict[str | None, list[DocumentNode]]]] = {}
        for doc_id, doc_nodes in nodes_by_document.items():
            document_maps[doc_id] = self._maps(doc_nodes)

        ordered: list[DocumentNode] = []
        seen: set[str] = set()

        # 3. Expand each matched node
        for node in matched_nodes:
            # Fast abort check for concurrent API cancellations
            if cancelled and cancelled[0]:
                break

            by_id, by_parent = document_maps.get(node.document_id, ({}, {}))

            # Add the core hit
            self._append(node, ordered, seen)

            # Add parent (if it's not a top-level document container)
            parent = by_id.get(node.parent_id or "")
            if parent is not None and parent.node_type != "document":
                self._append(parent, ordered, seen)

            # Add immediate adjacent siblings
            if include_siblings and node.parent_id is not None:
                siblings = sorted(
                    by_parent.get(node.parent_id, []),
                    key=lambda item: item.position,
                )
                index = next(
                    (i for i, sibling in enumerate(siblings) if sibling.id == node.id),
                    -1,
                )
                
                if index > 0:
                    self._append(siblings[index - 1], ordered, seen)
                if 0 <= index < len(siblings) - 1:
                    self._append(siblings[index + 1], ordered, seen)

            # Add descendants down to `expand_depth`
            for descendant in self._descendants(node, by_parent, expand_depth, cancelled):
                self._append(descendant, ordered, seen)

        return ordered

    def navigate(
        self,
        matched_nodes: list[DocumentNode],
        node_repo: NodeLookup,
        expand_depth: int = 2,
        include_siblings: bool = True,
        cancelled: list[bool] | None = None,
    ) -> list[DocumentNode]:
        """Backward-compatible alias for deterministic expansion."""
        return self.expand(
            matched_nodes,
            node_repo,
            expand_depth=expand_depth,
            include_siblings=include_siblings,
            cancelled=cancelled,
        )

    @staticmethod
    def _load_documents(
        matched_nodes: list[DocumentNode],
        node_repo: NodeLookup,
    ) -> dict[str, list[DocumentNode]]:
        document_ids = {node.document_id for node in matched_nodes}
        return {
            document_id: node_repo.list_nodes_for_document(document_id)
            for document_id in document_ids
        }

    @staticmethod
    def _maps(
        nodes: list[DocumentNode],
    ) -> tuple[dict[str, DocumentNode], dict[str | None, list[DocumentNode]]]:
        """Creates fast-lookup dictionaries for tree traversal."""
        by_id = {node.id: node for node in nodes}
        by_parent: dict[str | None, list[DocumentNode]] = {}
        for node in nodes:
            by_parent.setdefault(node.parent_id, []).append(node)
        return by_id, by_parent

    def _descendants(
        self,
        node: DocumentNode,
        by_parent: dict[str | None, list[DocumentNode]],
        remaining_depth: int,
        cancelled: list[bool] | None = None,
    ) -> list[DocumentNode]:
        """Recursively fetches children down to a specific depth."""
        if remaining_depth <= 0 or (cancelled and cancelled[0]):
            return []
            
        descendants: list[DocumentNode] = []
        for child in sorted(by_parent.get(node.id, []), key=lambda item: item.position):
            descendants.append(child)
            descendants.extend(
                self._descendants(child, by_parent, remaining_depth - 1, cancelled)
            )
        return descendants

    @staticmethod
    def _append(
        node: DocumentNode,
        ordered: list[DocumentNode],
        seen: set[str],
    ) -> None:
        """Deduplicates and appends a node to the final list."""
        if node.id in seen:
            return
        seen.add(node.id)
        ordered.append(node)
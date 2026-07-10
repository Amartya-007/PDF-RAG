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
    ) -> list[DocumentNode]:
        if not matched_nodes:
            return []

        nodes_by_document = self._load_documents(matched_nodes, node_repo)
        ordered: list[DocumentNode] = []
        seen: set[str] = set()

        for node in matched_nodes:
            document_nodes = nodes_by_document.get(node.document_id, [])
            by_id, by_parent = self._maps(document_nodes)

            self._append(node, ordered, seen)
            parent = by_id.get(node.parent_id or "")
            if parent is not None and parent.node_type != "document":
                self._append(parent, ordered, seen)

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

            for descendant in self._descendants(node, by_parent, expand_depth):
                self._append(descendant, ordered, seen)

        return ordered

    def navigate(
        self,
        matched_nodes: list[DocumentNode],
        node_repo: NodeLookup,
        expand_depth: int = 2,
        include_siblings: bool = True,
    ) -> list[DocumentNode]:
        """Backward-compatible alias for deterministic expansion."""
        return self.expand(
            matched_nodes,
            node_repo,
            expand_depth=expand_depth,
            include_siblings=include_siblings,
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
    ) -> list[DocumentNode]:
        if remaining_depth <= 0:
            return []
        descendants: list[DocumentNode] = []
        for child in sorted(by_parent.get(node.id, []), key=lambda item: item.position):
            descendants.append(child)
            descendants.extend(
                self._descendants(child, by_parent, remaining_depth - 1)
            )
        return descendants

    @staticmethod
    def _append(
        node: DocumentNode,
        ordered: list[DocumentNode],
        seen: set[str],
    ) -> None:
        if node.id in seen:
            return
        seen.add(node.id)
        ordered.append(node)

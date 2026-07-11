"""StructureBuilder — converts a flat LayoutNode list into a DocumentNode tree.

Pipeline
--------
LayoutNode list (flat)
    ↓  HeadingDetector classification (batch)
Annotated nodes with is_heading + depth
    ↓  Parent-stack tree assembly
DocumentNode tree: document (depth 0) → chapter (1) → section (2) →
                   subsection (3) → paragraph/table/list (leaf)

When no headings are detected at all, the document falls back to fixed
5-page groups (node_type "section", depth 1) so a headingless document still
produces a bounded hierarchy instead of one giant node.

Paragraph size invariant
------------------------
Target: 80-250 words. Hard cap: 500 words. Body text exceeding the cap is
split at sentence boundaries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.app.domain.models.node import DocumentNode, stable_id
from backend.app.ingestion.heading_detector import HeadingDetector

if TYPE_CHECKING:
    from backend.app.ingestion.layout_parser import LayoutNode

# Constants for paragraph management
_MAX_PARAGRAPH_WORDS = 500
_TARGET_PARAGRAPH_WORDS = 200
_PAGE_GROUP_SIZE = 5

# Depth -> node_type for detected headings (depth 1 = chapter, 2 = section,
# 3+ = subsection; HeadingDetector currently only emits depths 1-3).
_HEADING_NODE_TYPES = {1: "chapter", 2: "section", 3: "subsection"}

# Pre-compiled regex
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_COUNT_RE = re.compile(r"\S+")


@dataclass
class _StackEntry:
    """One ancestor frame while walking the flat node list."""

    depth: int
    node_id: str
    # heading_path that THIS node's own children should carry: the ancestor
    # titles up to and including this node's own title (if it has one).
    child_heading_path: list[str]


class StructureBuilder:
    """Orchestrates the conversion of flat LayoutNodes into a DocumentNode tree.

    ``document_id`` may be supplied at construction time (for one-off/test
    usage) or per call to :meth:`build` (for a long-lived builder instance
    shared across many documents, as in the ingestion service). A value must
    be available from one of the two.
    """

    def __init__(
        self,
        heading_detector: HeadingDetector | None = None,
        *,
        document_id: str | None = None,
    ) -> None:
        self._detector = heading_detector or HeadingDetector()
        self._document_id = document_id

    def build(
        self, nodes: list[LayoutNode], document_id: str | None = None
    ) -> list[DocumentNode]:
        """Convert flat layout nodes into a hierarchical tree of DocumentNodes."""
        doc_id = document_id or self._document_id
        if doc_id is None:
            raise ValueError(
                "StructureBuilder requires a document_id, either at construction "
                "time or as an argument to build()."
            )
        if not nodes:
            return []

        results = self._detector.detect(nodes)
        if any(result.is_heading for result in results):
            return self._build_with_headings(doc_id, nodes, results)
        return self._build_page_groups(doc_id, nodes)

    # ── Heading-driven tree assembly ────────────────────────────────────

    def _build_with_headings(
        self,
        doc_id: str,
        nodes: list[LayoutNode],
        results: list,
    ) -> list[DocumentNode]:
        max_page = max((n.page_number for n in nodes), default=1)
        root_id = stable_id("node", doc_id, depth=0, position=0, heading_path=[])
        out: list[DocumentNode] = [
            DocumentNode(
                id=root_id,
                document_id=doc_id,
                parent_id=None,
                node_type="document",
                title=None,
                text="",
                page_start=1,
                page_end=max_page,
                depth=0,
                position=0,
                heading_path=[],
            )
        ]

        stack: list[_StackEntry] = [
            _StackEntry(depth=0, node_id=root_id, child_heading_path=[])
        ]
        position_counters: dict[str, int] = {root_id: 0}

        def next_position(parent_id: str) -> int:
            pos = position_counters.get(parent_id, 0)
            position_counters[parent_id] = pos + 1
            return pos

        for layout_node, result in zip(nodes, results):
            if result.is_heading:
                # Pop back to the nearest ancestor with a strictly smaller depth.
                while stack[-1].depth >= result.depth:
                    stack.pop()
                parent = stack[-1]

                position = next_position(parent.node_id)
                heading_path = list(parent.child_heading_path)
                node_id = stable_id(
                    "node", doc_id, depth=result.depth, position=position,
                    heading_path=heading_path,
                )
                node_type = _HEADING_NODE_TYPES.get(result.depth, "subsection")

                out.append(DocumentNode(
                    id=node_id,
                    document_id=doc_id,
                    parent_id=parent.node_id,
                    node_type=node_type,
                    title=layout_node.text,
                    text="",
                    page_start=layout_node.page_number,
                    page_end=layout_node.page_number,
                    depth=result.depth,
                    position=position,
                    heading_path=heading_path,
                ))

                stack.append(_StackEntry(
                    depth=result.depth,
                    node_id=node_id,
                    child_heading_path=heading_path + [layout_node.text],
                ))
                position_counters.setdefault(node_id, 0)
            else:
                parent = stack[-1]
                for chunk in self._chunk_text(layout_node.text):
                    position = next_position(parent.node_id)
                    heading_path = list(parent.child_heading_path)
                    node_id = stable_id(
                        "node", doc_id, depth=parent.depth + 1, position=position,
                        heading_path=heading_path,
                    )
                    out.append(DocumentNode(
                        id=node_id,
                        document_id=doc_id,
                        parent_id=parent.node_id,
                        node_type="paragraph",
                        title=None,
                        text=chunk,
                        page_start=layout_node.page_number,
                        page_end=layout_node.page_number,
                        depth=parent.depth + 1,
                        position=position,
                        heading_path=heading_path,
                    ))

        return out

    # ── Headingless fallback: fixed-size page groups ────────────────────

    def _build_page_groups(
        self, doc_id: str, nodes: list[LayoutNode]
    ) -> list[DocumentNode]:
        max_page = max((n.page_number for n in nodes), default=1)
        root_id = stable_id("node", doc_id, depth=0, position=0, heading_path=[])
        out: list[DocumentNode] = [
            DocumentNode(
                id=root_id,
                document_id=doc_id,
                parent_id=None,
                node_type="document",
                title=None,
                text="",
                page_start=1,
                page_end=max_page,
                depth=0,
                position=0,
                heading_path=[],
            )
        ]

        group_count = (max_page + _PAGE_GROUP_SIZE - 1) // _PAGE_GROUP_SIZE
        for group_index in range(group_count):
            group_start = group_index * _PAGE_GROUP_SIZE + 1
            group_end = min((group_index + 1) * _PAGE_GROUP_SIZE, max_page)

            group_id = stable_id(
                "node", doc_id, depth=1, position=group_index, heading_path=[]
            )
            out.append(DocumentNode(
                id=group_id,
                document_id=doc_id,
                parent_id=root_id,
                node_type="section",
                title=None,
                text="",
                page_start=group_start,
                page_end=group_end,
                depth=1,
                position=group_index,
                heading_path=[],
            ))

            child_position = 0
            for layout_node in nodes:
                if not (group_start <= layout_node.page_number <= group_end):
                    continue
                for chunk in self._chunk_text(layout_node.text):
                    node_id = stable_id(
                        "node", doc_id, depth=2, position=child_position,
                        heading_path=[],
                    )
                    out.append(DocumentNode(
                        id=node_id,
                        document_id=doc_id,
                        parent_id=group_id,
                        node_type="paragraph",
                        title=None,
                        text=chunk,
                        page_start=layout_node.page_number,
                        page_end=layout_node.page_number,
                        depth=2,
                        position=child_position,
                        heading_path=[],
                    ))
                    child_position += 1

        return out

    # ── Shared helpers ───────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        """Split text at sentence boundaries if it exceeds hard limits.

        Falls back to fixed-size word windows when the text has no sentence
        punctuation to split on (or when a single "sentence" is itself over
        the hard cap), so the 500-word cap is always enforced.
        """
        words = _WORD_COUNT_RE.findall(text)
        if len(words) <= _MAX_PARAGRAPH_WORDS:
            return [text]

        sentences = _SENTENCE_RE.split(text)
        if len(sentences) <= 1:
            return [
                " ".join(words[i : i + _TARGET_PARAGRAPH_WORDS])
                for i in range(0, len(words), _TARGET_PARAGRAPH_WORDS)
            ]

        chunks: list[str] = []
        current: list[str] = []
        count = 0

        for sentence in sentences:
            sentence_words = _WORD_COUNT_RE.findall(sentence)
            sentence_len = len(sentence_words)

            if sentence_len > _MAX_PARAGRAPH_WORDS:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                    count = 0
                for i in range(0, sentence_len, _TARGET_PARAGRAPH_WORDS):
                    chunks.append(" ".join(sentence_words[i : i + _TARGET_PARAGRAPH_WORDS]))
                continue

            if count + sentence_len > _TARGET_PARAGRAPH_WORDS and current:
                chunks.append(" ".join(current))
                current = []
                count = 0
            current.append(sentence)
            count += sentence_len

        if current:
            chunks.append(" ".join(current))
        return chunks

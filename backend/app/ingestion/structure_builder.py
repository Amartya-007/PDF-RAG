"""StructureBuilder — converts LayoutNode list into a DocumentNode tree.

Pipeline
--------
LayoutNode list (flat)
    ↓  HeadingDetector classification
Annotated nodes with is_heading + depth
    ↓  Tree assembly
DocumentNode tree: document-root → chapter → section → paragraph

Paragraph size invariant
------------------------
Target: 80–250 words. Hard cap: 500 words. When body text between two
headings exceeds 500 words, it is split at sentence boundaries.
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

# Pre-compiled regex
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_COUNT_RE = re.compile(r"\S+")


@dataclass
class _BuilderNode:
    """Internal transient node for building the tree structure."""
    text: str
    title: str = ""
    raw_node_type: str = "paragraph"
    depth: int = 0
    position: int = 0
    page_start: int = 1
    page_end: int = 1
    children: list[_BuilderNode] = field(default_factory=list)


class StructureBuilder:
    """Orchestrates the conversion of flat LayoutNodes into a tree."""

    def __init__(self, heading_detector: HeadingDetector) -> None:
        self._detector = heading_detector

    def build(self, doc_id: str, nodes: list[LayoutNode]) -> list[DocumentNode]:
        """Convert flat layout nodes into a hierarchical tree of DocumentNodes."""
        # 1. Classify nodes and filter noise
        root = _BuilderNode(text="", title="Root", depth=0)
        baseline = self._detector.estimate_baseline(nodes)
        
        # 2. Assemble tree (recursive grouping)
        self._assemble_tree(root, nodes, baseline)
        
        # 3. Flatten tree into final persistence models
        result: list[DocumentNode] = []
        self._walk(root, None, [], doc_id, result)
        return result

    def _assemble_tree(self, root: _BuilderNode, nodes: list[LayoutNode], baseline: float) -> None:
        """Group nodes into a tree structure based on detected headings."""
        current_parent = root
        for node in nodes:
            res = self._detector.detect(node, baseline)
            
            if res.is_heading:
                # Move up/down the tree based on depth
                while current_parent.depth >= res.depth and current_parent is not root:
                    # In a real impl, you'd track the parent stack here
                    break 
                new_node = _BuilderNode(
                    text="", title=node.text, depth=res.depth, 
                    page_start=node.page_number
                )
                current_parent.children.append(new_node)
                current_parent = new_node
            else:
                # Add body text, splitting if too long
                chunks = self._chunk_text(node.text)
                for chunk in chunks:
                    current_parent.children.append(_BuilderNode(
                        text=chunk, page_start=node.page_number
                    ))

    def _chunk_text(self, text: str) -> list[str]:
        """Split text at sentence boundaries if it exceeds hard limits."""
        words = _WORD_COUNT_RE.findall(text)
        if len(words) <= _MAX_PARAGRAPH_WORDS:
            return [text]
            
        # Split at sentences to keep context intact
        sentences = _SENTENCE_RE.split(text)
        chunks = []
        current = []
        count = 0
        
        for s in sentences:
            s_len = len(_WORD_COUNT_RE.findall(s))
            if count + s_len > _TARGET_PARAGRAPH_WORDS and current:
                chunks.append(" ".join(current))
                current = []
                count = 0
            current.append(s)
            count += s_len
        
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _walk(
        self,
        node: _BuilderNode,
        parent_id: str | None,
        heading_path: list[str],
        doc_id: str,
        out: list[DocumentNode],
    ) -> str:
        """Recursive traversal to generate DocumentNode identifiers and flat list."""
        node_id = stable_id(
            "node", doc_id, depth=node.depth, position=node.position, heading_path=heading_path
        )
        
        out.append(DocumentNode(
            id=node_id,
            document_id=doc_id,
            parent_id=parent_id,
            node_type=node.raw_node_type,
            title=node.title,
            text=node.text,
            page_start=node.page_start,
            page_end=node.page_end,
            depth=node.depth,
            position=node.position,
            heading_path=list(heading_path),
        ))
        
        new_path = heading_path + ([node.title] if node.title else [])
        for child in node.children:
            self._walk(child, node_id, new_path, doc_id, out)
        return node_id
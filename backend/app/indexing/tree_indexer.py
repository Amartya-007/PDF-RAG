"""TreeIndexer — hierarchical JSON tree construction for vectorless RAG.

Parses document sections into a tree (title, page range, summary, children).
During retrieval, this tree acts as a context filter for the LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.generation.ollama_client import OllamaClient

# Pre-compiled regex for performance
_HEADING_RE = re.compile(r"^(?:\d+[\.\d]*\s+)?[A-Z][A-Za-z0-9 \-,:\'\"]{0,90}$")


@dataclass
class TreeNode:
    """A node in the document tree representing a section."""
    node_id: str
    title: str
    level: int
    page_start: int
    page_end: int
    raw_text: str
    summary: str = ""
    children: list[TreeNode] = field(default_factory=list)


@dataclass
class DocumentTree:
    """Root structure for a parsed document."""
    document_id: str
    filename: str
    nodes: list[TreeNode] = field(default_factory=list)

    def all_nodes(self) -> list[TreeNode]:
        """Flatten the tree into a list of all nodes."""
        nodes = []
        def _recurse(node: TreeNode):
            nodes.append(node)
            for child in node.children:
                _recurse(child)
        for root in self.nodes:
            _recurse(root)
        return nodes


class TreeIndexer:
    """Orchestrates tree building and LLM-based summarization."""

    def __init__(self, ollama: OllamaClient) -> None:
        self.ollama = ollama

    def index(self, tree: DocumentTree) -> None:
        """Run summarization across all nodes in the tree."""
        for node in tree.all_nodes():
            self._summarize_node(node)

    def _summarize_node(self, node: TreeNode) -> None:
        """Attempt LLM summary; fallback to heuristic on failure."""
        prompt = (
            f"Summarise this document section in 40-80 words. Be factual and concise.\n\n"
            f"Title: {node.title}\nText:\n{node.raw_text[:1200]}\n\nSummary:"
        )
        try:
            node.summary = self.ollama.generate(prompt).strip()[:400]
        except Exception:
            node.summary = self._heuristic_summary(node)

    @staticmethod
    def _heuristic_summary(node: TreeNode) -> str:
        """Simple fallback summary: first two sentences."""
        # Use a non-capturing group for the split to be safe
        sentences = re.split(r"(?<=[.!?])\s+", node.raw_text.strip())
        return " ".join(sentences[:2])[:300]

    @staticmethod
    def looks_like_heading(line: str) -> bool:
        """Heuristic check to determine if a line functions as a document heading."""
        line = line.strip()
        if not line or len(line) > 100:
            return False
        if line.endswith((".", "?", "!", ",", ";")):
            return False
        # Headings are usually short
        if not 1 <= len(line.split()) <= 14:
            return False
        return bool(_HEADING_RE.match(line))
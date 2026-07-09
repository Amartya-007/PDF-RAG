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
Target: 80–250 words.  Hard cap: 500 words.  When body text between two
headings exceeds 500 words, it is split at sentence boundaries into
multiple paragraph nodes with the same parent.

Headingless documents
---------------------
When no headings are detected, the structure is:
  document-root (depth 0)
    └── page-group (depth 1, 5 pages each)
          └── paragraph (depth 2)
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from backend.app.domain.models.node import DocumentNode, stable_id
from backend.app.ingestion.heading_detector import HeadingDetector, HeadingResult
from backend.app.ingestion.layout_parser import LayoutNode

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_COUNT_RE = re.compile(r"\S+")

_MAX_PARAGRAPH_WORDS = 500
_TARGET_PARAGRAPH_WORDS = 200
_PAGE_GROUP_SIZE = 5          # pages per group when no headings exist


def _word_count(text: str) -> int:
    return len(_WORD_COUNT_RE.findall(text))


def _split_at_sentences(text: str, max_words: int) -> list[str]:
    """Split *text* into chunks not exceeding *max_words* at sentence boundaries."""
    sentences = _SENTENCE_RE.split(text.strip())
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        w = _word_count(sentence)
        if w > max_words:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_words = 0
            chunks.extend(_split_words(sentence, max_words))
            continue
        if current_words + w > max_words and current:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += w

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


def _split_words(text: str, max_words: int) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
    ]


@dataclass
class _BuilderNode:
    """Transient node used during tree assembly before IDs are computed."""
    raw_node_type: str
    title: str | None
    text: str
    page_start: int
    page_end: int
    depth: int
    children: list["_BuilderNode"] = field(default_factory=list)
    position: int = 0
    heading_path: list[str] = field(default_factory=list)


class StructureBuilder:
    """Converts a flat list of ``LayoutNode`` objects into a ``DocumentNode`` tree.

    Args:
        detector:       Heading classification component. If ``None``, a fresh
                        ``HeadingDetector`` is created for each ``build`` call.
        document_id:    Parent document identifier used in ``stable_id`` calls.
    """

    def __init__(
        self,
        detector: HeadingDetector | None = None,
        document_id: str = "",
    ) -> None:
        self._detector = detector or HeadingDetector()
        self._document_id = document_id

    # ── Public API ─────────────────────────────────────────────────────────

    def build(
        self, layout_nodes: list[LayoutNode], document_id: str | None = None
    ) -> list[DocumentNode]:
        """Convert *layout_nodes* into a ``DocumentNode`` tree.

        Args:
            layout_nodes: Output of ``LayoutParser.parse``.
            document_id:  Override the document_id set in ``__init__``.

        Returns:
            Flat list of ``DocumentNode`` objects whose ``parent_id``
            references form a tree.  The first element is always the root.
        """
        doc_id = document_id or self._document_id or str(uuid.uuid4())
        if not layout_nodes:
            return []

        classifications = self._detector.detect(layout_nodes)

        # Check whether any headings were detected
        has_headings = any(r.is_heading for r in classifications)
        if has_headings:
            builder_nodes = self._build_headed(layout_nodes, classifications)
        else:
            builder_nodes = self._build_headingless(layout_nodes)

        return self._materialise(builder_nodes, doc_id)

    # ── Headed document assembly ───────────────────────────────────────────

    def _build_headed(
        self,
        layout_nodes: list[LayoutNode],
        classifications: list[HeadingResult],
    ) -> list[_BuilderNode]:
        """Assemble a tree from classified layout nodes."""
        root = _BuilderNode(
            raw_node_type="document",
            title=None,
            text="",
            page_start=layout_nodes[0].page_number,
            page_end=layout_nodes[-1].page_number,
            depth=0,
        )
        # Stack holds the open structural nodes at each depth level.
        # Index 0 = root; index N = the currently-open node at depth N.
        stack: list[_BuilderNode] = [root]
        pending_body: list[str] = []
        pending_start = layout_nodes[0].page_number
        pending_end = layout_nodes[0].page_number

        def flush_body() -> None:
            """Convert accumulated body text into paragraph children."""
            nonlocal pending_start, pending_end
            text = " ".join(pending_body).strip()
            if not text:
                pending_body.clear()
                return
            parent = stack[-1]
            for chunk in _split_at_sentences(text, _MAX_PARAGRAPH_WORDS):
                if chunk.strip():
                    para = _BuilderNode(
                        raw_node_type="paragraph",
                        title=None,
                        text=chunk,
                        page_start=pending_start,
                        page_end=pending_end,
                        depth=parent.depth + 1,
                    )
                    parent.children.append(para)
            pending_body.clear()

        for layout_node, result in zip(layout_nodes, classifications):
            pending_end = layout_node.page_number
            if result.is_heading:
                flush_body()
                pending_start = layout_node.page_number
                # Pop the stack back to the parent level for this heading depth
                target_depth = result.depth
                while len(stack) > 1 and stack[-1].depth >= target_depth:
                    stack.pop()
                parent = stack[-1]
                node_type = {1: "chapter", 2: "section"}.get(target_depth, "subsection")
                struct_node = _BuilderNode(
                    raw_node_type=node_type,
                    title=layout_node.text.strip(),
                    text=layout_node.text.strip(),
                    page_start=layout_node.page_number,
                    page_end=layout_node.page_number,
                    depth=target_depth,
                )
                parent.children.append(struct_node)
                stack.append(struct_node)
            else:
                if layout_node.text.strip():
                    pending_body.append(layout_node.text.strip())

        flush_body()
        return [root]

    # ── Headingless document assembly ──────────────────────────────────────

    def _build_headingless(self, layout_nodes: list[LayoutNode]) -> list[_BuilderNode]:
        """Fallback structure: document-root → page-groups → paragraphs."""
        pages: dict[int, list[str]] = {}
        for node in layout_nodes:
            pages.setdefault(node.page_number, []).append(node.text.strip())

        page_numbers = sorted(pages)
        root = _BuilderNode(
            raw_node_type="document",
            title=None,
            text="",
            page_start=page_numbers[0] if page_numbers else 1,
            page_end=page_numbers[-1] if page_numbers else 1,
            depth=0,
        )

        for group_idx in range(0, len(page_numbers), _PAGE_GROUP_SIZE):
            group_pages = page_numbers[group_idx : group_idx + _PAGE_GROUP_SIZE]
            group_start = group_pages[0]
            group_end = group_pages[-1]
            group_node = _BuilderNode(
                raw_node_type="section",
                title=f"Pages {group_start}–{group_end}",
                text="",
                page_start=group_start,
                page_end=group_end,
                depth=1,
            )
            for page_num in group_pages:
                page_text = " ".join(pages[page_num])
                for chunk in _split_at_sentences(page_text, _MAX_PARAGRAPH_WORDS):
                    if chunk.strip():
                        group_node.children.append(_BuilderNode(
                            raw_node_type="paragraph",
                            title=None,
                            text=chunk,
                            page_start=page_num,
                            page_end=page_num,
                            depth=2,
                        ))
            root.children.append(group_node)

        return [root]

    # ── Materialise: assign positions, heading_paths, stable IDs ──────────

    def _materialise(
        self, builder_roots: list[_BuilderNode], doc_id: str
    ) -> list[DocumentNode]:
        """Convert the mutable ``_BuilderNode`` tree into immutable ``DocumentNode`` objects."""
        result: list[DocumentNode] = []
        for root in builder_roots:
            self._assign_positions(root)
            self._walk(root, parent_id=None, heading_path=[], doc_id=doc_id, out=result)
        return result

    def _walk(
        self,
        node: _BuilderNode,
        parent_id: str | None,
        heading_path: list[str],
        doc_id: str,
        out: list[DocumentNode],
    ) -> str:
        node_id = stable_id(
            prefix="node",
            document_id=doc_id,
            depth=node.depth,
            position=node.position,
            heading_path=heading_path,
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
        child_heading_path = heading_path + ([node.title] if node.title else [])
        for child in node.children:
            self._walk(child, node_id, child_heading_path, doc_id, out)
        return node_id

    @staticmethod
    def _assign_positions(root: _BuilderNode) -> None:
        """Set sequential position values among siblings (depth-first)."""
        def _visit(parent: _BuilderNode) -> None:
            for i, child in enumerate(parent.children):
                child.position = i
                _visit(child)
        _visit(root)

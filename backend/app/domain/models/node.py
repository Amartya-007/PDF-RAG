"""DocumentNode — the primary hierarchical text unit for the RAG pipeline.

Each ingested document is decomposed into a DocumentNode tree:
  document-root (depth 0)
    └── chapter (depth 1)
          └── section (depth 2)
                └── paragraph / table / list (depth 3+)

Node IDs are deterministic so re-ingesting the same document produces
identical node IDs, enabling safe upserts without duplication.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


# Valid values for DocumentNode.node_type
NODE_TYPES = frozenset(
    {"document", "chapter", "section", "subsection", "paragraph", "table", "list"}
)


def stable_id(
    prefix: str,
    document_id: str,
    depth: int,
    position: int,
    heading_path: list[str],
) -> str:
    """Produce a deterministic, collision-resistant identifier.

    The ID is a truncated SHA-256 hex digest of the canonical JSON
    representation of the inputs.  Re-ingesting the same document with
    the same structure will always produce the same IDs.

    Args:
        prefix:       Namespace prefix, e.g. ``"node"``.
        document_id:  Parent document identifier.
        depth:        Hierarchy depth of the node (0 = document root).
        position:     Sequential index among siblings at the same depth.
        heading_path: Ordered list of ancestor heading titles.

    Returns:
        A string of the form ``"{prefix}_{32-char-hex}"``.
    """
    payload = json.dumps(
        [prefix, document_id, depth, position, heading_path],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:32]
    return f"{prefix}_{digest}"


@dataclass(slots=True)
class DocumentNode:
    """A hierarchical text unit extracted from a PDF document.

    Attributes:
        id:           Deterministic identifier produced by :func:`stable_id`.
        document_id:  Parent document identifier.
        parent_id:    ID of the parent node; ``None`` only for the document root.
        node_type:    Structural type — one of ``NODE_TYPES``.
        title:        Heading text, or ``None`` for leaf/paragraph nodes.
        text:         Full text content of this node.
        page_start:   First page covered by this node (1-based).
        page_end:     Last page covered by this node (1-based, inclusive).
        depth:        Level in the tree hierarchy (0 = document root).
        position:     Sequential index among siblings at the same depth.
        heading_path: Ordered ancestor heading titles from the document root
                      down to this node's immediate parent.
    """

    id: str
    document_id: str
    parent_id: str | None
    node_type: str
    title: str | None
    text: str
    page_start: int
    page_end: int
    depth: int
    position: int
    heading_path: list[str]

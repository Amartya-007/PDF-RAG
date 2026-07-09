from __future__ import annotations

from backend.app.ingestion.layout_parser import LayoutNode
from backend.app.ingestion.structure_builder import StructureBuilder


def test_structure_builder_creates_headed_parent_child_tree() -> None:
    nodes = StructureBuilder(document_id="doc1").build(
        [
            LayoutNode(text="1. Introduction", page_number=1),
            LayoutNode(text="Introductory body text.", page_number=1),
            LayoutNode(text="1.1 Scope", page_number=2),
            LayoutNode(text="Scope body text.", page_number=2),
        ]
    )

    root = nodes[0]
    chapter = next(node for node in nodes if node.node_type == "chapter")
    section = next(node for node in nodes if node.node_type == "section")
    paragraphs = [node for node in nodes if node.node_type == "paragraph"]

    assert root.parent_id is None
    assert chapter.parent_id == root.id
    assert section.parent_id == chapter.id
    assert section.heading_path == ["1. Introduction"]
    assert paragraphs[-1].parent_id == section.id
    assert paragraphs[-1].heading_path == ["1. Introduction", "1.1 Scope"]


def test_structure_builder_headingless_document_uses_five_page_groups() -> None:
    layout_nodes = [
        LayoutNode(text=f"This is page {page} body text.", page_number=page)
        for page in range(1, 7)
    ]

    nodes = StructureBuilder(document_id="doc1").build(layout_nodes)

    groups = [node for node in nodes if node.depth == 1]
    assert [(group.page_start, group.page_end) for group in groups] == [(1, 5), (6, 6)]
    assert all(group.node_type == "section" for group in groups)


def test_structure_builder_splits_paragraphs_at_hard_500_word_cap() -> None:
    long_text = " ".join(f"word{i}" for i in range(600))

    nodes = StructureBuilder(document_id="doc1").build(
        [LayoutNode(text=long_text, page_number=1)]
    )

    paragraphs = [node for node in nodes if node.node_type == "paragraph"]
    assert len(paragraphs) >= 2
    assert all(len(paragraph.text.split()) <= 500 for paragraph in paragraphs)

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from uuid import uuid4

from backend.app.ingestion.layout_parser import LayoutNode, LayoutParser
from backend.app.ingestion.parser.pdf_parser import PageText, PdfParser


def temp_file(name: str, text: str) -> Path:
    path = Path.cwd() / "backend" / ".test-tmp" / f"layout-{uuid4().hex}-{name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_layout_parser_plain_text_uses_default_visual_metadata() -> None:
    path = temp_file("plain.txt", "First paragraph.\n\nSecond paragraph.")

    nodes = LayoutParser().parse(path)

    assert nodes == [
        LayoutNode(text="First paragraph.", page_number=1),
        LayoutNode(text="Second paragraph.", page_number=1),
    ]


def test_extract_block_info_includes_font_bbox_indent_and_line_spacing() -> None:
    block = {
        "bbox": (72.0, 100.0, 300.0, 140.0),
        "lines": [
            {"bbox": (72.0, 100.0, 300.0, 110.0), "spans": [
                {"text": "Chapter ", "size": 16.0, "font": "Arial-Bold", "flags": 16},
            ]},
            {"bbox": (72.0, 124.0, 300.0, 134.0), "spans": [
                {"text": "One", "size": 14.0, "font": "Arial-Bold", "flags": 16},
            ]},
        ],
    }

    text, font_size, font_name, is_bold, bbox, indent, line_spacing = (
        LayoutParser._extract_block_info(block)
    )

    assert text == "Chapter One"
    assert font_size == 16.0
    assert font_name == "Arial-Bold"
    assert is_bold is True
    assert bbox == (72.0, 100.0, 300.0, 140.0)
    assert indent == 72.0
    assert line_spacing == 24.0


def test_pdf_with_no_pymupdf_text_falls_back_to_pdf_parser(
    monkeypatch,
) -> None:
    fake_pymupdf = ModuleType("pymupdf")
    fake_pymupdf.TEXT_PRESERVE_WHITESPACE = 0

    class FakePage:
        def get_text(self, *_args, **_kwargs) -> dict[str, list[dict]]:
            return {"blocks": []}

    class FakeDoc:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _index: int) -> FakePage:
            return FakePage()

        def close(self) -> None:
            pass

    fake_pymupdf.open = lambda _path: FakeDoc()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pymupdf", fake_pymupdf)
    monkeypatch.setattr(
        PdfParser,
        "parse",
        lambda self, path: [PageText(page_number=1, text="Docling fallback text.")],
    )
    path = temp_file("scanned.pdf", "")

    nodes = LayoutParser().parse(path)

    assert nodes == [LayoutNode(text="Docling fallback text.", page_number=1)]


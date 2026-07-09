from __future__ import annotations

from backend.app.ingestion.heading_detector import HeadingDetector
from backend.app.ingestion.layout_parser import LayoutNode


def classify(node: LayoutNode):
    return HeadingDetector(body_font_size_baseline=10.0).detect([node])[0]


def test_heading_detector_assigns_numbering_depths() -> None:
    detector = HeadingDetector()
    nodes = [
        LayoutNode(text="1. Introduction", page_number=1),
        LayoutNode(text="1.2 Recovery", page_number=1),
        LayoutNode(text="1.2.3 Log Records", page_number=1),
    ]

    results = detector.detect(nodes)

    assert [result.depth for result in results] == [1, 2, 3]
    assert all(result.is_heading for result in results)


def test_heading_detector_uses_font_size_bold_caps_and_indent_signals() -> None:
    assert classify(
        LayoutNode(text="Large Heading", page_number=1, font_size=13.0)
    ).is_heading
    assert classify(
        LayoutNode(text="Bold Heading", page_number=1, is_bold=True)
    ).is_heading
    assert classify(
        LayoutNode(text="SECTION IV", page_number=1, indent=0.0)
    ).is_heading


def test_heading_detector_uses_line_spacing_signal_for_plain_text_headings() -> None:
    result = classify(
        LayoutNode(
            text="Overview",
            page_number=1,
            indent=0.0,
            line_spacing=24.0,
        )
    )

    assert result.is_heading


def test_heading_detector_rejects_long_body_paragraph() -> None:
    result = classify(
        LayoutNode(
            text=(
                "This paragraph explains the transaction process in enough detail "
                "that it should not be mistaken for a heading by the detector."
            ),
            page_number=1,
            indent=24.0,
            line_spacing=12.0,
        )
    )

    assert not result.is_heading
    assert result.depth == 0


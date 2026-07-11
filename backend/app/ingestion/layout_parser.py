"""LayoutParser — PDF text extraction with layout metadata.

Produces ``LayoutNode`` objects enriched with font size, boldness, bounding
box, indentation, and line spacing so ``HeadingDetector`` can use visual
cues rather than pure text heuristics.

For born-digital PDFs via PyMuPDF: font-level metadata is extracted per
text block. For plain text / Markdown / scanned PDFs: visual fields
default to ``None`` / ``False`` / ``0.0``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.ingestion.parser.pdf_parser import PageText, PdfParser

# Pre-compiled regex for performance
_PARA_SPLIT_RE = re.compile(r"\n{2,}")


@dataclass(slots=True)
class LayoutNode:
    """A text block extracted from a document with optional layout metadata.

    Attributes:
        text:         Raw text content of this block.
        page_number:  Source page (1-based).
        font_size:    Point size of the dominant font in the block, or None.
        font_name:    Name of the dominant font, or None.
        is_bold:      True when the block is in a bold-weight font.
        bbox:         Bounding box (x0, y0, x1, y1) in PDF points, or None.
        indent:       Left-margin indentation (0.0 when unavailable).
        line_spacing: Vertical distance between lines (0.0 when unavailable).
    """
    text: str
    page_number: int
    font_size: float | None = None
    font_name: str | None = None
    is_bold: bool = False
    bbox: tuple[float, float, float, float] | None = None
    indent: float = 0.0
    line_spacing: float = 0.0


class LayoutParser:
    """Parses files into a sequence of structured LayoutNodes."""

    def parse(self, path: Path) -> list[LayoutNode]:
        """Entry point for parsing; routes by file extension.

        Attributes:
            path: Path to the document (PDF, TXT, or MD).
        """
        if path.suffix.lower() == ".pdf":
            return self._parse_pdf(path)
        return self._parse_plain(path)

    def _parse_pdf(self, path: Path) -> list[LayoutNode]:
        """Extract nodes using PyMuPDF for rich layout metadata.

        The import is deliberately local (not module-level) so the PDF
        backend can be swapped or mocked per-call, and so a missing/broken
        PyMuPDF install degrades to the Docling-based ``PdfParser`` instead
        of crashing ingestion.
        """
        try:
            import pymupdf
        except ImportError:
            return self._parse_with_docling_fallback(path)

        nodes: list[LayoutNode] = []
        try:
            with pymupdf.open(path) as doc:
                for page_index in range(len(doc)):
                    page = doc[page_index]
                    blocks = page.get_text("dict").get("blocks", [])
                    for block in blocks:
                        if block.get("type") != 0 or not block.get("lines"):
                            continue
                        (
                            text, font_size, font_name, is_bold, bbox, indent,
                            line_spacing,
                        ) = self._extract_block_info(block)
                        if not text:
                            continue
                        nodes.append(LayoutNode(
                            text=text,
                            page_number=page_index + 1,
                            font_size=font_size,
                            font_name=font_name,
                            is_bold=is_bold,
                            bbox=bbox,
                            indent=indent,
                            line_spacing=line_spacing,
                        ))
        except Exception:
            return self._parse_with_docling_fallback(path)

        if not nodes:
            return self._parse_with_docling_fallback(path)
        return nodes

    @staticmethod
    def _parse_with_docling_fallback(path: Path) -> list[LayoutNode]:
        """Fall back to the Docling-backed ``PdfParser`` (e.g. scanned PDFs)."""
        return [
            LayoutNode(text=page_text.text, page_number=page_text.page_number)
            for page_text in PdfParser().parse(path)
            if page_text.text.strip()
        ]

    @staticmethod
    def _extract_block_info(
        block: dict[str, Any],
    ) -> tuple[str, float | None, str | None, bool, tuple[float, float, float, float] | None, float, float]:
        """Extract text and visual metadata from a PyMuPDF ``dict``-mode block.

        Returns a tuple of (text, font_size, font_name, is_bold, bbox, indent,
        line_spacing). ``font_size`` is the largest span size in the block
        (the dominant/heading-relevant size); ``line_spacing`` is the vertical
        gap between the first two line baselines, or 0.0 for single-line
        blocks.
        """
        lines = block.get("lines", [])
        line_texts: list[str] = []
        sizes: list[float] = []
        names: list[str] = []
        is_bold = False

        for line in lines:
            spans = line.get("spans", [])
            line_texts.append("".join(span.get("text", "") for span in spans))
            for span in spans:
                if span.get("size") is not None:
                    sizes.append(span["size"])
                font_name = span.get("font") or ""
                if font_name:
                    names.append(font_name)
                if (span.get("flags", 0) & 16) or "bold" in font_name.lower():
                    is_bold = True

        text = "".join(line_texts).strip()
        font_size = max(sizes) if sizes else None
        font_name = names[0] if names else None
        bbox = tuple(block["bbox"]) if block.get("bbox") else None
        indent = bbox[0] if bbox else 0.0

        line_spacing = 0.0
        if len(lines) >= 2:
            first_y = lines[0].get("bbox", (0.0, 0.0, 0.0, 0.0))[1]
            second_y = lines[1].get("bbox", (0.0, 0.0, 0.0, 0.0))[1]
            line_spacing = second_y - first_y

        return text, font_size, font_name, is_bold, bbox, indent, line_spacing

    def _parse_plain(self, path: Path) -> list[LayoutNode]:
        """Fallback for plain text / Markdown documents."""
        text = path.read_text(encoding="utf-8", errors="replace")
        pages = self._split_into_pages(text)
        
        nodes: list[LayoutNode] = []
        for i, page_text in enumerate(pages, start=1):
            for para in _PARA_SPLIT_RE.split(page_text):
                if stripped := para.strip():
                    nodes.append(LayoutNode(text=stripped, page_number=i))
        return nodes

    @staticmethod
    def _split_into_pages(text: str, lines_per_page: int = 50) -> list[str]:
        """Split plain text into artificial pages for consistent indexing."""
        lines = text.splitlines(keepends=True)
        return [
            "".join(lines[i : i + lines_per_page]) 
            for i in range(0, len(lines), lines_per_page)
        ] or [""]
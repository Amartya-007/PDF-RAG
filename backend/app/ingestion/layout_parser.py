"""LayoutParser — PDF text extraction with layout metadata.

Produces ``LayoutNode`` objects enriched with font size, boldness, bounding
box, indentation, and line spacing so ``HeadingDetector`` can use visual
cues rather than pure text heuristics.

For born-digital PDFs via PyMuPDF: font-level metadata is extracted per
text block.  For plain text / Markdown / scanned PDFs: visual fields
default to ``None`` / ``False`` / ``0.0``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from backend.app.ingestion.parser.pdf_parser import PageText


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
    """Converts a file path into a list of ``LayoutNode`` objects.

    Tries PyMuPDF for born-digital PDFs; falls back to ``PdfParser``
    output (text-only, no visual metadata) for plain text or scanned PDFs.
    """

    def parse(self, path: Path) -> list[LayoutNode]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._parse_pdf(path)
        # Plain text or Markdown: no visual metadata available
        return self._parse_plain(path)

    # ── PDF path ───────────────────────────────────────────────────────────

    def _parse_pdf(self, path: Path) -> list[LayoutNode]:
        try:
            import pymupdf  # type: ignore
            return self._parse_with_pymupdf(path, pymupdf)
        except ImportError:
            pass
        # Fallback: use existing PdfParser (text only)
        from backend.app.ingestion.parser.pdf_parser import PdfParser
        pages = PdfParser().parse(path)
        return self._pages_to_layout_nodes(pages)

    def _parse_with_pymupdf(self, path: Path, pymupdf) -> list[LayoutNode]:  # type: ignore[no-untyped-def]
        nodes: list[LayoutNode] = []
        doc = pymupdf.open(str(path))
        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_number = page_idx + 1
                # Extract blocks with dict metadata (font, bbox, etc.)
                block_dict = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
                for block in block_dict.get("blocks", []):
                    if block.get("type") != 0:   # type 0 = text block
                        continue
                    block_text, font_size, font_name, is_bold, bbox, indent = (
                        self._extract_block_info(block)
                    )
                    if not block_text.strip():
                        continue
                    nodes.append(LayoutNode(
                        text=block_text,
                        page_number=page_number,
                        font_size=font_size,
                        font_name=font_name,
                        is_bold=is_bold,
                        bbox=bbox,
                        indent=indent,
                        line_spacing=0.0,
                    ))
        finally:
            doc.close()
        return nodes

    @staticmethod
    def _extract_block_info(block: dict) -> tuple[
        str, float | None, str | None, bool, tuple | None, float
    ]:
        """Pull text and dominant font attributes from a PyMuPDF block dict."""
        lines = block.get("lines", [])
        all_text: list[str] = []
        sizes: list[float] = []
        fonts: list[str] = []
        bold_flags: list[bool] = []

        for line in lines:
            for span in line.get("spans", []):
                t = span.get("text", "")
                if t:
                    all_text.append(t)
                    sizes.append(span.get("size", 0.0))
                    fonts.append(span.get("font", ""))
                    # Bit 4 (value 16) in PyMuPDF flags = bold
                    bold_flags.append(bool(span.get("flags", 0) & 16))

        text = "".join(all_text).strip()
        font_size = max(sizes) if sizes else None
        # Dominant font: the one that appears most often
        font_name = max(set(fonts), key=fonts.count) if fonts else None
        is_bold = bool(sum(bold_flags) > len(bold_flags) / 2) if bold_flags else False

        raw_bbox = block.get("bbox")
        bbox = tuple(raw_bbox) if raw_bbox and len(raw_bbox) == 4 else None
        indent = float(raw_bbox[0]) if raw_bbox else 0.0

        return text, font_size, font_name, is_bold, bbox, indent

    # ── Plain text / Markdown path ─────────────────────────────────────────

    def _parse_plain(self, path: Path) -> list[LayoutNode]:
        text = path.read_text(encoding="utf-8", errors="replace")
        pages = self._split_into_pages(text)
        nodes: list[LayoutNode] = []
        for page_number, page_text in enumerate(pages, start=1):
            for para in self._split_paragraphs(page_text):
                if para.strip():
                    nodes.append(LayoutNode(
                        text=para,
                        page_number=page_number,
                    ))
        return nodes

    @staticmethod
    def _split_into_pages(text: str, lines_per_page: int = 50) -> list[str]:
        lines = text.splitlines(keepends=True)
        pages: list[str] = []
        for i in range(0, len(lines), lines_per_page):
            pages.append("".join(lines[i : i + lines_per_page]))
        return pages or [""]

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    @staticmethod
    def _pages_to_layout_nodes(pages: list[PageText]) -> list[LayoutNode]:
        nodes: list[LayoutNode] = []
        for page in pages:
            for para in re.split(r"\n{2,}", page.text):
                if para.strip():
                    nodes.append(LayoutNode(text=para.strip(), page_number=page.page_number))
        return nodes

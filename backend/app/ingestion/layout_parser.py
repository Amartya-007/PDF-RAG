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

# Optional PDF support
try:
    import fitz  # PyMuPDF
    _HAS_PDF = True
except ImportError:
    _HAS_PDF = False

from backend.app.ingestion.parser.pdf_parser import PageText

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
        if path.suffix.lower() == ".pdf" and _HAS_PDF:
            return self._parse_pdf(path)
        return self._parse_plain(path)

    def _parse_pdf(self, path: Path) -> list[LayoutNode]:
        """Extract nodes using PyMuPDF (fitz) for rich layout metadata."""
        nodes: list[LayoutNode] = []
        with fitz.open(path) as doc:
            for page in doc:
                blocks = page.get_text("dict")["blocks"]
                for b in blocks:
                    if b.get("type") == 0:  # Text block
                        text = "".join(l["text"] for l in b["lines"])
                        # Extract first line metadata for heading detection
                        line0 = b["lines"][0]["spans"][0]
                        nodes.append(LayoutNode(
                            text=text.strip(),
                            page_number=page.number + 1,
                            font_size=line0.get("size"),
                            font_name=line0.get("font"),
                            is_bold="bold" in line0.get("font", "").lower(),
                            bbox=b["bbox"],
                            indent=b["bbox"][0],
                        ))
        return nodes

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
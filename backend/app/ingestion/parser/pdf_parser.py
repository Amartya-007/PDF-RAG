from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from backend.app.domain.exceptions import ParseError, UnsupportedFileTypeError
from backend.app.models import PageText


class PdfParser:
    def __init__(self, force_ocr: bool = False) -> None:
        self.force_ocr = force_ocr

    def parse(self, path: Path) -> list[PageText]:
        """Parse *path* and return one PageText per logical page.

        Raises:
            UnsupportedFileTypeError: for non-PDF/txt/md files.
            ParseError: when no usable text can be extracted.
        """
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return [PageText(page_number=1, text=path.read_text(encoding="utf-8"))]
        if suffix != ".pdf":
            raise UnsupportedFileTypeError(path.suffix)
        return self._parse_pdf(path)

    def _parse_pdf(self, path: Path) -> list[PageText]:
        if self.force_ocr:
            docling_pages = self._try_docling(path)
            if docling_pages:
                return docling_pages

        # PyMuPDF first: it is 5-20x faster than Docling for normal, born-digital
        # PDFs because it just reads the embedded text layer instead of running
        # layout analysis / OCR models. Docling is only worth its cost on
        # scanned or image-only PDFs where there is no text layer to read.
        pymupdf_pages = self._try_pymupdf(path)
        if pymupdf_pages and self._has_sufficient_text(pymupdf_pages):
            return pymupdf_pages

        # Fall back to Docling for scanned/complex PDFs (it can OCR), or if
        # PyMuPDF isn't installed at all.
        docling_pages = self._try_docling(path)
        if docling_pages:
            return docling_pages

        if pymupdf_pages:
            # PyMuPDF got *some* text but it looked too sparse, and Docling
            # wasn't available/usable. Better to return what we have than
            # fail outright.
            return pymupdf_pages

        raise ParseError(
            "PDF parsing is installed, but no searchable text was extracted from this PDF. "
            "If this is a scanned/image-only PDF, install Docling/OCR support with "
            "`py -m pip install -e .[pdf]` or import a text-searchable PDF."
        )

    @staticmethod
    def _has_sufficient_text(pages: list[PageText], min_chars_per_page: int = 20) -> bool:
        """Heuristic: does this look like a real digital text layer, or just
        sparse noise (page numbers, headers) from a scanned PDF that PyMuPDF
        partially picked up? If sparse, prefer Docling's OCR path instead."""
        if not pages:
            return False
        total_chars = sum(len(page.text.strip()) for page in pages)
        avg_chars = total_chars / len(pages)
        return avg_chars >= min_chars_per_page

    def _try_docling(self, path: Path) -> list[PageText]:
        try:
            from docling.document_converter import DocumentConverter  # type: ignore
        except Exception:
            return []

        converter = DocumentConverter()
        result = converter.convert(str(path))
        markdown = result.document.export_to_markdown()
        if markdown.strip():
            return [PageText(page_number=1, text=markdown, section_path=(path.stem,))]
        return []

    def _try_pymupdf(self, path: Path) -> list[PageText]:
        fitz = self._load_pymupdf()

        pages: list[PageText] = []
        with fitz.open(path) as doc:
            for index, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text.strip():
                    pages.append(PageText(page_number=index, text=text, section_path=(path.stem,)))
        return pages

    def _load_pymupdf(self) -> ModuleType:
        errors: list[str] = []
        for module_name in ("pymupdf", "fitz"):
            try:
                return importlib.import_module(module_name)
            except Exception as exc:
                errors.append(f"{module_name}: {exc}")
        raise ParseError(
            "PDF parsing requires PyMuPDF, but it could not be imported.\n\n"
            "Fix:\n"
            "  py -m pip install -e .[desktop]\n\n"
            "Details:\n"
            + "\n".join(errors)
        )

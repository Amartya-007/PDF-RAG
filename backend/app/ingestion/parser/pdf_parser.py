from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from backend.app.models import PageText


class ParseError(RuntimeError):
    pass


class PdfParser:
    def parse(self, path: Path) -> list[PageText]:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return [PageText(page_number=1, text=path.read_text(encoding="utf-8"))]
        if suffix != ".pdf":
            raise ParseError(f"Unsupported file type: {path.suffix}")
        return self._parse_pdf(path)

    def _parse_pdf(self, path: Path) -> list[PageText]:
        docling_pages = self._try_docling(path)
        if docling_pages:
            return docling_pages

        pymupdf_pages = self._try_pymupdf(path)
        if pymupdf_pages:
            return pymupdf_pages

        raise ParseError(
            "PDF parsing is installed, but no searchable text was extracted from this PDF. "
            "If this is a scanned/image-only PDF, install Docling/OCR support with "
            "`py -m pip install -e .[pdf]` or import a text-searchable PDF."
        )

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

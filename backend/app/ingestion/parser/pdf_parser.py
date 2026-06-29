from __future__ import annotations

from pathlib import Path

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
            "PDF parsing requires Docling or PyMuPDF. Install optional dependencies with "
            "`py -m pip install -e .[pdf]`."
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
        try:
            import fitz  # type: ignore
        except Exception:
            return []

        pages: list[PageText] = []
        with fitz.open(path) as doc:
            for index, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text.strip():
                    pages.append(PageText(page_number=index, text=text, section_path=(path.stem,)))
        return pages

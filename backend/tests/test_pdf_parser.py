from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from backend.app.ingestion.parser.pdf_parser import PdfParser


class PdfParserTests(unittest.TestCase):
    def test_pymupdf_extracts_text_pdf(self) -> None:
        import pymupdf

        temp_dir = Path.cwd() / "backend" / ".test-tmp" / f"pdf-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = temp_dir / "sample.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Amartya Vishwakarma")
        doc.save(pdf_path)
        doc.close()

        pages = PdfParser().parse(pdf_path)

        self.assertEqual(len(pages), 1)
        self.assertIn("Amartya Vishwakarma", pages[0].text)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from backend.app.ingestion.chunking import Chunker
from backend.app.models import Document, PageText


class ChunkingTests(unittest.TestCase):
    def test_chunker_preserves_document_metadata(self) -> None:
        document = Document("doc_1", "policy.txt", "hash", "policy.txt")
        pages = [PageText(1, "Earned leave may be carried forward.\n\nMaximum balance is 30 days.")]

        chunks = Chunker(target_words=8, overlap_words=0).chunk_pages(document, pages)

        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0].document_id, "doc_1")
        self.assertEqual(chunks[0].filename, "policy.txt")
        self.assertEqual(chunks[0].page_start, 1)


if __name__ == "__main__":
    unittest.main()

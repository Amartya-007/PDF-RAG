from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from backend.app.indexing.sparse import BM25Index
from backend.app.models import Chunk


class SparseTests(unittest.TestCase):
    def test_bm25_finds_exact_terms(self) -> None:
        temp_dir = Path.cwd() / "backend" / ".test-tmp" / f"sparse-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        index = BM25Index(temp_dir / "bm25.json")
        chunks = [
            Chunk("c1", "d1", "a.txt", 1, 1, (), "Revenue includes settled payments."),
            Chunk("c2", "d1", "a.txt", 2, 2, (), "Refunds reduce final totals."),
        ]
        index.build(chunks)

        results = index.search("settled payments", top_k=2)

        self.assertEqual(results[0][0], "c1")


if __name__ == "__main__":
    unittest.main()

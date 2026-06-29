from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.rag_service import RagService


class RagServiceTests(unittest.TestCase):
    def test_text_ingest_and_ask_uses_citations(self) -> None:
        temp_dir = Path.cwd() / "backend" / ".test-tmp" / f"rag-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["RAG_DATA_DIR"] = str(temp_dir / "data")
        os.environ["RAG_SQLITE_PATH"] = ":memory:"
        os.environ["RAG_USE_OLLAMA"] = "0"
        path = temp_dir / "policy.txt"
        path.write_text(
            "Leave Policy\n\nEmployees may carry forward up to 30 days of earned leave.",
            encoding="utf-8",
        )

        service = RagService(get_settings())
        service.ingest(path, build_okf=False)
        answer = service.ask("How many earned leave days can employees carry forward?")

        self.assertTrue(answer.answerable)
        self.assertIn("[S1]", answer.answer)
        self.assertEqual(answer.citations[0].filename, "policy.txt")
        service.close()


if __name__ == "__main__":
    unittest.main()

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

    def test_reimport_failed_duplicate_rebuilds_chunks(self) -> None:
        temp_dir = Path.cwd() / "backend" / ".test-tmp" / f"rag-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["RAG_DATA_DIR"] = str(temp_dir / "data")
        os.environ["RAG_SQLITE_PATH"] = ":memory:"
        os.environ["RAG_USE_OLLAMA"] = "0"
        path = temp_dir / "resume.txt"
        path.write_text("Name: Amartya Vishwakarma\nCGPA: 8.5", encoding="utf-8")

        service = RagService(get_settings())
        document = service.ingest(path, build_okf=False)
        service.store.replace_chunks(document.document_id, [])
        service.store.update_document_status(document.document_id, "failed")

        rebuilt = service.ingest(path, build_okf=False)
        answer = service.ask("What is the name?")

        self.assertEqual(rebuilt.status, "ready")
        self.assertGreater(service.store.count_chunks_for_document(document.document_id), 0)
        self.assertTrue(answer.answerable)
        service.close()

    def test_stale_processing_documents_mark_failed_on_startup(self) -> None:
        temp_dir = Path.cwd() / "backend" / ".test-tmp" / f"rag-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["RAG_DATA_DIR"] = str(temp_dir / "data")
        os.environ["RAG_SQLITE_PATH"] = ":memory:"
        os.environ["RAG_USE_OLLAMA"] = "0"
        path = temp_dir / "stale.txt"
        path.write_text("Stale document text.", encoding="utf-8")

        service = RagService(get_settings())
        document = service.ingest(path, build_okf=False)
        service.store.replace_chunks(document.document_id, [])
        service.store.update_document_status(document.document_id, "processing")
        service.store.mark_stale_processing_documents_failed()

        documents = service.store.list_documents()

        self.assertEqual(documents[0].status, "failed")
        service.close()

    def test_fast_fact_question_skips_query_embedding(self) -> None:
        temp_dir = Path.cwd() / "backend" / ".test-tmp" / f"rag-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["RAG_DATA_DIR"] = str(temp_dir / "data")
        os.environ["RAG_SQLITE_PATH"] = ":memory:"
        os.environ["RAG_USE_OLLAMA"] = "0"
        path = temp_dir / "resume.txt"
        path.write_text(
            "Amartya Vishwakarma BTECH(CSE), CGPA: 8.09 "
            "2021 - 2025 Shri Ram Institute of Science & Technology Jabalpur, MP",
            encoding="utf-8",
        )

        class BrokenEmbedder:
            def embed(self, texts: list[str]) -> list[list[float]]:
                raise AssertionError("fast fact queries should not call embeddings")

        service = RagService(get_settings())
        service.ingest(path, build_okf=False)
        service.embedder = BrokenEmbedder()  # type: ignore[assignment]
        answer = service.ask("whats amartya collage name?", include_debug=True)

        self.assertTrue(answer.debug["fast_fact_query"])
        self.assertIn("Shri Ram Institute of Science & Technology", answer.answer)
        service.close()


if __name__ == "__main__":
    unittest.main()

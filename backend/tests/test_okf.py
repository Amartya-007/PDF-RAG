from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.knowledge.okf import parse_okf_markdown, validate_okf_bundle
from backend.app.rag_service import RagService


class OkfTests(unittest.TestCase):
    def test_generated_okf_has_indexes_links_and_metadata(self) -> None:
        service, temp_dir = self._service("generate")
        path = temp_dir / "policy.txt"
        path.write_text(
            "Revenue Policy\n\nRevenue includes settled payments. "
            "Refunds reduce revenue. Payments belong to orders.",
            encoding="utf-8",
        )

        service.ingest(path, build_okf=True)

        self.assertTrue((service.settings.okf_dir / "index.md").exists())
        self.assertTrue((service.settings.okf_dir / "concepts" / "index.md").exists())
        concept_files = [
            path
            for path in (service.settings.okf_dir / "concepts").glob("*.md")
            if path.name != "index.md"
        ]
        self.assertGreater(len(concept_files), 1)

        parsed = parse_okf_markdown(concept_files[0])
        self.assertEqual(parsed.metadata["type"], "concept")
        self.assertIn("source_chunk_ids", parsed.metadata)
        self.assertIn("related", parsed.metadata)

        issues = validate_okf_bundle(service.settings.okf_dir)
        errors = [issue for issue in issues if issue.severity == "error"]
        self.assertEqual(errors, [])
        service.close()

    def test_generated_okf_slugs_are_document_scoped(self) -> None:
        service, temp_dir = self._service("duplicate-slugs")
        first = temp_dir / "first.txt"
        second = temp_dir / "second.txt"
        text = (
            "Resume Profile\n\n"
            "Python developer with backend APIs, projects, education, and technical skills. "
            "Python backend APIs projects education skills."
        )
        first.write_text(text, encoding="utf-8")
        second.write_text(text.replace("Python", "Python language"), encoding="utf-8")

        service.ingest(first, build_okf=True)
        service.ingest(second, build_okf=True)

        slugs = [concept.slug for concept in service.store.list_concepts()]
        self.assertEqual(len(slugs), len(set(slugs)))
        self.assertTrue(any(slug.startswith("first-") for slug in slugs))
        self.assertTrue(any(slug.startswith("second-") for slug in slugs))
        service.close()

    def test_imported_okf_expands_to_source_chunks_during_retrieval(self) -> None:
        service, temp_dir = self._service("import")
        source = temp_dir / "source.txt"
        source.write_text(
            "Revenue includes settled payments. Refunds reduce revenue.",
            encoding="utf-8",
        )
        service.ingest(source, build_okf=False)
        chunk_id = service.store.list_chunks()[0].chunk_id

        bundle = temp_dir / "bundle"
        concepts = bundle / "concepts"
        concepts.mkdir(parents=True, exist_ok=True)
        (concepts / "revenue-calculation.md").write_text(
            f"""---
id: revenue-calculation
type: concept
title: Revenue Calculation
slug: revenue-calculation
aliases:
  - Sales calculation
tags:
  - revenue
related:
  - refunds
depends_on:
  - payments
verification_status: verified
source_chunk_ids:
  - {chunk_id}
---

# Revenue Calculation

Revenue calculation depends on settled payments and refunds.
""",
            encoding="utf-8",
        )

        imported = service.import_okf_bundle(bundle)
        self.assertEqual(len(imported), 1)

        chunks, debug = service.retrieve("How is sales calculation affected by refunds?", True)

        self.assertEqual(chunks[0].chunk_id, chunk_id)
        self.assertTrue(debug["okf_concept_results"])
        self.assertTrue(debug["okf_source_results"])
        service.close()

    def test_okf_validation_requires_type(self) -> None:
        temp_dir = self._temp_dir("validate")
        bundle = temp_dir / "bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "bad.md").write_text(
            """---
title: Missing Type
---

# Missing Type
""",
            encoding="utf-8",
        )

        issues = validate_okf_bundle(bundle)

        self.assertTrue(any(issue.severity == "error" for issue in issues))

    def _service(self, name: str) -> tuple[RagService, Path]:
        temp_dir = self._temp_dir(name)
        os.environ["RAG_DATA_DIR"] = str(temp_dir / "data")
        os.environ["RAG_SQLITE_PATH"] = ":memory:"
        os.environ["RAG_USE_OLLAMA"] = "0"
        return RagService(get_settings()), temp_dir

    def _temp_dir(self, name: str) -> Path:
        temp_dir = Path.cwd() / "backend" / ".test-tmp" / f"okf-{name}-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir


if __name__ == "__main__":
    unittest.main()

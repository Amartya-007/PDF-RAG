from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from backend.app.core.config import get_settings
from desktop.controller import DesktopController


class DesktopControllerTests(unittest.TestCase):
    def test_controller_status_ingest_and_ask(self) -> None:
        temp_dir = self._temp_dir()
        os.environ["RAG_DATA_DIR"] = str(temp_dir / "data")
        os.environ["RAG_SQLITE_PATH"] = ":memory:"
        os.environ["RAG_USE_OLLAMA"] = "0"
        path = temp_dir / "policy.txt"
        path.write_text("Refunds reduce revenue. Settled payments count as revenue.", encoding="utf-8")

        controller = DesktopController(get_settings())
        document = controller.ingest(path, build_okf=False)
        answer = controller.ask("What reduces revenue?")
        status = controller.status()

        self.assertEqual(document.filename, "policy.txt")
        self.assertTrue(answer.answerable)
        self.assertEqual(status["documents"], 1)
        self.assertEqual(status["chunks"], 1)
        self.assertTrue(status["ollama_ready"])
        controller.close()

    def test_controller_updates_model_preferences(self) -> None:
        temp_dir = self._temp_dir()
        os.environ["RAG_DATA_DIR"] = str(temp_dir / "data")
        os.environ["RAG_SQLITE_PATH"] = ":memory:"
        os.environ["RAG_USE_OLLAMA"] = "0"

        with patch.dict(os.environ, {"LOCALAPPDATA": str(temp_dir)}, clear=False):
            controller = DesktopController(get_settings())
            controller.update_preferences(
                use_ollama=True,
                ollama_base_url="http://localhost:11434",
                active_model="qwen3.5:9b",
                embedding_model="qwen3-embedding:4b",
            )

            self.assertTrue(controller.settings.use_ollama)
            self.assertEqual(controller.settings.active_model, "qwen3.5:9b")
            controller.close()

    def _temp_dir(self) -> Path:
        temp_dir = Path.cwd() / "backend" / ".test-tmp" / f"desktop-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir


if __name__ == "__main__":
    unittest.main()

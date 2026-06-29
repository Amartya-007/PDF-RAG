from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from desktop.preferences import (
    DesktopPreferences,
    apply_preferences,
    load_preferences,
    preferences_path,
    save_preferences,
)


class DesktopPreferencesTests(unittest.TestCase):
    def test_save_load_and_apply_preferences(self) -> None:
        temp_dir = Path.cwd() / "backend" / ".test-tmp" / f"prefs-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        preferences = DesktopPreferences(
            use_ollama=True,
            ollama_base_url="http://localhost:11434",
            active_model="custom-answer:latest",
            embedding_model="custom-embed:latest",
        )

        with patch.dict(os.environ, {"LOCALAPPDATA": str(temp_dir)}, clear=False):
            save_preferences(preferences)
            loaded = load_preferences()
            apply_preferences(loaded)

            self.assertTrue(preferences_path().exists())
            self.assertEqual(os.environ["RAG_USE_OLLAMA"], "1")
            self.assertEqual(os.environ["RAG_ACTIVE_MODEL"], "custom-answer:latest")
            self.assertEqual(os.environ["RAG_EMBEDDING_MODEL"], "custom-embed:latest")

        self.assertEqual(loaded, preferences)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from dataclasses import replace

from backend.app.core.config import get_settings
from desktop.model_readiness import ModelReadinessChecker


class FakeChecker(ModelReadinessChecker):
    def __init__(self, available: list[str] | None = None, fail: bool = False) -> None:
        super().__init__()
        self.available = available or []
        self.fail = fail

    def _ollama_models(self, base_url: str) -> list[str]:
        if self.fail:
            raise OSError("offline")
        return self.available


class ModelReadinessTests(unittest.TestCase):
    def test_disabled_ollama_is_ready_with_fallbacks(self) -> None:
        settings = replace(get_settings(), use_ollama=False)

        readiness = FakeChecker().check(settings)

        self.assertTrue(readiness.ready)
        self.assertFalse(readiness.ollama_required)

    def test_missing_ollama_returns_setup_commands(self) -> None:
        settings = replace(get_settings(), use_ollama=True)

        readiness = FakeChecker(fail=True).check(settings)

        self.assertFalse(readiness.ready)
        self.assertIn("ollama serve", readiness.setup_commands)

    def test_missing_models_are_reported(self) -> None:
        settings = replace(get_settings(), use_ollama=True)

        readiness = FakeChecker(available=["qwen3.5:4b"]).check(settings)

        self.assertFalse(readiness.ready)
        self.assertIn(settings.embedding_model, readiness.missing_models)

    def test_model_tags_must_match_when_required_has_tag(self) -> None:
        settings = replace(get_settings(), use_ollama=True, active_model="qwen3.5:4b")

        readiness = FakeChecker(available=["qwen3.5:9b", settings.embedding_model]).check(settings)

        self.assertFalse(readiness.ready)
        self.assertIn("qwen3.5:4b", readiness.missing_models)


if __name__ == "__main__":
    unittest.main()

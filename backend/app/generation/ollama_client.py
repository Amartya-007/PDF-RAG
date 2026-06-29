from __future__ import annotations

import json
import urllib.error
import urllib.request

from backend.app.core.config import Settings


class GenerationError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.settings.active_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.settings.temperature},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.settings.ollama_base_url.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise GenerationError(f"Ollama generation failed: {exc}") from exc

        answer = data.get("response")
        if not isinstance(answer, str):
            raise GenerationError("Ollama response did not include generated text.")
        return answer.strip()

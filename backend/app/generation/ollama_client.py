"""Ollama generation client using the official SDK.

Uses `ollama.Client` which maintains a persistent httpx connection pool,
so repeated calls within a session avoid TCP handshake overhead.
Falls back to raw urllib if the SDK is not installed.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from backend.app.core.config import Settings
from backend.app.domain.exceptions import GenerationError

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: object = None  # lazy ollama.Client

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import ollama  # type: ignore
                self._client = ollama.Client(host=self.settings.ollama_base_url)
                logger.debug("Using ollama SDK client")
            except ImportError:
                self._client = False
                logger.debug("ollama SDK not available; using urllib fallback")
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        if client:
            return self._generate_sdk(client, prompt)
        return self._generate_urllib(prompt)

    def _generate_sdk(self, client: object, prompt: str) -> str:
        try:
            response = client.generate(  # type: ignore[union-attr]
                model=self.settings.active_model,
                prompt=prompt,
                stream=False,
                keep_alive=600,  # 10 min keep-alive in memory
                options={
                    "temperature": self.settings.temperature,
                    "num_ctx": 3072,    # evidence already trimmed to ~90w × 4 sources
                    "num_predict": 384, # ~300 word cap, fast on limited VRAM
                },
            )
            text = response.response if hasattr(response, "response") else response.get("response", "")
            if not isinstance(text, str):
                raise GenerationError("Ollama SDK response missing text")
            return text.strip()
        except Exception as exc:
            if "GenerationError" in type(exc).__name__:
                raise
            raise GenerationError(f"Ollama SDK generate failed: {exc}") from exc

    def _generate_urllib(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.settings.active_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": self.settings.temperature,
                "num_ctx": 3072,
                "num_predict": 384,
            },
        }).encode("utf-8")
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
            raise GenerationError(f"Ollama urllib generate failed: {exc}") from exc
        answer = data.get("response")
        if not isinstance(answer, str):
            raise GenerationError("Ollama response did not include generated text.")
        return answer.strip()

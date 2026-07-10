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
from typing import TYPE_CHECKING, Any

from backend.app.core.config import Settings
from backend.app.domain.exceptions import GenerationError

if TYPE_CHECKING:
    import ollama

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for generating text using the Ollama API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: ollama.Client | None = None
        self._sdk_available: bool | None = None

    def _get_client(self) -> ollama.Client | None:
        """Lazily initialize the Ollama SDK client."""
        if self._sdk_available is False:
            return None
        
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.settings.ollama_base_url)
                self._sdk_available = True
                logger.debug("Using ollama SDK client")
            except ImportError:
                self._sdk_available = False
                logger.debug("ollama SDK not available; using urllib fallback")
        
        return self._client

    def generate(self, prompt: str) -> str:
        """Generate a response from the model.

        Attributes:
            prompt: The full prompt string for the model.
        """
        if client := self._get_client():
            return self._generate_sdk(client, prompt)
        return self._generate_urllib(prompt)

    def _generate_sdk(self, client: ollama.Client, prompt: str) -> str:
        """Internal helper for SDK-based generation."""
        try:
            response = client.generate(
                model=self.settings.active_model,
                prompt=prompt,
                stream=False,
                keep_alive=600,
                options={
                    "temperature": self.settings.temperature,
                    "num_ctx": 3072,
                    "num_predict": 384,
                },
            )
            
            # response is a dictionary-like object in the SDK
            text = response.get("response") if isinstance(response, dict) else getattr(response, "response", "")
            
            if not isinstance(text, str):
                raise GenerationError("Ollama SDK response missing expected text field")
            return text.strip()
            
        except Exception as exc:
            if isinstance(exc, GenerationError):
                raise
            raise GenerationError(f"Ollama SDK generate failed: {exc}") from exc

    def _generate_urllib(self, prompt: str) -> str:
        """Internal helper for standard library-based fallback generation."""
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
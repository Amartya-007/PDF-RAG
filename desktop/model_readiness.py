from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from backend.app.core.config import Settings


@dataclass(frozen=True)
class ModelReadiness:
    ollama_required: bool
    ollama_reachable: bool
    available_models: list[str] = field(default_factory=list)
    required_models: list[str] = field(default_factory=list)
    missing_models: list[str] = field(default_factory=list)
    setup_commands: list[str] = field(default_factory=list)
    message: str = ""

    @property
    def ready(self) -> bool:
        if not self.ollama_required:
            return True
        return self.ollama_reachable and not self.missing_models


class ModelReadinessChecker:
    def __init__(self, timeout_seconds: float = 3.0) -> None:
        self.timeout_seconds = timeout_seconds

    def check(self, settings: Settings) -> ModelReadiness:
        required = self._required_models(settings)
        if not settings.use_ollama:
            return ModelReadiness(
                ollama_required=False,
                ollama_reachable=False,
                required_models=required,
                message="Ollama is disabled. The app will use local development fallbacks.",
            )

        try:
            available = self.list_models(settings.ollama_base_url)
        except OSError:
            return ModelReadiness(
                ollama_required=True,
                ollama_reachable=False,
                required_models=required,
                missing_models=required,
                setup_commands=["ollama serve", *[f"ollama pull {model}" for model in required]],
                message=(
                    "Ollama is not reachable. Start Ollama locally, then pull the required models once."
                ),
            )

        missing = [model for model in required if not self._has_model(available, model)]
        return ModelReadiness(
            ollama_required=True,
            ollama_reachable=True,
            available_models=available,
            required_models=required,
            missing_models=missing,
            setup_commands=[f"ollama pull {model}" for model in missing],
            message=(
                "All required Ollama models are available."
                if not missing
                else "Ollama is running, but one or more required models are missing."
            ),
        )

    def list_models(self, base_url: str) -> list[str]:
        url = f"{base_url.rstrip('/')}/api/tags"
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OSError(str(exc)) from exc

        models = data.get("models", [])
        names: list[str] = []
        for item in models:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.append(item["name"])
        return names

    def _required_models(self, settings: Settings) -> list[str]:
        models = [
            settings.active_model,
            settings.embedding_model,
        ]
        return list(dict.fromkeys(model for model in models if model))

    def _has_model(self, available: list[str], required: str) -> bool:
        if ":" in required:
            return required in available
        return any(model == required or model.split(":", 1)[0] == required for model in available)

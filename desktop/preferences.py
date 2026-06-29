from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from desktop.runtime import default_app_data_dir


@dataclass(frozen=True)
class DesktopPreferences:
    use_ollama: bool = True
    ollama_base_url: str = "http://localhost:11434"
    active_model: str = "qwen3.5:4b"
    embedding_model: str = "qwen3-embedding:4b"


def preferences_path() -> Path:
    return default_app_data_dir().parent / "settings.json"


def load_preferences() -> DesktopPreferences:
    path = preferences_path()
    if not path.exists():
        return DesktopPreferences()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DesktopPreferences()
    return DesktopPreferences(
        use_ollama=bool(data.get("use_ollama", True)),
        ollama_base_url=str(data.get("ollama_base_url", "http://localhost:11434")),
        active_model=str(data.get("active_model", "qwen3.5:4b")),
        embedding_model=str(data.get("embedding_model", "qwen3-embedding:4b")),
    )


def save_preferences(preferences: DesktopPreferences) -> None:
    path = preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(preferences.__dict__, indent=2), encoding="utf-8")


def apply_preferences(preferences: DesktopPreferences) -> None:
    os.environ["RAG_USE_OLLAMA"] = "1" if preferences.use_ollama else "0"
    os.environ["OLLAMA_BASE_URL"] = preferences.ollama_base_url
    os.environ["RAG_ACTIVE_MODEL"] = preferences.active_model
    os.environ["RAG_EMBEDDING_MODEL"] = preferences.embedding_model

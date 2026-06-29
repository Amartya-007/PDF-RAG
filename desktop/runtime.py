from __future__ import annotations

import os
from pathlib import Path


APP_NAME = "Local PDF RAG"


def configure_desktop_environment() -> Path:
    raw_data_dir = os.environ.get("RAG_DATA_DIR")
    if raw_data_dir:
        data_dir = Path(raw_data_dir).expanduser()
    else:
        data_dir = default_app_data_dir()
        os.environ["RAG_DATA_DIR"] = str(data_dir)
    os.environ.setdefault("RAG_ACTIVE_MODEL", "qwen3.5:4b")
    return data_dir


def default_app_data_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_NAME / "data"
    return Path.home() / ".local-pdf-rag" / "data"

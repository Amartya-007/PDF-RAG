from __future__ import annotations

import importlib
from pathlib import Path
import tomllib

from backend.app.core.config import get_settings
from backend.app.services.rag_service_v2 import RagServiceV2


ROOT = Path(__file__).resolve().parents[2]


def _active_text_files() -> list[Path]:
    ignored_dirs = {
        ".git",
        ".kiro",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "backend/.test-tmp",
        "backend/__pycache__",
        "backend/data",
    }
    ignored_files = {
        "backend/tests/test_legacy_removal.py",
        "CHANGELOG.md",
        "Model-to-use-instructions.md",
    }
    suffixes = {".env", ".md", ".py", ".txt", ".toml"}
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT).as_posix()
        if path.is_dir() or rel in ignored_files:
            continue
        if any(rel == d or rel.startswith(f"{d}/") for d in ignored_dirs):
            continue
        if path.suffix.lower() in suffixes or path.name == ".env.example":
            files.append(path)
    return files


def test_desktop_package_and_vector_modules_are_removed() -> None:
    removed_paths = [
        ROOT / "desktop",
        ROOT / "backend/app/indexing/embeddings.py",
        ROOT / "backend/app/indexing/vector_store.py",
        ROOT / "backend/app/ingestion/pipeline.py",
        ROOT / "backend/tests/test_desktop_controller.py",
        ROOT / "backend/tests/test_desktop_preferences.py",
        ROOT / "backend/tests/test_desktop_runtime.py",
        ROOT / "backend/tests/test_model_readiness.py",
    ]

    assert [path.relative_to(ROOT).as_posix() for path in removed_paths if path.exists()] == []


def test_packaging_no_longer_exposes_desktop_or_pyside() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"].get("optional-dependencies", {})
    scripts = pyproject["project"].get("scripts", {})
    package_find = pyproject["tool"]["setuptools"]["packages"]["find"]
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "desktop" not in optional
    assert scripts == {}
    assert "desktop*" not in package_find["include"]
    assert "PySide6" not in requirements
    assert "shiboken6" not in requirements
    assert "pyinstaller" not in requirements.lower()


def test_active_sources_do_not_reference_removed_desktop_or_vector_apis() -> None:
    forbidden = [
        "backend.app.indexing.embeddings",
        "backend.app.indexing.vector_store",
        "EmbeddingService",
        "LocalVectorStore",
        "cosine_similarity",
        "PySide6",
        "desktop.",
        "RAG_EMBEDDING_MODEL",
        "RAG_DENSE_TOP_K",
        "RAG_FUSION_TOP_K",
        "RAG_RERANK_TOP_K",
        "RAG_EMBEDDING_BATCH_SIZE",
        "RAG_ALLOW_HASH_EMBEDDINGS",
    ]
    offenders: list[str] = []

    for path in _active_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}: {token}")

    assert offenders == []


def test_legacy_rag_service_import_uses_vectorless_coordinator() -> None:
    module = importlib.import_module("backend.app.rag_service")

    assert module.RagService is RagServiceV2
    assert not hasattr(get_settings(), "embedding_model")
    assert importlib.util.find_spec("desktop") is None

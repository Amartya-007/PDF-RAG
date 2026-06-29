from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings, get_settings
from backend.app.models import Answer, Chunk, Document
from backend.app.rag_service import RagService
from desktop.model_readiness import ModelReadiness, ModelReadinessChecker
from desktop.preferences import DesktopPreferences, apply_preferences, save_preferences


class DesktopController:
    def __init__(
        self,
        settings: Settings | None = None,
        service: RagService | None = None,
        readiness_checker: ModelReadinessChecker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.service = service or RagService(self.settings)
        self.readiness_checker = readiness_checker or ModelReadinessChecker()

    def status(self) -> dict[str, Any]:
        documents = self.service.store.list_documents()
        chunks = self.service.store.list_chunks()
        concepts = self.service.store.list_concepts()
        readiness = self.model_readiness()
        return {
            "data_dir": str(self.settings.data_dir),
            "documents": len(documents),
            "chunks": len(chunks),
            "concepts": len(concepts),
            "ollama_ready": readiness.ready,
            "ollama_message": readiness.message,
        }

    def list_documents(self) -> list[Document]:
        return self.service.store.list_documents()

    def ingest(self, path: str | Path, build_okf: bool = True) -> Document:
        return self.service.ingest(Path(path), build_okf=build_okf)

    def ask(self, question: str, include_debug: bool = False) -> Answer:
        return self.service.ask(question, include_debug=include_debug)

    def retrieve(self, question: str, include_debug: bool = True) -> tuple[list[Chunk], dict[str, object]]:
        return self.service.retrieve(question, include_debug=include_debug)

    def import_okf_bundle(self, path: str | Path) -> list[object]:
        return self.service.import_okf_bundle(Path(path))

    def validate_okf_bundle(self, path: str | Path) -> list[dict[str, str]]:
        return self.service.validate_okf_bundle(Path(path))

    def model_readiness(self) -> ModelReadiness:
        return self.readiness_checker.check(self.settings)

    def model_readiness_dict(self) -> dict[str, Any]:
        return asdict(self.model_readiness())

    def available_ollama_models(self, base_url: str | None = None) -> list[str]:
        return self.readiness_checker.list_models(base_url or self.settings.ollama_base_url)

    def update_preferences(
        self,
        *,
        use_ollama: bool,
        ollama_base_url: str,
        active_model: str,
        embedding_model: str,
    ) -> None:
        preferences = DesktopPreferences(
            use_ollama=use_ollama,
            ollama_base_url=ollama_base_url,
            active_model=active_model,
            embedding_model=embedding_model,
        )
        save_preferences(preferences)
        apply_preferences(preferences)
        self.service.close()
        self.settings = get_settings()
        self.service = RagService(self.settings)

    def close(self) -> None:
        self.service.close()

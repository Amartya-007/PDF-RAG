from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.app.models import Document
from backend.app.rag_service import RagService


@dataclass(frozen=True)
class IngestionJob:
    path: Path
    build_okf: bool = True


class InlineWorker:
    def __init__(self, service: RagService) -> None:
        self.service = service

    def run_ingestion(self, job: IngestionJob) -> Document:
        return self.service.ingest(job.path, build_okf=job.build_okf)

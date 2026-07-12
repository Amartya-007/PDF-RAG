"""FastAPI application exposing the PDF-RAG backend to a frontend client.

Every route here maps onto an existing, tested ``RagServiceV2`` method or
``MetadataStore``/``JobRepository`` call -- this module intentionally
contains no business logic of its own beyond request/response shaping.

See ``docs/apis.md`` for the full frontend-facing API reference.
"""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

try:
    from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "FastAPI API requires optional dependencies. Run `py -m pip install -e .[backend]`."
    ) from exc

from backend.app.api.schemas.citation import CitationOut
from backend.app.api.schemas.document import DocumentOut
from backend.app.api.schemas.job import JobOut
from backend.app.api.schemas.query import QueryRequest
from backend.app.api.schemas.response import AnswerResponse
from backend.app.api.schemas.session import (
    SessionCreateRequest,
    SessionOut,
    SessionRenameRequest,
)
from backend.app.api.schemas.settings import SettingsOut, SettingsPatchRequest
from backend.app.domain.exceptions import RagError
from backend.app.knowledge.okf import validate_okf_bundle
from backend.app.rag_service import RagService

app = FastAPI(title="Local PDF RAG")


def get_service() -> RagService:
    """Dependency-injected singleton RagService instance for this process."""
    if not hasattr(app.state, "service"):
        app.state.service = RagService()
    return app.state.service


def _configure_cors() -> None:
    """Adds CORS middleware once, using the frontend origin from settings."""
    service = get_service()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(service.settings.frontend_origin).rstrip("/")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


_configure_cors()


class OkfImportRequest(BaseModel):
    path: str


# ── Health ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def status(service: RagService = Depends(get_service)) -> dict[str, Any]:
    return service.status()


# ── Settings ─────────────────────────────────────────────────────────────

@app.get("/api/settings", response_model=SettingsOut)
def get_settings_endpoint(service: RagService = Depends(get_service)) -> SettingsOut:
    return SettingsOut.model_validate(service.settings, from_attributes=True)


@app.patch("/api/settings", response_model=SettingsOut)
def patch_settings(
    request: SettingsPatchRequest, service: RagService = Depends(get_service)
) -> SettingsOut:
    """Updates a small, safe subset of runtime settings in-place.

    Only fields explicitly listed in ``SettingsPatchRequest`` can be changed;
    everything else (paths, ports, model provider URLs) requires a restart.
    """
    updates = request.model_dump(exclude_unset=True, exclude_none=True)
    for key, value in updates.items():
        setattr(service.settings, key, value)
    return SettingsOut.model_validate(service.settings, from_attributes=True)


# ── Sessions ─────────────────────────────────────────────────────────────

@app.get("/api/sessions", response_model=list[SessionOut])
def list_sessions(service: RagService = Depends(get_service)) -> list[Any]:
    return service.list_sessions()


@app.post("/api/sessions", response_model=SessionOut, status_code=201)
def create_session(
    request: SessionCreateRequest, service: RagService = Depends(get_service)
) -> Any:
    return service.create_session(request.title)


@app.patch("/api/sessions/{session_id}")
def rename_session(
    session_id: str,
    request: SessionRenameRequest,
    service: RagService = Depends(get_service),
) -> dict[str, str]:
    service.rename_session(session_id, request.title)
    return {"session_id": session_id, "title": request.title}


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, service: RagService = Depends(get_service)) -> None:
    service.delete_session(session_id)


# ── Documents ────────────────────────────────────────────────────────────

@app.get("/api/documents", response_model=list[DocumentOut])
def list_documents(
    session_id: str | None = None, service: RagService = Depends(get_service)
) -> list[Any]:
    return service.list_documents(session_id)


@app.post("/api/documents", response_model=DocumentOut)
def upload_document(
    file: UploadFile = File(...),
    session_id: str | None = None,
    service: RagService = Depends(get_service),
) -> Any:
    """Ingests an uploaded file and returns the resulting document record.

    NOTE: this call is synchronous from the client's point of view -- the
    HTTP response only arrives once ingestion has fully finished (FastAPI
    runs the handler in a worker thread so it doesn't block the event loop
    or other requests, but *this* request does wait). Use
    ``GET /api/jobs/{job_id}`` afterwards for the job's recorded stage
    history, or poll ``GET /api/documents`` for status if you'd rather not
    hold the upload request open in the UI.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing")

    suffix = Path(file.filename).suffix or ".pdf"
    if suffix.lower() not in service.settings.allowed_file_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {service.settings.allowed_file_extensions}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        shutil.copyfileobj(file.file, handle)
        temp_path = Path(handle.name)

    try:
        return service.ingest(temp_path, session_id=session_id)
    except RagError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str, service: RagService = Depends(get_service)) -> None:
    service.delete_document(document_id)


# ── Jobs ─────────────────────────────────────────────────────────────────

@app.get("/api/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, service: RagService = Depends(get_service)) -> Any:
    job = service._job_repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/documents/{document_id}/jobs", response_model=list[JobOut])
def list_document_jobs(document_id: str, service: RagService = Depends(get_service)) -> list[Any]:
    return service._job_repo.list_jobs_for_document(document_id)


# ── Query ────────────────────────────────────────────────────────────────

def _to_citation_out(citation: Any) -> CitationOut:
    return CitationOut(
        document_id=citation.document_id,
        document_name=citation.filename,
        page_start=citation.page_start,
        page_end=citation.page_end,
        heading_path=citation.heading_path,
        excerpt=citation.excerpt,
        node_id=citation.chunk_id,
    )


@app.post("/api/query", response_model=AnswerResponse)
def query(request: QueryRequest, service: RagService = Depends(get_service)) -> AnswerResponse:
    answer = service.ask(
        request.question,
        session_id=request.session_id,
        include_debug=request.include_debug,
    )
    return AnswerResponse(
        answer=answer.answer,
        answerable=answer.answerable,
        strategy=answer.strategy,
        session_id=request.session_id,
        query_id=str(uuid.uuid4()),
        citations=[_to_citation_out(c) for c in answer.citations],
        debug=answer.debug if request.include_debug else None,
    )


@app.post("/api/debug/retrieve")
def debug_retrieve(request: QueryRequest, service: RagService = Depends(get_service)) -> dict[str, Any]:
    chunks, debug = service.retrieve(request.question, include_debug=True, session_id=request.session_id)
    return {"chunks": chunks, "debug": debug}


# ── OKF (knowledge bundle) ────────────────────────────────────────────────

@app.post("/api/okf/validate")
def validate_okf(request: OkfImportRequest) -> dict[str, Any]:
    issues = validate_okf_bundle(Path(request.path))
    return {"issues": issues}


@app.post("/api/okf/import")
def import_okf(request: OkfImportRequest, service: RagService = Depends(get_service)) -> dict[str, Any]:
    try:
        concepts = service.import_okf_bundle(Path(request.path))
        return {
            "imported_concepts": len(concepts),
            "concept_ids": [concept.concept_id for concept in concepts],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
    from pydantic import BaseModel
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "FastAPI API requires optional dependencies. Run `py -m pip install -e .[backend]`."
    ) from exc

from backend.app.rag_service import RagService

app = FastAPI(title="Local PDF RAG")

# 1. Dependency Injection setup
# This ensures we don't rely on global state and makes the API easily testable.
def get_service() -> RagService:
    # In a production app, this might pull from app.state or a DI container
    if not hasattr(app.state, "service"):
        app.state.service = RagService()
    return app.state.service


class QueryRequest(BaseModel):
    question: str
    include_debug: bool = False


class OkfImportRequest(BaseModel):
    path: str


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/documents")
def list_documents(service: RagService = Depends(get_service)) -> list[Any]:
    # FastAPI automatically serializes Pydantic models and Dataclasses recursively.
    # We don't need to manually map __dict__.
    return service.store.list_documents()


@app.post("/api/documents")
def upload_document(
    file: UploadFile = File(...), 
    service: RagService = Depends(get_service)
) -> Any:
    """
    CRITICAL FIX: This is now a synchronous `def`. 
    FastAPI will automatically run this in a background threadpool, 
    preventing the heavy PDF ingestion from freezing the web server!
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing")
        
    suffix = Path(file.filename).suffix or ".pdf"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        # Stream the file directly to disk in chunks to prevent RAM exhaustion
        shutil.copyfileobj(file.file, handle)
        temp_path = Path(handle.name)
        
    try:
        document = service.ingest(temp_path)
        return document
    except Exception as e:
        # Catch ingestion errors so the API doesn't crash ungracefully
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {e}")
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/api/query")
def query(
    request: QueryRequest, 
    service: RagService = Depends(get_service)
) -> dict[str, Any]:
    answer = service.ask(request.question, include_debug=request.include_debug)
    
    # Return the objects directly. FastAPI's JSON encoder handles the serialization.
    return {
        "answer": answer.answer,
        "answerable": answer.answerable,
        "citations": answer.citations,
        "debug": answer.debug,
    }


@app.post("/api/okf/validate")
def validate_okf(
    request: OkfImportRequest, 
    service: RagService = Depends(get_service)
) -> dict[str, Any]:
    issues = service.validate_okf_bundle(Path(request.path))
    return {"issues": issues}


@app.post("/api/okf/import")
def import_okf(
    request: OkfImportRequest, 
    service: RagService = Depends(get_service)
) -> dict[str, Any]:
    try:
        concepts = service.import_okf_bundle(Path(request.path))
        return {
            "imported_concepts": len(concepts),
            "concept_ids": [concept.concept_id for concept in concepts],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/debug/retrieve")
def debug_retrieve(
    request: QueryRequest, 
    service: RagService = Depends(get_service)
) -> dict[str, Any]:
    chunks, debug = service.retrieve(request.question, include_debug=True)
    return {
        "chunks": chunks,
        "debug": debug,
    }
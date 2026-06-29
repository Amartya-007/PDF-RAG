from __future__ import annotations

import tempfile
from pathlib import Path

try:
    from fastapi import FastAPI, File, UploadFile
    from pydantic import BaseModel
except Exception as exc:  # pragma: no cover - import-time setup guidance
    raise RuntimeError(
        "FastAPI API requires optional dependencies. Run `py -m pip install -e .[backend]`."
    ) from exc

from backend.app.rag_service import RagService


app = FastAPI(title="Local PDF RAG")
service = RagService()


class QueryRequest(BaseModel):
    question: str
    include_debug: bool = False


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/documents")
def list_documents() -> list[dict[str, str]]:
    return [document.__dict__ for document in service.store.list_documents()]


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str]:
    suffix = Path(file.filename or "document.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(await file.read())
        temp_path = Path(handle.name)
    try:
        document = service.ingest(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return document.__dict__


@app.post("/api/query")
def query(request: QueryRequest) -> dict[str, object]:
    answer = service.ask(request.question, include_debug=request.include_debug)
    return {
        "answer": answer.answer,
        "answerable": answer.answerable,
        "citations": [citation.__dict__ for citation in answer.citations],
        "debug": answer.debug,
    }


@app.post("/api/debug/retrieve")
def debug_retrieve(request: QueryRequest) -> dict[str, object]:
    chunks, debug = service.retrieve(request.question, include_debug=True)
    return {
        "chunks": [chunk.__dict__ for chunk in chunks],
        "debug": debug,
    }

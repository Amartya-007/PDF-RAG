# Running PDF-RAG

Vectorless, backend-first PDF RAG application: FastAPI backend, SQLite
metadata + FTS5/BM25 lexical retrieval, hierarchical document nodes,
extractive or Ollama-generated answers.

## Install

```powershell
cd D:\PDF-RAG
uv venv .venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt

# Required: the spaCy English model isn't installable via pip alone
python -m spacy download en_core_web_sm
```

For optional OCR support (scanned PDFs, via Docling):

```powershell
uv pip install -e ".[pdf]"
```

## Optional Ollama Setup

Ollama is used for answer *synthesis* when enabled. Retrieval remains
lexical (FTS5/BM25) either way — Ollama never affects what's retrieved,
only how the final answer is worded.

```powershell
ollama pull qwen3.5:4b
$env:RAG_USE_OLLAMA = "1"
$env:RAG_ACTIVE_MODEL = "qwen3.5:4b"
```

## Run Tests

```powershell
.venv\Scripts\python.exe -m pytest backend\tests
```

Expected: `89 passed`. See `docs/baseline_2026-07-11.md` for what was
failing before, and what fixed it.

## Developer Server

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.api.main:app --reload
```

- Server: `http://127.0.0.1:8000`
- Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Full endpoint reference for frontend development: `docs/apis.md`

CORS is configured from `RAG_FRONTEND_ORIGIN` (default
`http://localhost:3000`) — set this to your frontend dev server's origin
before starting the backend, since it's read once at startup.

## Quick smoke test

Once the server is running:

```powershell
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/sessions
curl -X POST http://127.0.0.1:8000/api/documents -F "file=@C:\path\to\your.pdf"
curl -X POST http://127.0.0.1:8000/api/query -H "Content-Type: application/json" -d "{\"question\": \"What is this document about?\"}"
```

## Data

By default, local data is stored under `backend/data`:

```text
documents/          uploaded source copies
indexes/            lexical (FTS5/BM25/heading/phrase) and OKF-compatible index files
knowledge/          generated/imported OKF Markdown
metadata.sqlite3    SQLite metadata, sessions, documents, jobs, and nodes
trees/              optional tree artifacts
```

Set `RAG_SQLITE_PATH=:memory:` for a throwaway, in-memory database (used
by the test suite; not recommended for real usage since it doesn't
persist across restarts).

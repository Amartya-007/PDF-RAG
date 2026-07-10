# Running Local PDF RAG

The current project is a vectorless backend-first PDF RAG application.

## Install

```powershell
cd D:\PDF-RAG
uv venv .venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
uv pip install -e ".[backend,dev]"
```

For optional OCR support:

```powershell
uv pip install -e ".[pdf]"
```

## Optional Ollama Setup

Ollama is used for answer synthesis when enabled. Retrieval remains lexical.

```powershell
ollama pull qwen3.5:4b
$env:RAG_USE_OLLAMA = "1"
$env:RAG_ACTIVE_MODEL = "qwen3.5:4b"
```

## Run Tests

```powershell
.venv\Scripts\python.exe -m pytest backend\tests
```

## Developer Server

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.api.main:app --reload
```

## Data

By default, local data is stored under `backend/data`:

```text
documents/          uploaded source copies
indexes/            lexical and OKF-compatible index files
knowledge/          generated OKF Markdown
metadata.sqlite3    SQLite metadata and nodes
trees/              optional tree artifacts
```

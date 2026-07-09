# Local PDF RAG

Offline document question-answering with SQLite metadata, FTS5 lexical
retrieval, structural document nodes, and source citations.

## Requirements

| Requirement | Version |
| --- | --- |
| Python | 3.12 or 3.13 |
| uv | Any recent version |
| Ollama | Optional, for generated answers |

## Setup

```powershell
cd D:\PDF-RAG
uv venv .venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
uv pip install -e ".[backend,dev]"
```

Install OCR support for scanned PDFs when needed:

```powershell
uv pip install -e ".[pdf]"
```

## Ollama

Ollama is optional. When enabled, it is used for answer synthesis only.

```powershell
ollama pull qwen3.5:4b
$env:RAG_USE_OLLAMA = "1"
$env:RAG_ACTIVE_MODEL = "qwen3.5:4b"
```

## Environment

```powershell
$env:RAG_DATA_DIR = "backend/data"
$env:RAG_SQLITE_PATH = "backend/data/metadata.sqlite3"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:RAG_ACTIVE_MODEL = "qwen3.5:4b"
$env:RAG_SPARSE_TOP_K = "40"
$env:RAG_FINAL_CONTEXT_CHUNKS = "8"
$env:RAG_TEMPERATURE = "0.1"
$env:RAG_USE_OLLAMA = "0"
```

## Tests

```powershell
.venv\Scripts\python.exe -m pytest backend\tests
```

## Project Structure

```text
D:\PDF-RAG\
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI schemas and app wiring
│   │   ├── core/            # Settings, hashing, text helpers
│   │   ├── database/        # SQLite store and repositories
│   │   ├── domain/          # Domain models and exceptions
│   │   ├── generation/      # Extractive and Ollama answerers
│   │   ├── indexing/        # FTS5, BM25, heading, phrase indexes
│   │   ├── ingestion/       # Parsing, layout, structure building
│   │   ├── retrieval/       # Lexical retrieval and tree navigation
│   │   ├── services/        # Thin orchestration services
│   │   └── verification/    # Citation validation
│   └── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Notes

The application is now vectorless. Retrieval uses document nodes, SQLite FTS5,
heading and phrase indexes, and BM25 support for OKF compatibility. The removed
desktop and dense-retrieval configuration is recorded in `CHANGELOG.md`.

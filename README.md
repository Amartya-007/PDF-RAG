# Local PDF RAG

Fully offline, high-accuracy document Q&A with source citations.
Ask questions across 10–20 PDF files. Every answer cites the exact document,
page, and text excerpt. No cloud API, no paid service, no internet required.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.12 or 3.13 |
| uv (package manager) | any recent |
| Ollama | any recent |
| RAM | 8 GB minimum, 16 GB recommended |
| Storage | 15 GB free (models + index) |

---

## One-time setup

### 1 — Install uv

```powershell
# Windows PowerShell (run once)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal after installation.

### 2 — Create the virtual environment

```powershell
cd D:\PDF-RAG
uv venv .venv --python 3.12
```

### 3 — Activate the virtual environment

```powershell
# PowerShell
.venv\Scripts\Activate.ps1

# CMD
.venv\Scripts\activate.bat
```

You will see `(.venv)` at the start of your prompt when it is active.

### 4 — Install all dependencies

```powershell
# Option A — install from requirements.txt (pinned, fastest)
uv pip install -r requirements.txt
uv pip install -e .

# Option B — install from pyproject.toml (resolves latest compatible versions)
uv pip install -e ".[desktop,dev]"
```

### 5 — Install and start Ollama

Download Ollama from https://ollama.com and install it once.

Pull the required models (internet required, one time only):

```powershell
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:4b
```

Optional — pull the larger, higher-quality generation model:

```powershell
ollama pull qwen3.5:9b
```

Verify models are available:

```powershell
ollama list
```

---

## Running the desktop app

Make sure the virtual environment is activated (`.venv\Scripts\Activate.ps1`),
then run:

```powershell
cd D:\PDF-RAG
$env:RAG_USE_OLLAMA = "1"
python -m desktop.app
```

Or as a single command:

```powershell
cd D:\PDF-RAG; $env:RAG_USE_OLLAMA = "1"; .venv\Scripts\python.exe -m desktop.app
```

---

## Using the app

1. Click **+ New Chat** in the sidebar.
2. Select one or more PDF files in the file picker that opens automatically.
3. Wait for the status bar to show `Embedding N/N chunks… (100%)`.
4. Type a question in the input box at the bottom and press **Enter** or click **Ask**.
5. The answer appears in the chat. Citations are shown in the right panel.

**Tips**

- Right-click a chat in the sidebar to rename or delete it.
- Right-click a document under a chat to remove it.
- Click **Import PDFs / Text** in the sidebar to add more files to an existing chat.
- Click **Repair Stuck Imports** if a document shows `Failed` after a crash.

---

## Environment variables

All settings have sensible defaults. Override them by setting environment
variables before launching the app.

```powershell
# Model selection
$env:RAG_ACTIVE_MODEL       = "qwen3.5:4b"    # generation model
$env:RAG_EMBEDDING_MODEL    = "qwen3-embedding:4b"
$env:OLLAMA_BASE_URL        = "http://localhost:11434"

# Enable / disable Ollama (set to 0 for offline hash-embedding fallback)
$env:RAG_USE_OLLAMA         = "1"

# Data directory (default: AppData\Local\Local PDF RAG\data)
$env:RAG_DATA_DIR           = "C:\MyData\rag-data"

# Retrieval tuning
$env:RAG_DENSE_TOP_K        = "40"
$env:RAG_SPARSE_TOP_K       = "40"
$env:RAG_FUSION_TOP_K       = "50"
$env:RAG_RERANK_TOP_K       = "30"
$env:RAG_FINAL_CONTEXT_CHUNKS = "8"

# Generation
$env:RAG_TEMPERATURE        = "0.1"
$env:RAG_EMBEDDING_BATCH_SIZE = "64"
```

---

## Running tests

```powershell
cd D:\PDF-RAG
.venv\Scripts\Activate.ps1
pytest backend/tests -v
```

---

## Project structure

```
D:\PDF-RAG\
├── backend/
│   ├── app/
│   │   ├── domain/           # Exceptions + enums (no dependencies)
│   │   ├── ports/            # Protocol interfaces for all replaceable components
│   │   ├── core/             # Config, hashing, text utilities
│   │   ├── models.py         # Domain dataclasses (Document, Chunk, Answer, …)
│   │   ├── rag_service.py    # Main orchestrator
│   │   ├── database/         # SQLite metadata store
│   │   ├── ingestion/        # PDF parsing, cleaning, chunking, pipeline
│   │   ├── indexing/         # Vector store (numpy), BM25 (bm25s), embeddings
│   │   ├── retrieval/        # Query classifier, hybrid search, RRF, reranker
│   │   ├── generation/       # Ollama client, extractive answerer, prompts
│   │   ├── verification/     # Citation entailment checks
│   │   └── knowledge/        # OKF knowledge graph (optional)
│   └── tests/
├── desktop/
│   ├── app.py                # Entry point
│   ├── main_window.py        # PySide6 UI
│   ├── controller.py         # UI ↔ service bridge
│   ├── workers.py            # Qt thread pool workers
│   └── theme.py              # Colours and stylesheet
├── docs/
├── .venv/                    # Virtual environment (not committed)
├── pyproject.toml
├── requirements.txt          # Pinned dependencies
└── README.md
```

---

## Troubleshooting

### App crashes with `ModuleNotFoundError`
The virtual environment is not active. Run `.venv\Scripts\Activate.ps1` first.

### `ollama: command not found`
Ollama is not installed or not on PATH. Download from https://ollama.com.

### Answers are slow (30–120 s)
- Use a smaller model: `$env:RAG_ACTIVE_MODEL = "qwen3.5:4b"`
- Make sure the model is loaded: `ollama list`
- First answer after a cold start is always slower (model load time).

### PDF shows `Failed` after import
Click **Repair Stuck Imports** in the sidebar.
If that does not help, the PDF may be scanned (image-only).
Install OCR support:

```powershell
uv pip install -e ".[pdf]"
```

### Embedding takes too long on first import
The `qwen3-embedding:4b` model is 2.5 GB. First use loads it from disk.
All subsequent re-imports of the same file skip embedding entirely
(SQLite cache).

# Local PDF RAG

**Offline document Q&A with source citations — runs entirely on your machine.**

Local PDF RAG is a fully private, GPU-accelerated RAG system for querying PDF collections. It ingests documents, builds a local hybrid search index, and answers questions by citing the exact page and chunk the answer came from. No cloud API, no internet required during normal use.

---

## Screenshots

The desktop app features a three-panel layout: document library on the left, chat in the centre, and source citations on the right.

> After launching, import a PDF → type a question → see the answer with cited source excerpts appear in real time.

---

## Hardware Target

| Component | Spec |
|-----------|------|
| RAM | 24 GB DDR5 |
| GPU | NVIDIA RTX 3050 · 4 GB VRAM |
| Storage | 100 GB free (Gen 5 SSD) |
| OS | Windows 11 (primary) · Linux supported |

Everything runs locally via [Ollama](https://ollama.com). No paid API keys needed.

---

## Model Stack

| Role | Model | Why |
|------|-------|-----|
| Answer generation | `qwen3.5:9b` | Best quality within 4 GB VRAM |
| Fast / fallback generation | `qwen3.5:4b` | Half the VRAM, usable for dev |
| Dense embeddings | `qwen3-embedding:4b` | Matches generation model family |
| Reranking | `Qwen/Qwen3-Reranker-0.6B` | Lightweight cross-encoder via sentence-transformers |
| Sparse retrieval | BM25 (built-in) | No extra service needed |
| PDF parsing | PyMuPDF → Docling fallback | Fast path first, OCR fallback for scanned PDFs |

---

## Quick Start

### 1 · Install Ollama and pull models

```powershell
# https://ollama.com — install the Windows app, then:
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:4b
```

### 2 · Install the desktop app

```powershell
git clone https://github.com/Amartya-007/PDF-RAG.git
cd PDF-RAG
py -m pip install -e .[desktop]
```

### 3 · Run

```powershell
# Enable Ollama and launch the desktop app
$env:RAG_USE_OLLAMA = "1"
py -m desktop.app
```

Or copy `.env.example` and set `RAG_USE_OLLAMA=1` in it before launching.

---

## CLI Usage

The CLI is useful for scripting, batch ingestion, and debugging retrieval without the GUI.

```powershell
# Initialise the local data store
py -m backend.app.cli init

# Ingest one or more documents
py -m backend.app.cli ingest .\report.pdf
py -m backend.app.cli ingest .\notes.txt

# Ask a question
py -m backend.app.cli ask "What are the annual leave rules?"

# Check system status (documents, chunks, Ollama readiness)
py -m backend.app.cli status

# Debug retrieval — shows ranked chunks before the answer is generated
py -m backend.app.cli retrieve "sick leave policy" --debug
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_USE_OLLAMA` | `0` | Set to `1` to enable Ollama for generation and embeddings |
| `RAG_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `RAG_ACTIVE_MODEL` | `qwen3.5:9b` | Generation model name |
| `RAG_EMBEDDING_MODEL` | `qwen3-embedding:4b` | Embedding model name |
| `RAG_EMBEDDING_BATCH_SIZE` | `32` | Chunks per Ollama embedding request (tune for VRAM) |
| `RAG_DATA_DIR` | `backend/data` | Where documents, indexes, and SQLite live |
| `RAG_FORCE_OCR` | `0` | Set to `1` to always use Docling/OCR (for scanned PDFs) |
| `RAG_ALLOW_HASH_EMBEDDINGS` | `1` | Fallback to deterministic hash embeddings when Ollama is off |

---

## Installing Optional Extras

```powershell
# PDF parsing extras (Docling for OCR / scanned PDFs)
py -m pip install -e .[pdf]

# FastAPI + Uvicorn HTTP server
py -m pip install -e .[backend]
py -m uvicorn backend.app.api.main:app --reload

# Reranker (Qwen3-Reranker via sentence-transformers)
py -m pip install -e .[reranker]
```

---

## Build a Windows Executable

```powershell
py -m pip install -e .[desktop]
pyinstaller desktop/packaging/local_pdf_rag_desktop.spec
# Output: dist/LocalPDFRAG/LocalPDFRAG.exe
```

---

## How It Works

```
PDF / TXT / MD
     │
     ▼
 PdfParser
 ├─ PyMuPDF  ← used first for born-digital PDFs (fast)
 └─ Docling  ← fallback for scanned / image-only PDFs (OCR)
     │
     ▼
 Chunker  →  text chunks with page + metadata
     │
     ├──► Dense index   (Ollama qwen3-embedding, batched)
     ├──► Sparse index  (BM25)
     └──► OKF generator (concept mindmap, incremental)
               │
               ▼
         Question
               │
    ┌──────────┴──────────┐
    │                     │
Dense retrieval    Sparse retrieval
    │                     │
    └──────────┬──────────┘
               │
        OKF concept expansion
               │
        Reciprocal Rank Fusion
               │
           Reranker
               │
        Context assembly
               │
      Ollama qwen3.5:9b
               │
          Answer + Citations
         (filename · page · chunk)
```

### OKF Knowledge Layer

The OKF (Open Knowledge Format) layer generates a portable Markdown mindmap from each ingested document. Each concept is a `.md` file with YAML frontmatter containing:

- `id`, `title`, `slug` — stable identifiers
- `related` — slugs of other concepts linked by sentence-level co-occurrence
- `source_chunk_ids` — the exact chunk IDs this concept was extracted from
- `tags`, `aliases`, `verification_status`

**How concepts are extracted (v2):**

1. Tokenise each chunk into sentences (not the whole chunk at once).
2. Extract 1-3 word n-grams within sentence boundaries, filtering stopwords at start/end.
3. Score candidates by distinct-sentence coverage × length bonus — multi-word phrases that repeat across many sentences score highest.
4. Deduplicate: greedily drop candidates that are substrings of higher-ranked ones.
5. Compute relations via Jaccard similarity on sentence-level co-occurrence key sets `(chunk_id, sentence_index)` — this gives genuinely distinct relation lists per concept instead of every concept linking to the same set.
6. Concept IDs are content-derived, so incremental re-indexing skips unchanged concepts.

OKF concept hits during retrieval expand back to their original source chunks before reranking, so citations always point to the real PDF page.

---

## Project Structure

```
PDF-RAG/
├── backend/
│   ├── app/
│   │   ├── api/              FastAPI REST endpoints
│   │   ├── cli.py            CLI entry point
│   │   ├── core/             Config, IDs, hashing, text utilities
│   │   ├── database/         SQLite metadata store (documents, chunks, concepts)
│   │   ├── generation/       Ollama client, answer prompt, extractive fallback
│   │   ├── indexing/         Dense vector store, BM25 sparse index, embeddings
│   │   ├── ingestion/        PDF parser, text cleaner, chunker pipeline
│   │   ├── knowledge/        OKF concept generator, validator, importer
│   │   ├── models.py         Shared dataclasses (Document, Chunk, Answer, Citation)
│   │   ├── rag_service.py    Main orchestration: ingest → index → answer
│   │   ├── retrieval/        Query analysis, fusion, reranking, context assembly
│   │   ├── verification/     Citation and numeric fact validation
│   │   └── workers/          Background job runner hooks
│   └── tests/                23 stdlib unit tests (no external services needed)
├── desktop/
│   ├── app.py                Entry point, applies QSS theme at startup
│   ├── controller.py         Desktop ↔ RagService bridge
│   ├── main_window.py        Three-panel PySide6 UI
│   ├── theme.py              Design tokens and full QSS stylesheet
│   ├── workers.py            QThreadPool worker for background tasks
│   ├── preferences.py        Persist user settings to disk
│   └── packaging/            PyInstaller spec for Windows .exe
├── docs/
│   ├── architecture.md
│   └── RUNNING.md
├── evaluation/               Evaluation dataset skeleton
├── infrastructure/           Qdrant / Docker helpers
├── scripts/
└── pyproject.toml
```

---

## Running the Tests

All 23 tests run without Ollama, Docling, or any external service — the test suite uses hash embeddings and file-based fixtures.

```powershell
py -m unittest discover -s backend/tests
```

Expected output:

```
Ran 23 tests in ~3s
OK
```

---

## FAQ

**Do I need internet?**
No. After pulling Ollama models once, everything runs offline.

**Can I use it without Ollama?**
Yes — leave `RAG_USE_OLLAMA=0`. The app falls back to deterministic hash embeddings and an extractive (non-generative) answerer. Great for testing without a GPU.

**My PDF isn't being parsed correctly.**
Install Docling for OCR support (`pip install -e .[pdf]`) and set `RAG_FORCE_OCR=1`. Docling handles scanned and image-based PDFs. Text-native PDFs should parse fine with the default PyMuPDF path.

**Ingestion is slow for large PDFs.**
Tune `RAG_EMBEDDING_BATCH_SIZE` (default 32). Larger values use more VRAM but reduce round-trips to Ollama. Also ensure `ollama ps` shows `qwen3-embedding:4b` loaded on GPU, not CPU.

**How do I add a new document?**
Drag it into the desktop app's Import dialog, or run `py -m backend.app.cli ingest path/to/file.pdf`. The index updates incrementally — only new chunks are embedded.

---

## Acknowledgements

- [Ollama](https://ollama.com) · [Qwen3](https://huggingface.co/Qwen) · [PyMuPDF](https://pymupdf.readthedocs.io) · [Docling](https://github.com/DS4SD/docling) · [PySide6](https://doc.qt.io/qtforpython-6/)

---

*Built by [Amartya Vishwakarma](https://github.com/Amartya-007) · Contributions welcome*

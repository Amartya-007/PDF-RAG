# Architecture

The system keeps two searchable layers:

1. `source_chunks`: authoritative chunks extracted from uploaded PDFs or text files.
2. `okf_concepts`: generated Markdown concept files with links back to source chunks.

Final answers must cite source chunks, not OKF files. OKF improves routing and broad concept retrieval, but the PDF-derived evidence remains the truth layer.

The main product shell is a PySide6 desktop app. It calls the Python service directly in-process, so normal use does not require a hosted web server or internet connection.

```text
PDFs
  |
Docling or PyMuPDF parsing
  |
Cleaning and hierarchical chunking
  |
SQLite metadata + local vector index + BM25
  |
Hybrid dense/sparse retrieval
  +
OKF concept retrieval and source expansion
  |
RRF fusion
  |
Reranking
  |
Evidence block
  |
Ollama generation or extractive fallback
  |
Citation validation
```

## Desktop Shell

The desktop app adds a thin UI/controller layer over `RagService`:

- document import and library view
- chat question entry and answer display
- citation/source excerpt panel
- settings dialog with data directory, Ollama URL, active model, embedding model, and readiness checks
- PySide6 worker threads for ingestion and answering so the UI remains responsive

The first Windows build uses PyInstaller and does not bundle Ollama model weights. Users install Ollama and pull models once, then run offline.

## Production Upgrade Path

- Replace `LocalVectorStore` with Qdrant dense and sparse collections.
- Replace heuristic `Reranker` with `Qwen/Qwen3-Reranker-0.6B`.
- Use Docling OCR/table modes for PDF parsing.
- Add persistent background jobs for long PDF ingestion.
- Add evaluation datasets under `evaluation/datasets`.
- Replace the initial desktop source excerpt panel with page-level PDF preview/highlighting.
- Add richer desktop retrieval inspector and evaluation dashboards after the core app is stable.

## OKF Bundle Layout

Generated OKF files use this shape:

```text
knowledge/
  index.md
  concepts/
    index.md
    revenue.md
    payments.md
```

Concept files contain YAML frontmatter with `type: concept`, identity fields, relationship fields, and source chunk bindings. The Markdown body includes human-readable links between related concepts plus excerpts and original source references. Imported OKF bundles are validated before they are copied into this layout.

## Three-Path Retrieval

Questions now retrieve through:

1. OKF concept dense/sparse search, then expansion to bound source chunks.
2. Raw PDF/source dense retrieval.
3. Raw PDF/source BM25 retrieval.

All source chunk candidates are fused with Reciprocal Rank Fusion, reranked, and then passed to answer generation. The OKF layer helps routing, but final answers still cite original source chunks.

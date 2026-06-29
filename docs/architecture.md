# Architecture

The system keeps two searchable layers:

1. `source_chunks`: authoritative chunks extracted from uploaded PDFs or text files.
2. `okf_concepts`: generated Markdown concept files with links back to source chunks.

Final answers must cite source chunks, not OKF files. OKF improves routing and broad concept retrieval, but the PDF-derived evidence remains the truth layer.

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

## Production Upgrade Path

- Replace `LocalVectorStore` with Qdrant dense and sparse collections.
- Replace heuristic `Reranker` with `Qwen/Qwen3-Reranker-0.6B`.
- Use Docling OCR/table modes for PDF parsing.
- Add persistent background jobs for long PDF ingestion.
- Add evaluation datasets under `evaluation/datasets`.
- Build the Next.js frontend against the API endpoints in `backend/app/api/main.py`.

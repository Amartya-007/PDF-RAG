# Local PDF RAG

Source-grounded local RAG for PDF collections, with an optional OKF knowledge layer.

This repository starts with a runnable Python backend core that works without paid APIs. It can ingest documents, chunk extracted text, build a local hybrid search index, retrieve cited evidence, and produce either an Ollama-generated answer or a conservative extractive fallback.

The heavier production components from the proposal are intentionally integration points:

- Docling for high-quality PDF parsing and OCR
- Qdrant for persistent vector/hybrid search
- Ollama for local embeddings and answer generation
- Qwen3 reranker through `sentence-transformers`
- FastAPI for the HTTP API

## Quick Start

```powershell
py -m backend.app.cli init
py -m backend.app.cli ingest .\some-document.pdf
py -m backend.app.cli ask "What does this document say about leave limits?"
```

If your first files are PDFs and you have not installed a PDF parser yet, install the optional backend dependencies:

```powershell
py -m pip install -e .[backend]
```

For local generation and embeddings, install Ollama and pull the models from `Model-to-use-instructions.md`:

```powershell
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:4b
```

Then set `RAG_USE_OLLAMA=1` in your environment or copy `.env.example` into your preferred launcher.

## Development Commands

```powershell
py -m unittest discover -s backend/tests
py -m backend.app.cli status
py -m backend.app.cli retrieve "your question" --debug
py -m backend.app.cli validate-okf .\path\to\okf-bundle
py -m backend.app.cli import-okf .\path\to\okf-bundle
```

After installing FastAPI/Uvicorn:

```powershell
py -m uvicorn backend.app.api.main:app --reload
```

## Project Shape

```text
backend/
  app/
    api/              FastAPI endpoints
    core/             config, IDs, hashing, text helpers
    database/         SQLite metadata store
    generation/       answer prompt, Ollama client, fallback answerer
    indexing/         local vector and sparse indexes
    ingestion/        parser, cleaning, chunking pipeline
    knowledge/        OKF concept generation and validation
    retrieval/        query analysis, fusion, reranking, context assembly
    verification/     citation and numeric validation
    workers/          job runner hooks
  tests/              stdlib unit tests
docs/                 architecture notes
evaluation/           evaluation dataset skeleton
frontend/             future Next.js UI placeholder
infrastructure/       Qdrant/Docker helper files
```

## Current MVP Behavior

- Text-like files ingest immediately.
- PDFs use Docling first when installed, PyMuPDF second when installed, and otherwise return a clear setup error.
- Dense embeddings use Ollama when available, with a deterministic local hash embedding fallback for development tests.
- Sparse retrieval uses an in-repo BM25 implementation.
- Retrieval combines source dense, source sparse, and OKF concept-assisted source expansion with Reciprocal Rank Fusion.
- Answers cite document name, page number, and chunk ID.
- Unsupported answers fall back to: `I could not find sufficient evidence in the uploaded documents to answer this question.`
- OKF bundles can be generated from ingested chunks, validated, imported, indexed, and used for retrieval.

The source PDF chunks remain the authority. OKF Markdown concepts are derived artifacts used to improve retrieval, not final citations.

## OKF Support

The OKF layer uses portable Markdown files with YAML frontmatter:

- Required validation: every concept file must have frontmatter with `type`; `type: concept` files must include `id` and `title`.
- Metadata supported: `aliases`, `tags`, `related`, `depends_on`, `verification_status`, `source_chunk_ids`, `source_documents`, and `source_chunks`.
- Generated bundles include `index.md`, `concepts/index.md`, and Markdown links between related concepts.
- Imported bundles are copied into `backend/data/knowledge/`, stored in SQLite metadata, and indexed in both dense and BM25 concept indexes.
- Query retrieval follows three paths: OKF concept retrieval, raw source dense retrieval, and raw source sparse retrieval. OKF concept hits expand back to original source chunk IDs before final reranking and citation.

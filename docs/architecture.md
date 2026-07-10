# Architecture

The application is now a vectorless PDF RAG backend. Source documents are
parsed into hierarchical `DocumentNode` records, persisted in SQLite, and
indexed through lexical structures.

```text
PDF / text / Markdown
  |
LayoutParser
  |
HeadingDetector
  |
StructureBuilder
  |
SQLite nodes + chunks
  |
FTS5 + heading index + phrase index + BM25
  |
RetrievalService
  |
AnswerService
  |
Extractive answer or Ollama synthesis
  |
Citations
```

## Storage

- `documents` stores source-file metadata.
- `nodes` stores the hierarchical source tree used by retrieval.
- `chunks` remain only for OKF compatibility.
- `answers` and `answer_citations` can persist chat results.

## Indexes

`IndexManager` owns the vectorless indexes:

- `FTS5Index` for full-text node search.
- `HeadingIndex` for section-title lookup.
- `PhraseIndex` for exact phrase lookup.
- `MetadataIndex` for document-to-node bookkeeping.
- `BM25Index` for legacy OKF compatibility.

## Services

- `IngestionService` parses, structures, persists, and indexes documents.
- `RetrievalService` combines lexical hits with tree navigation and ranking.
- `AnswerService` chooses extractive answers or optional Ollama synthesis.
- `backend.app.rag_service.RagService` is a compatibility import for the new
  service coordinator.

## OKF

Generated OKF files remain Markdown concepts linked back to source chunks.
They can help organize knowledge, but final answers must cite source evidence.

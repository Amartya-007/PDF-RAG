# Requirements Document

## Introduction

This document defines requirements for a High-Accuracy Local PDF RAG System — a fully offline
Retrieval-Augmented Generation application that lets users upload 10–20 PDF documents and ask
natural-language questions, receiving grounded, cited answers without any paid API or external
service. The system runs entirely on local hardware (24 GB RAM, NVIDIA RTX 3050 4 GB VRAM)
using Ollama for model inference. It prioritises retrieval accuracy, citation fidelity, hallucination
reduction, and end-to-end source traceability over document, page, section, and chunk level.

---

## Glossary

- **System**: The complete local PDF RAG application, comprising the FastAPI backend, Next.js frontend, Qdrant vector database, SQLite metadata store, and Ollama model runtime.
- **RAG_Pipeline**: The end-to-end workflow from user query to grounded answer: query analysis → embedding → hybrid retrieval → reranking → context assembly → generation → verification.
- **Document_Ingestion_Pipeline**: The sequential workflow that accepts an uploaded PDF, validates it, parses it with Docling, extracts structure, cleans text, chunks content, embeds chunks, and indexes them in Qdrant and SQLite.
- **Parser**: The Docling-based component that converts a raw PDF file into structured document elements including headings, paragraphs, lists, tables, images, page numbers, and reading order.
- **Chunker**: The component that receives structured document elements from the Parser and produces hierarchical child and parent text chunks with associated metadata.
- **Child_Chunk**: A small, precisely bounded text segment of 250–500 tokens used for retrieval precision.
- **Parent_Chunk**: A larger section of 800–1,500 tokens used for generation context after reranking.
- **Embedder**: The component that converts text into dense float vectors using the `qwen3-embedding:4b` model via Ollama.
- **Sparse_Indexer**: The component that builds and queries a BM25 or Qdrant sparse-vector index over chunk text for exact-term retrieval.
- **Vector_Store**: The Qdrant server instance that persists dense vectors, sparse vectors, and chunk metadata payloads.
- **Metadata_Store**: The SQLite database that holds authoritative records for documents, pages, chunks, parent-child relationships, processing status, conversations, messages, and evaluation results.
- **Retrieval_Engine**: The component that combines dense and sparse search results using Reciprocal Rank Fusion and passes candidates to the Reranker.
- **Reranker**: The `Qwen/Qwen3-Reranker-0.6B` cross-encoder model loaded via sentence-transformers that scores query-passage pairs and reorders candidates.
- **Context_Builder**: The component that selects the top reranked child chunks, expands them to their Parent_Chunks, deduplicates overlapping text, and assembles a token-budgeted evidence block for generation.
- **Generator**: The Ollama-served `qwen3.5:9b` (production) or `qwen3.5:4b` (development) model that produces a grounded answer from the assembled evidence block.
- **Claim_Verifier**: The component that parses the Generator output into discrete factual claims and checks whether each claim is supported by its cited source chunk.
- **Answer**: The final response returned to the user, containing the text, per-claim source citations, and a verification summary.
- **Citation**: A reference attached to a claim that identifies the source Document, page number, section path, Chunk_ID, and a supporting excerpt.
- **Answerability_Classifier**: The component that determines, before generation, whether the assembled evidence is sufficient to answer the query.
- **Query_Rewriter**: The component that rewrites conversational or ambiguous queries into standalone, retrieval-optimised forms.
- **OCR**: Optical Character Recognition, applied by Docling to pages where native text is absent or corrupted.
- **RRF**: Reciprocal Rank Fusion — the score-fusion algorithm that merges ranked lists from dense and sparse retrieval channels.
- **Ingestion_Worker**: The background task that executes the Document_Ingestion_Pipeline asynchronously after upload.
- **Evaluation_Runner**: The component that executes a labelled question set against the RAG_Pipeline and computes retrieval and generation metrics.
- **Debug_Trace**: A structured record of every intermediate retrieval step for a given query, stored for inspection in the debug API and retrieval inspector UI.
- **Conflict**: A situation where two or more source chunks from different documents contain statements that cannot simultaneously be true.
- **SHA256_Hash**: A 256-bit cryptographic digest of a file's binary content used to detect exact-duplicate uploads.
- **HNSW**: Hierarchical Navigable Small World graph — the approximate nearest-neighbour index algorithm used by Qdrant for dense vector search.
- **Frontend**: The Next.js + TypeScript + Tailwind CSS single-page application that provides the upload interface, query interface, source panel, PDF viewer, and evaluation dashboard.

---

## Requirements

---

### Requirement 1: Document Upload and Validation

**User Story:** As a user, I want to upload PDF files to the system, so that I can later ask questions about their contents.

#### Acceptance Criteria

1. WHEN a user submits a file to `POST /api/documents`, THE System SHALL accept only files with a `application/pdf` MIME type and a valid PDF file signature (`%PDF-`).
2. WHEN a valid PDF is submitted, THE System SHALL compute its SHA256_Hash and reject the upload with HTTP 409 if an identical hash already exists in the Metadata_Store.
3. WHEN a valid, non-duplicate PDF is submitted, THE System SHALL enforce a maximum file size of 200 MB and a maximum page count of 1,000 pages, rejecting files that exceed either limit with HTTP 422 and a descriptive error message.
4. WHEN a PDF passes all validation checks, THE System SHALL persist the original file to local storage, create a document record in the Metadata_Store with status `pending`, and return a document ID with HTTP 202.
5. WHEN a document record is created, THE System SHALL enqueue an ingestion job and return the document ID to the caller before ingestion begins.
6. IF file storage fails after a document record is created, THEN THE System SHALL set the document status to `failed`, record the error, and return HTTP 500.

---

### Requirement 2: Asynchronous Document Ingestion Pipeline

**User Story:** As a user, I want uploaded PDFs processed in the background, so that the system does not block while parsing large documents.

#### Acceptance Criteria

1. WHEN an ingestion job is dequeued, THE Ingestion_Worker SHALL update the document status to `processing` in the Metadata_Store before beginning any parsing.
2. WHEN processing begins, THE Ingestion_Worker SHALL invoke the Parser with the stored file path and record the parser name and version in the Metadata_Store.
3. WHEN ingestion completes successfully, THE Ingestion_Worker SHALL set the document status to `ready` and record a completion timestamp.
4. IF any stage of ingestion raises an unrecoverable error, THEN THE Ingestion_Worker SHALL set the document status to `failed`, persist the error message and failed stage name, and stop without consuming further resources.
5. WHEN a `failed` ingestion is retried via `POST /api/documents/{id}/reprocess`, THE Ingestion_Worker SHALL resume from the last successfully completed stage rather than restarting from upload.
6. THE Ingestion_Worker SHALL process one document at a time to prevent simultaneous model-loading from exceeding the 24 GB RAM budget.
7. THE System SHALL expose document processing status via `GET /api/documents/{id}/status`, returning the current stage, percentage complete, and any error message.

---

### Requirement 3: Structure-Aware PDF Parsing

**User Story:** As a developer, I want the Parser to extract document structure faithfully, so that chunking preserves semantic boundaries and retrieval accuracy is high.

#### Acceptance Criteria

1. WHEN the Parser processes a PDF, THE Parser SHALL extract headings, paragraphs, lists, tables, captions, page numbers, reading order, and bounding boxes for each element.
2. WHEN a page contains no extractable native text or contains corrupted text, THE Parser SHALL activate OCR for that page and store the OCR engine name, language, per-page confidence score, and page resolution in the Metadata_Store.
3. WHEN a page contains valid native text, THE Parser SHALL use native text without OCR to preserve text fidelity.
4. WHEN the Parser encounters a table, THE Parser SHALL extract it as a structured object preserving column headers, rows, a table title if present, the page number, and adjacent paragraph context.
5. WHEN OCR produces a per-page confidence score below 0.80, THE Parser SHALL flag the page as low-confidence and store that flag in the Metadata_Store for downstream confidence adjustment.
6. THE Parser SHALL detect and tag repeated page-level content (headers, footers, page numbers, confidentiality lines) and exclude that content from retrieval text while preserving it in the display copy.
7. WHEN PyMuPDF is configured as the fallback parser and the Docling converter raises an unhandled exception, THE Parser SHALL transparently retry the page using PyMuPDF and record the fallback in the Metadata_Store.

---

### Requirement 4: Hierarchical Structure-Aware Chunking

**User Story:** As a developer, I want the Chunker to produce hierarchical child and parent chunks that respect document structure, so that retrieval is precise and generation context is coherent.

#### Acceptance Criteria

1. THE Chunker SHALL produce Child_Chunks with a target length of 250–500 tokens, measured after tokenisation.
2. THE Chunker SHALL produce Parent_Chunks with a target length of 800–1,500 tokens, measured after tokenisation.
3. THE Chunker SHALL align chunk boundaries to structural elements — headings, paragraphs, lists, table boundaries, section transitions — and SHALL NOT split a table row from its header, a numbered clause from its sub-clause, or a heading from the paragraph immediately following it.
4. WHEN chunking, THE Chunker SHALL prefix each Child_Chunk with a context header containing the document title, top-level section title, and sub-section title before embedding.
5. THE Chunker SHALL assign every Child_Chunk a unique Chunk_ID, a `parent_chunk_id` referencing its Parent_Chunk, and metadata including `document_id`, `filename`, `page_start`, `page_end`, `section_path`, `chunk_type`, `language`, `ocr_confidence`, `parser_version`, and `embedding_version`.
6. WHEN two chunks have identical text after normalisation, THE Chunker SHALL create a single canonical chunk and store a duplicate reference relationship in the Metadata_Store instead of indexing the same content twice.
7. WHERE a page is flagged as low-confidence OCR, THE Chunker SHALL propagate the `ocr_confidence` flag to all chunks derived from that page.

---

### Requirement 5: Dense Embedding and Vector Indexing

**User Story:** As a developer, I want all chunks to be embedded with a consistent model and stored in Qdrant, so that semantic retrieval is accurate and the index remains consistent.

#### Acceptance Criteria

1. THE Embedder SHALL use the `qwen3-embedding:4b` model via the Ollama API at `OLLAMA_BASE_URL` for all embedding operations.
2. THE Embedder SHALL embed chunks in batches of `RAG_EMBEDDING_BATCH_SIZE` (default 4) and SHALL NOT hold more than one batch in memory at a time.
3. WHEN the same model and version are used, THE Embedder SHALL produce identical vector representations for identical input strings, ensuring index consistency.
4. THE Vector_Store SHALL store each embedded chunk as a Qdrant point containing the dense vector, the sparse vector, and the full metadata payload defined in Requirement 4, Criterion 5.
5. WHEN the embedding model name or version changes, THE System SHALL invalidate all existing vectors for affected documents and re-embed them before the next query is served.
6. THE Vector_Store SHALL configure Qdrant HNSW indexes with payload indexes on `document_id` and `page_start` to support fast metadata-filtered retrieval.
7. WHEN a document is deleted via `DELETE /api/documents/{id}`, THE Vector_Store SHALL remove all Qdrant points associated with that document ID and THE Metadata_Store SHALL soft-delete the document record.

---

### Requirement 6: Sparse Retrieval Indexing

**User Story:** As a developer, I want a sparse retrieval channel alongside dense embeddings, so that exact-term queries for names, IDs, dates, and section numbers return accurate results.

#### Acceptance Criteria

1. THE Sparse_Indexer SHALL index all Child_Chunk texts using BM25 or Qdrant sparse vectors as the primary sparse retrieval method.
2. WHEN a chunk is ingested, THE Sparse_Indexer SHALL tokenise its text, compute sparse term weights, and store the sparse vector in the same Qdrant point as the dense vector.
3. THE Sparse_Indexer SHALL handle queries containing exact identifiers (order numbers, dates in ISO format, section numbers, acronyms, proper names) and return matching chunks ranked by sparse relevance score.
4. WHEN the sparse index is rebuilt due to model or schema changes, THE Sparse_Indexer SHALL complete re-indexing without dropping query availability for documents that have not changed.

---

### Requirement 7: Query Analysis and Rewriting

**User Story:** As a user, I want my follow-up and conversational questions to be resolved into standalone queries, so that retrieval is not confused by pronouns or missing context.

#### Acceptance Criteria

1. WHEN a query is received, THE Query_Rewriter SHALL classify it as one of: `direct_factual`, `exact_identifier`, `definition`, `comparison`, `summary`, `multi_document`, `table_lookup`, `numeric_date`, `procedural`, `follow_up`, or `unanswerable`.
2. WHEN a query is classified as `follow_up`, THE Query_Rewriter SHALL incorporate the most recent conversation context and produce a standalone, self-contained query before retrieval begins.
3. THE Query_Rewriter SHALL preserve the original query text alongside the rewritten form in the Debug_Trace.
4. WHEN a query is classified as `comparison` or `multi_document`, THE Query_Rewriter SHALL generate up to three alternative phrasings of the query to improve retrieval recall.
5. IF query rewriting fails due to an exception, THEN THE System SHALL fall back to using the original query unmodified and log the failure.

---

### Requirement 8: Hybrid Retrieval with Reciprocal Rank Fusion

**User Story:** As a user, I want the system to combine semantic and keyword search results, so that both conceptually similar passages and exact-term matches are surfaced.

#### Acceptance Criteria

1. WHEN a retrieval request is made, THE Retrieval_Engine SHALL execute a dense semantic search retrieving the top `RAG_DENSE_TOP_K` (default 40) Child_Chunks from the Vector_Store.
2. WHEN a retrieval request is made, THE Retrieval_Engine SHALL execute a sparse keyword search retrieving the top `RAG_SPARSE_TOP_K` (default 40) Child_Chunks from the Sparse_Indexer.
3. WHEN both result lists are available, THE Retrieval_Engine SHALL merge them using Reciprocal Rank Fusion with a constant `k=60` to produce a unified ranked candidate list of up to `RAG_FUSION_TOP_K` (default 50) unique chunks.
4. THE Retrieval_Engine SHALL NOT combine raw dense and sparse scores directly without normalisation.
5. WHEN `document_ids` is specified in the query request, THE Retrieval_Engine SHALL apply a Qdrant payload filter to restrict retrieval to the specified documents.
6. THE Retrieval_Engine SHALL record the dense rank, sparse rank, and RRF score for every candidate in the Debug_Trace.

---

### Requirement 9: Cross-Encoder Reranking

**User Story:** As a developer, I want a dedicated reranker to score query-passage relevance more precisely than embedding similarity, so that the strongest evidence reaches the Generator.

#### Acceptance Criteria

1. THE Reranker SHALL use the `Qwen/Qwen3-Reranker-0.6B` model loaded via sentence-transformers and SHALL attempt to run on CUDA; IF CUDA memory is insufficient, THEN THE Reranker SHALL fall back to CPU without raising an error to the caller.
2. THE Reranker SHALL score the top `RAG_RERANK_TOP_K` (default 30) candidates from the Retrieval_Engine using the query and chunk text as a pair.
3. THE Reranker SHALL NOT be loaded simultaneously with the Generator on the GPU when combined VRAM usage would exceed 4 GB; THE System SHALL unload the Reranker before loading the Generator.
4. WHEN reranking completes, THE Reranker SHALL return the candidates sorted by descending relevance score, with per-candidate scores stored in the Debug_Trace.
5. THE System SHALL select the top `RAG_FINAL_CONTEXT_CHUNKS` (default 8) candidates after reranking for context assembly.

---

### Requirement 10: Context Assembly and Parent-Chunk Expansion

**User Story:** As a developer, I want the best reranked child chunks expanded to their parent sections, so that the Generator receives coherent, well-bounded evidence without redundancy.

#### Acceptance Criteria

1. WHEN the top `RAG_FINAL_CONTEXT_CHUNKS` child chunks are selected, THE Context_Builder SHALL load each chunk's corresponding Parent_Chunk from the Metadata_Store.
2. THE Context_Builder SHALL deduplicate overlapping text when multiple selected child chunks share the same Parent_Chunk.
3. THE Context_Builder SHALL assemble a structured evidence block where each source is labelled with its document filename, page number, section path, and a source identifier (S1, S2, …).
4. THE Context_Builder SHALL enforce a total evidence token budget of `RAG_GENERATION_CONTEXT` (default 16,384) tokens, truncating lower-ranked sources when the budget is exceeded, preserving higher-ranked sources in full.
5. WHEN a chunk originated from a low-confidence OCR page, THE Context_Builder SHALL annotate that source with an OCR confidence warning in the evidence block.

---

### Requirement 11: Answerability Classification

**User Story:** As a user, I want the system to refuse to generate an answer when the retrieved evidence is insufficient, so that I am never misled by unsupported responses.

#### Acceptance Criteria

1. BEFORE invoking the Generator, THE Answerability_Classifier SHALL evaluate the assembled evidence against the query and produce a binary `answerable` decision plus a reason string.
2. WHEN `answerable` is `false`, THE System SHALL return the response: "I could not find sufficient evidence in the uploaded documents to answer this question." and SHALL NOT invoke the Generator.
3. THE Answerability_Classifier SHALL apply a minimum retrieval confidence threshold; WHEN the highest reranker score is below 0.30, THE Answerability_Classifier SHALL classify the query as unanswerable.
4. THE Answerability_Classifier decision and reason SHALL be stored in the Debug_Trace for every query.

---

### Requirement 12: Grounded Answer Generation

**User Story:** As a user, I want the system to generate answers using only the retrieved evidence, so that every claim is traceable to a source document.

#### Acceptance Criteria

1. WHEN generating an answer, THE Generator SHALL use only the evidence block assembled by the Context_Builder and SHALL NOT draw on parametric knowledge not present in the evidence.
2. THE Generator SHALL cite every important factual claim using inline source identifiers (e.g., [S1], [S2]) that correspond to entries in the evidence block.
3. THE Generator SHALL preserve exact numbers, dates, amounts, names, identifiers, and section references as they appear in the evidence, without rounding, reformulating, or inventing values.
4. WHEN the evidence contains conflicting statements from different documents, THE Generator SHALL present both statements with their respective source identifiers and explicitly note the conflict rather than choosing one silently.
5. THE Generator SHALL use `temperature: 0.1` and `top_p: 0.9` by default to minimise variation in factual claims.
6. WHEN the active model is `qwen3.5:9b`, THE Generator SHALL use a context window of `RAG_GENERATION_CONTEXT` (16,384 tokens); WHEN the active model is `qwen3.5:4b`, THE Generator SHALL use `RAG_DEVELOPMENT_CONTEXT` (8,192 tokens).
7. THE Generator SHALL stream answer tokens to the Frontend using Server-Sent Events.

---

### Requirement 13: Claim-Level Citation Verification

**User Story:** As a developer, I want every claim in the generated answer verified against its cited source, so that unsupported claims are removed before the answer reaches the user.

#### Acceptance Criteria

1. WHEN an answer is generated, THE Claim_Verifier SHALL parse the answer into discrete factual claims.
2. FOR EACH claim, THE Claim_Verifier SHALL verify that the cited source chunk(s) contain text that entails or directly supports the claim.
3. WHEN a claim is not supported by its cited source, THE Claim_Verifier SHALL mark it as `unsupported` and exclude it from the final Answer or replace it with an insufficient-evidence note.
4. THE Claim_Verifier SHALL extract all numeric values, dates, and named identifiers from the answer and confirm that each value appears verbatim in at least one cited source chunk; WHEN a value is absent, THE Claim_Verifier SHALL flag it as a numeric mismatch.
5. WHEN at least one `unsupported` claim is detected, THE System SHALL include a verification summary in the Answer indicating the number of claims removed and the reason.
6. THE Claim_Verifier SHALL produce a structured verification report containing `supported_claims`, `unsupported_claims`, `numeric_mismatches`, and `conflicts` for every query, stored in the Debug_Trace.

---

### Requirement 14: Conflict Detection Between Documents

**User Story:** As a user, I want the system to alert me when different documents contain contradictory information, so that I can make an informed decision about which source to trust.

#### Acceptance Criteria

1. WHEN the assembled evidence contains two or more source chunks from different documents that contain statements about the same entity or rule that cannot simultaneously be true, THE System SHALL detect this as a Conflict.
2. WHEN a Conflict is detected, THE Generator SHALL include both conflicting statements in the answer, each labelled with its source identifier, document name, and page number.
3. WHEN a Conflict is detected, THE Answer SHALL contain an explicit conflict notice stating that the documents disagree and recommending that the user verify against the primary authoritative source.
4. THE Claim_Verifier SHALL record all detected Conflicts in the structured verification report.

---

### Requirement 15: REST API Endpoints

**User Story:** As a frontend developer, I want a stable REST API, so that the Next.js frontend can upload documents, query the system, and display results.

#### Acceptance Criteria

1. THE System SHALL expose `POST /api/documents` accepting multipart form-data with a PDF file field and returning `{document_id, status, filename}` with HTTP 202 on success.
2. THE System SHALL expose `GET /api/documents` returning a paginated list of documents with fields: `document_id`, `filename`, `status`, `page_count`, `chunk_count`, `created_at`.
3. THE System SHALL expose `GET /api/documents/{id}/status` returning `{document_id, status, stage, progress_pct, error}`.
4. THE System SHALL expose `DELETE /api/documents/{id}` which removes the document record, deletes all associated Qdrant vectors, and returns HTTP 204 on success.
5. THE System SHALL expose `POST /api/query` accepting `{question, document_ids?, search_mode?, include_debug?}` and returning `{answer, citations, verification_summary, debug_trace?}`.
6. THE System SHALL expose `GET /api/documents/{id}/pages/{page}` returning a rendered PNG image of the specified page at a minimum resolution of 150 DPI for the PDF viewer.
7. THE System SHALL expose `POST /api/debug/retrieve` accepting a query and returning the full Debug_Trace including dense results, sparse results, fusion results, reranked results, and answerability decision.
8. THE System SHALL expose `POST /api/evaluations/run` accepting an evaluation dataset reference and returning `{job_id}` while running the evaluation asynchronously.
9. WHEN any endpoint receives a request with missing or malformed required fields, THE System SHALL return HTTP 422 with a structured error body listing each invalid field and its validation message.

---

### Requirement 16: Metadata Storage and Data Model

**User Story:** As a developer, I want a relational metadata store that tracks every document, chunk, conversation, and evaluation result, so that the system can audit, recover, and report accurately.

#### Acceptance Criteria

1. THE Metadata_Store SHALL maintain a `documents` table containing: `document_id`, `filename`, `sha256_hash`, `file_path`, `status`, `stage`, `page_count`, `parser_version`, `embedding_model`, `embedding_version`, `created_at`, `updated_at`, `error_message`.
2. THE Metadata_Store SHALL maintain a `chunks` table containing: `chunk_id`, `document_id`, `parent_chunk_id`, `page_start`, `page_end`, `section_path`, `chunk_type`, `text`, `token_count`, `language`, `ocr_confidence`, `is_duplicate`, `canonical_chunk_id`, `created_at`.
3. THE Metadata_Store SHALL maintain a `conversations` table and a `messages` table with foreign-key relationships, storing the user query, rewritten query, generated answer, verification summary, and timestamps.
4. THE Metadata_Store SHALL maintain a `retrieval_logs` table storing the Debug_Trace JSON for every query.
5. THE Metadata_Store SHALL maintain an `evaluation_results` table storing dataset name, run timestamp, metric values, and per-question results.
6. THE System SHALL use Alembic for all schema migrations, and SHALL NOT modify schema directly; EVERY schema change SHALL be introduced through a versioned migration file.

---

### Requirement 17: Frontend Upload Interface

**User Story:** As a user, I want a clean upload interface, so that I can add PDF files to the system and monitor their ingestion progress.

#### Acceptance Criteria

1. THE Frontend SHALL provide a drag-and-drop upload area and a file picker button that accept PDF files and visually reject non-PDF files with an inline error message.
2. WHEN a file is uploading, THE Frontend SHALL display an upload progress bar showing bytes transferred and total file size.
3. WHEN a document is ingested, THE Frontend SHALL poll `GET /api/documents/{id}/status` at 2-second intervals and display the current stage name and percentage until the status reaches `ready` or `failed`.
4. WHEN ingestion reaches `failed`, THE Frontend SHALL display the error message and a retry button.
5. THE Frontend SHALL display the list of uploaded documents with status indicators showing `pending`, `processing`, `ready`, and `failed` states using distinct visual styling.

---

### Requirement 18: Frontend Query and Answer Interface

**User Story:** As a user, I want to type a question and receive a streamed, cited answer with a clickable source panel, so that I can verify every claim against the original document.

#### Acceptance Criteria

1. THE Frontend SHALL provide a text input for questions and a submit button that is disabled while a query is in progress.
2. WHEN the Generator begins streaming, THE Frontend SHALL render answer tokens progressively as they arrive via Server-Sent Events.
3. WHEN an answer is complete, THE Frontend SHALL render inline citation markers (e.g., [S1]) as clickable links that scroll the source panel to the corresponding citation entry.
4. WHEN a citation is clicked, THE Frontend SHALL open the PDF viewer, navigate to the cited page, and highlight the bounding-box region of the supporting excerpt when bounding-box data is available.
5. THE Frontend SHALL display, for each citation: document filename, page number, section path, and the supporting text excerpt.
6. WHEN a Conflict is present in the answer, THE Frontend SHALL render the conflict notice in a visually distinct style (e.g., amber warning box).
7. WHEN the answer is a refusal, THE Frontend SHALL display the refusal message in a distinct style and SHALL NOT display a source panel.

---

### Requirement 19: PDF Page Viewer

**User Story:** As a user, I want to view the original PDF page alongside retrieved citations, so that I can validate that the answer text matches the source document.

#### Acceptance Criteria

1. THE Frontend SHALL embed a PDF viewer powered by PDF.js that displays individual pages of uploaded documents.
2. WHEN a citation is activated, THE Frontend SHALL navigate the viewer to the cited page number within 500 ms.
3. WHERE bounding-box coordinates are available for a citation, THE Frontend SHALL render a semi-transparent highlight overlay on the supporting text region.
4. THE Frontend SHALL provide previous-page and next-page navigation controls within the viewer.

---

### Requirement 20: Evaluation Framework

**User Story:** As a developer, I want an automated evaluation pipeline that measures retrieval and generation quality against a labelled dataset, so that I can detect regressions before deploying pipeline changes.

#### Acceptance Criteria

1. THE Evaluation_Runner SHALL accept a JSON dataset file where each entry contains: `question`, `expected_answer`, `supporting_documents`, `supporting_pages`, `relevant_chunk_ids`, `answerable` label.
2. THE Evaluation_Runner SHALL compute Recall@K, Precision@K, Mean Reciprocal Rank (MRR), and nDCG@K for K ∈ {5, 10, 20} against the `relevant_chunk_ids` ground truth for every question.
3. THE Evaluation_Runner SHALL compute answer correctness, faithfulness, citation correctness, citation completeness, unsupported-claim rate, refusal correctness, and numeric accuracy for every question.
4. THE Evaluation_Runner SHALL measure and record ingestion latency per page, embedding latency per chunk, retrieval latency per query, reranking latency per query, and time-to-first-token per query.
5. WHEN an evaluation run completes, THE Evaluation_Runner SHALL write a JSON report to the `evaluation/reports/` directory containing per-question results and aggregate metric summaries.
6. WHEN a pipeline component changes (parser, chunker, embedding model, fusion method, reranker, prompt, or generator), THE System SHALL support running the full evaluation suite before the change is accepted.
7. THE System SHALL expose `POST /api/evaluations/run` which triggers the Evaluation_Runner asynchronously and returns a `job_id`; THE System SHALL expose `GET /api/evaluations/{job_id}` which returns the run status and, when complete, the report URL.

---

### Requirement 21: Retrieval Debug Interface

**User Story:** As a developer, I want a debug view that shows every stage of the retrieval pipeline for a given query, so that I can diagnose retrieval failures without instrumenting production code.

#### Acceptance Criteria

1. THE Frontend SHALL provide a retrieval debug panel, accessible via a toggle, that sends the query to `POST /api/debug/retrieve` and displays the full Debug_Trace.
2. THE Debug_Trace SHALL show, for each candidate chunk: dense rank, sparse rank, RRF score, reranker score, final inclusion decision, source document, page number, and a text excerpt.
3. THE Debug_Trace SHALL show the original query, rewritten query (if any), query classification, answerability decision, and the assembled evidence token count.
4. WHEN a chunk was excluded after reranking, THE Debug_Trace SHALL show it in a visually distinct style indicating exclusion, alongside its reranker score.

---

### Requirement 22: Observability and Structured Logging

**User Story:** As a developer, I want structured logs and per-query retrieval traces stored persistently, so that I can reproduce and analyse any query result after the fact.

#### Acceptance Criteria

1. THE System SHALL emit structured JSON log entries for every significant event: document upload, ingestion stage transitions, query received, retrieval completed, generation completed, and errors.
2. EVERY log entry SHALL include a timestamp, log level, component name, document_id or query_id where applicable, and a human-readable message.
3. THE System SHALL persist the full Debug_Trace for every query in the `retrieval_logs` table of the Metadata_Store.
4. THE System SHALL log model name, context-window size, temperature, and total prompt token count for every generation request.

---

### Requirement 23: Performance and Resource Management

**User Story:** As a developer, I want the system to operate within the hardware constraints of 24 GB RAM and 4 GB VRAM, so that it remains responsive during ingestion and query answering.

#### Acceptance Criteria

1. THE System SHALL NOT load the Reranker and the Generator simultaneously on the GPU when combined VRAM would exceed 4 GB; THE System SHALL sequence model usage to avoid exceeding this limit.
2. THE Embedder SHALL use a default batch size of 4 chunks (`RAG_EMBEDDING_BATCH_SIZE`) and SHALL release batch memory between iterations.
3. THE System SHALL support query embedding caching keyed on the query string so that repeated identical queries do not re-invoke the Embedder.
4. WHEN a document's SHA256_Hash matches an already-ingested document with status `ready`, THE System SHALL return the existing document record and SHALL NOT re-embed or re-index the content.
5. THE Ingestion_Worker SHALL support incremental processing such that if ingestion is interrupted, completed stages are not re-executed on restart.
6. WHILE the Ingestion_Worker is active, THE System SHALL continue to serve query requests using already-indexed documents without degradation.
7. THE System SHALL support a configurable `keep_alive` duration for Ollama model loading; the embedding model `keep_alive` SHALL default to 5 minutes and the generation model `keep_alive` SHALL default to 15 minutes during active sessions.

---

### Requirement 24: Privacy and Local-Only Operation

**User Story:** As a user, I want all my documents, queries, and answers to remain on my local machine, so that sensitive information is never transmitted to external services.

#### Acceptance Criteria

1. THE System SHALL make no outbound network connections to external APIs, cloud services, or telemetry endpoints during normal operation.
2. THE System SHALL route all model inference requests exclusively to the local Ollama instance at `OLLAMA_BASE_URL` (default `http://localhost:11434`).
3. THE System SHALL store all uploaded PDF files, parsed artefacts, embeddings, and metadata exclusively on the local filesystem paths configured in the application settings.
4. THE System SHALL not include any analytics, crash-reporting, or usage-telemetry libraries in the production build.

---

### Requirement 25: Parser and Chunk Round-Trip Fidelity

**User Story:** As a developer, I want to verify that parsed and chunked text faithfully represents the original PDF content, so that retrieval is not based on garbled or truncated text.

#### Acceptance Criteria

1. THE Parser SHALL preserve the complete textual content of every native-text page without dropping sentences or paragraphs.
2. WHEN a chunk's text is re-embedded using the same Embedder configuration, THE Embedder SHALL produce a vector with cosine similarity ≥ 0.9999 to the originally stored vector, confirming embedding determinism.
3. THE Chunker SHALL produce chunks such that concatenating all child chunks for a given page (in order) reconstructs the page text with only structural-whitespace differences (no missing words or transposed sentences).
4. FOR ALL valid parsed documents, serialising the Docling document object to its export format and re-importing it SHALL produce a document structure equivalent to the original parsed result (round-trip property).

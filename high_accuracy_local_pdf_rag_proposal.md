# High-Accuracy Local PDF RAG System Proposal

## 1. Project Title

**High-Accuracy, Low-Hallucination Local PDF RAG System**

## 2. Executive Summary

This project will build a fully local Retrieval-Augmented Generation system for asking questions across 10 to 20 uploaded PDF files.

The system will prioritize:

- High retrieval accuracy
- Fast search
- Strong source traceability
- Low hallucination
- Offline operation
- No paid APIs
- No document upload to external services
- Support for text PDFs, scanned PDFs, tables, and complex layouts

The revised architecture replaces a basic prototype stack with a stronger production design:

- Docling for structure-aware PDF parsing
- Selective OCR for scanned or broken pages
- Layout-aware, hierarchical chunking
- Qdrant for persistent hybrid search
- Dense and sparse retrieval
- Reciprocal Rank Fusion
- A dedicated cross-encoder reranker
- Query classification and rewriting
- Evidence-level answer generation
- Citation and claim verification
- Evaluation gates and regression tests

The system cannot guarantee mathematically zero hallucination. No current generative model can provide that guarantee. The practical target is:

```text
No unsupported answer is silently shown as fact.
```

---

## 3. Problem Statement

Important information is often distributed across multiple PDF documents such as policies, technical manuals, reports, research papers, government documents, contracts, forms, and scanned records.

Finding an answer manually becomes difficult when:

- There are 10 to 20 PDFs.
- Each PDF contains many pages.
- The answer may appear anywhere.
- The answer may require evidence from several pages.
- The answer may require comparison across documents.
- Some PDFs are scanned.
- Some PDFs contain complex tables.
- Some PDFs use multi-column layouts.
- Users do not know which file contains the answer.
- Confidential files cannot be uploaded to online AI services.
- Paid API usage is not acceptable.
- A local model may generate unsupported answers when retrieval is weak.

The goal is to build a system where users upload multiple PDFs, ask natural-language questions, and receive grounded answers with exact document and page references.

---

## 4. Primary Design Goals

### 4.1 Accuracy

The system must retrieve the correct evidence before answer generation begins.

### 4.2 Low Hallucination

The model must answer only from retrieved evidence. Unsupported claims must be removed or rejected.

### 4.3 Speed

Search should remain fast after ingestion, even with thousands of pages and tens of thousands of chunks.

### 4.4 Traceability

Every important factual statement should link to:

- Document name
- Page number
- Section title
- Chunk identifier
- Supporting excerpt

### 4.5 Privacy

All files, embeddings, questions, and answers must remain on the local machine.

### 4.6 Maintainability

Parsing, indexing, retrieval, reranking, generation, verification, and evaluation must remain separate modules.

---

## 5. Revised System Architecture

```text
Next.js Frontend
        |
        | REST API and streaming
        v
Python FastAPI Backend
        |
        +-- Upload and document management
        +-- Docling parsing pipeline
        +-- OCR and table extraction
        +-- Layout-aware chunking
        +-- Embedding service
        +-- Sparse retrieval service
        +-- Qdrant hybrid search
        +-- Query classification
        +-- Query rewriting and expansion
        +-- Reranking
        +-- Context assembly
        +-- Local answer generation
        +-- Claim and citation verification
        +-- Evaluation and observability
        |
        +-- SQLite or PostgreSQL metadata
        +-- Local filesystem
        +-- Qdrant local server
        +-- Local model runtime
```

---

## 6. Final Technology Stack

### 6.1 Backend

- Python 3.12+
- FastAPI
- Pydantic
- SQLAlchemy
- Alembic
- Uvicorn
- Structured logging
- Background job worker

### 6.2 Frontend

- Next.js
- TypeScript
- Tailwind CSS
- PDF.js for page preview
- Server-Sent Events or WebSocket streaming

### 6.3 PDF Parsing

**Primary parser: Docling**

Docling is preferred over plain text extraction because it supports document structure, OCR, tables, images, layout information, reading order, structured output, and accurate table extraction.

PyMuPDF should remain as a lightweight fallback and page-rendering utility.

### 6.4 Vector and Hybrid Search

**Primary vector database: Qdrant**

Qdrant is preferred over raw FAISS because it provides:

- Persistent collections
- Metadata filtering
- Dense vectors
- Sparse vectors
- Hybrid queries
- Reciprocal Rank Fusion
- Distribution-based score fusion
- HNSW indexing
- Quantization support
- Multivector support
- Document and page filtering
- Easier updates and deletions

FAISS remains excellent for pure vector similarity search, but it requires more custom code for metadata, filtering, deletion, hybrid retrieval, persistence, and index maintenance.

### 6.5 Metadata Storage

Use SQLite for a single-user desktop version.

Use PostgreSQL if the system later becomes multi-user or server-based.

Store:

- Documents
- File hashes
- Pages
- Chunks
- Parent-child relationships
- Processing status
- Conversations
- Messages
- Retrieval logs
- Evaluation results
- Model configuration
- Index version

### 6.6 Local Model Runtime

Support two runtime modes.

**Simple development mode:** Ollama

**Optimized deployment mode:** llama.cpp server

Use vLLM only for a machine with a suitable NVIDIA GPU and enough VRAM.

---

## 7. Recommended Model Strategy

A single model should not perform every task. Use separate models for embedding, sparse retrieval, reranking, answer generation, and optional verification.

### 7.1 Dense Embedding Model

**Balanced option: Qwen3-Embedding-0.6B**

Reasons:

- Strong multilingual retrieval
- 32K sequence support
- Instruction-aware embeddings
- Matryoshka Representation Learning support
- Practical local size
- Suitable for English, Hindi, and mixed-language content

**Higher-accuracy option: Qwen3-Embedding-4B**

Use this when slower ingestion is acceptable and hardware can support it.

### 7.2 Dense, Sparse, and Multivector Alternative

**BGE-M3** supports:

- Dense embeddings
- Sparse lexical weights
- ColBERT-style multivectors

This enables a three-channel retrieval system:

```text
Dense semantic retrieval
Sparse lexical retrieval
Late-interaction multivector retrieval
```

### 7.3 Sparse Retrieval

Start with BM25 or Qdrant sparse vectors.

Later compare:

- BM25
- BGE-M3 lexical weights
- SPLADE-compatible sparse retrieval

Sparse retrieval is required for exact terms such as IDs, dates, section numbers, legal clauses, product codes, names, acronyms, and rare technical terms.

### 7.4 Reranker

**Balanced option: Qwen3-Reranker-0.6B**

**Higher-accuracy option: Qwen3-Reranker-4B**

The reranker should process only the top candidate set.

```text
Dense retrieval: top 40
Sparse retrieval: top 40
Fusion: top 50 unique candidates
Reranker: score top 30 to 50
Final context: top 6 to 10
```

### 7.5 Answer Generation Model

**Balanced local baseline: Qwen3.5 9B**

**Faster fallback: Qwen3.5 4B**

Use a larger Qwen3.5 or Qwen3.6 model only when the machine has enough RAM or GPU memory.

A strong retrieval pipeline with a moderate model will usually produce more reliable RAG answers than a weak retrieval pipeline with a larger model.

---

## 8. Hardware-Aware Deployment Profiles

### 8.1 Balanced Laptop Profile

Suitable for a laptop with about 24 GB system RAM.

```text
Parser: Docling standard pipeline
Embedding: Qwen3-Embedding-0.6B
Sparse retrieval: BM25 or Qdrant sparse vectors
Vector database: Qdrant
Reranker: Qwen3-Reranker-0.6B
Generator: Qwen3.5 9B quantized
Runtime: Ollama or llama.cpp
```

### 8.2 Fast CPU Profile

```text
Parser: Docling with selective OCR
Embedding: smaller quantized embedding model
Sparse retrieval: BM25
Vector database: Qdrant
Reranker: Qwen3-Reranker-0.6B with fewer candidates
Generator: Qwen3.5 4B quantized
```

### 8.3 Maximum Accuracy Workstation Profile

```text
Parser: Docling accurate table mode
Embedding: Qwen3-Embedding-4B or 8B
Sparse retrieval: learned sparse retrieval
Optional late interaction: BGE-M3 ColBERT vectors
Vector database: Qdrant
Reranker: Qwen3-Reranker-4B or 8B
Generator: larger local Qwen model
Optional verifier: separate local model
Runtime: vLLM with GPU
```

---

## 9. Document Ingestion Pipeline

### 9.1 Upload Validation

For each PDF:

1. Validate MIME type.
2. Validate the file signature.
3. Enforce file-size and page-count limits.
4. Calculate SHA-256.
5. Reject exact duplicates.
6. Generate a stable document ID.
7. Save the original file.
8. Create an ingestion job.
9. Track every processing stage.

### 9.2 Parsing Strategy

Use Docling as the primary parser.

Extract:

- Headings
- Paragraphs
- Lists
- Tables
- Captions
- Page numbers
- Reading order
- Bounding boxes
- Images
- Section hierarchy
- OCR confidence where available

Use selective OCR:

```text
Native text available and valid:
Use native text.

Text missing or corrupted:
Run OCR.

Page contains mixed text and image regions:
Use layout-aware extraction.

Complex table:
Use accurate table mode.
```

Do not force OCR on every page. It increases ingestion time and can replace good native text with lower-quality OCR output.

### 9.3 OCR Quality Control

Store:

- OCR engine
- OCR language
- OCR confidence
- Page resolution
- Whether OCR replaced native text
- Low-confidence spans

Low-confidence pages should reduce answer confidence and remain visible in the source panel.

### 9.4 Table Preservation

Do not flatten every table into uncontrolled text.

Store tables as:

- Markdown
- Structured rows and columns
- Table title
- Page number
- Nearby paragraph context
- Cell coordinates when available

Create table-specific chunks containing the table title, headers, relevant rows, page number, and nearby explanation.

### 9.5 Header and Footer Detection

Detect and remove repeated headers, footers, page numbers, confidentiality lines, and repeated chapter names from retrieval text.

Keep the original page text for display and audit.

### 9.6 Document Versioning

Store parser and index versions. Reprocess a document when the parser, embedding model, chunking method, OCR engine, or retrieval schema changes.

---

## 10. Chunking Strategy

Fixed-size chunking alone is not sufficient. Use hierarchical, structure-aware chunking.

### 10.1 Child Chunks

Small chunks for precise retrieval.

```text
Target: 250 to 500 tokens
```

### 10.2 Parent Chunks

Larger sections for final context expansion.

```text
Target: 800 to 1,500 tokens
```

### 10.3 Document and Section Summaries

Create short summaries for document routing and broad questions. Do not use generated summaries as a substitute for source evidence.

### 10.4 Chunk Boundaries

Prefer boundaries based on headings, paragraphs, lists, tables, sections, pages, and semantic topic shifts.

Avoid splitting:

- Table rows from headers
- Definitions from explanations
- Numbered clauses from subclauses
- Headings from the paragraph below
- Sentences in the middle

### 10.5 Parent-Child Retrieval

Retrieve child chunks for precision, then expand the best results to parent sections for generation.

### 10.6 Contextualized Chunks

Add limited structural context before embedding.

```text
Document: Employee Leave Policy
Section: Earned Leave
Subsection: Carry Forward Rules

Chunk text:
Permanent employees may carry forward...
```

### 10.7 Chunk Deduplication

Detect exact duplicates, near duplicates, repeated boilerplate, and repeated templates. Store duplicate relationships instead of indexing identical content repeatedly.

---

## 11. Indexing Design

Each indexed point should contain:

```json
{
  "chunk_id": "chunk_123",
  "document_id": "doc_12",
  "filename": "policy.pdf",
  "page_start": 17,
  "page_end": 18,
  "section_path": [
    "Leave Policy",
    "Earned Leave",
    "Carry Forward"
  ],
  "chunk_type": "paragraph",
  "parent_chunk_id": "parent_88",
  "text": "The employee may carry forward...",
  "language": "en",
  "ocr_confidence": 0.98,
  "parser_version": "x",
  "embedding_version": "x"
}
```

Qdrant should store dense vectors, sparse vectors, optional multivectors, and metadata payload.

The SQL database remains the authoritative metadata store.

---

## 12. Retrieval Pipeline

### 12.1 Query Analysis

Classify the question as one of the following:

- Direct factual
- Exact identifier lookup
- Definition
- Comparison
- Summary
- Multi-document aggregation
- Table lookup
- Numeric or date question
- Procedural question
- Follow-up question
- Unanswerable or unrelated

### 12.2 Query Rewriting

Rewrite conversational follow-ups into standalone queries while preserving the original question.

Example:

```text
User:
What about contract staff?

Standalone query:
What are the earned leave carry-forward rules for contract staff?
```

### 12.3 Multi-Query Expansion

Generate a small number of alternative queries only when needed.

```text
Original:
What is the maximum leave balance?

Alternatives:
Maximum accumulated earned leave
Earned leave carry-forward limit
Maximum leave days an employee can retain
```

### 12.4 Hybrid Retrieval

Run at least two channels:

```text
Dense semantic search
Sparse lexical search
```

Optional third channel:

```text
Late-interaction multivector search
```

### 12.5 Fusion

Use Reciprocal Rank Fusion as the safe default.

Use weighted RRF only after creating a labelled evaluation set.

Do not combine raw dense and sparse scores directly without normalization because their score scales differ.

### 12.6 Metadata Filters

Apply filters for selected documents, date ranges, document category, language, department, version, or page range.

### 12.7 Candidate Retrieval

Initial balanced settings:

```text
Dense top K: 40
Sparse top K: 40
Optional multivector top K: 30
Fusion result: 50 unique candidates
Reranker input: 30 to 50 candidates
Final evidence: 6 to 10 chunks
```

Tune these values using evaluation data.

### 12.8 Reranking

Rerank using the query, chunk text, section title, document title, and optional parent context.

### 12.9 Diversity and Redundancy Control

Use Maximum Marginal Relevance, per-document caps, near-duplicate removal, adjacent-chunk merging, and source diversity rules.

### 12.10 Context Expansion

After reranking:

1. Select the strongest child chunks.
2. Load their parent sections.
3. Include adjacent chunks only when required.
4. Remove duplicate text.
5. Preserve source labels.
6. Fit the evidence into a controlled token budget.

---

## 13. Answer Generation Pipeline

The generator must receive structured evidence.

```text
SOURCE S1
Document: Employee Policy.pdf
Pages: 17 to 18
Section: Earned Leave > Carry Forward
Text:
...

SOURCE S2
Document: Contract Rules.pdf
Page: 9
Section: Leave Rules
Text:
...
```

### 13.1 Generation Rules

The model must:

- Use only supplied evidence.
- Answer directly.
- Cite every important claim.
- Avoid unsupported inference.
- Separate facts from interpretation.
- Report conflicts between documents.
- State when evidence is insufficient.
- Preserve exact numbers and dates.
- Avoid citing a source that does not support the claim.

### 13.2 Conservative Settings

```text
Temperature: 0.0 to 0.2
Maximum answer length: bounded
Structured output: enabled where possible
```

Low temperature reduces variation, but it does not remove hallucination.

### 13.3 Extractive-First Answering

For high-risk questions:

1. Extract exact supporting statements.
2. Generate the answer from those statements.
3. Attach citations.
4. Verify each claim against the statements.

---

## 14. Hallucination Reduction Architecture

### 14.1 Retrieval Threshold

Do not generate an answer when retrieval confidence is too low.

### 14.2 Answerability Classifier

Before generation, determine whether the evidence can answer the question.

```json
{
  "answerable": false,
  "reason": "No retrieved passage states the required limit."
}
```

### 14.3 Claim-Level Citations

Break the answer into claims. Each claim must map to one or more source chunks.

### 14.4 Citation Entailment Check

Verify whether the cited source actually supports each claim.

Unsupported claims should be removed, rewritten, marked uncertain, or replaced with an insufficient-evidence response.

### 14.5 Numeric and Date Validation

Extract numbers, dates, names, and identifiers from the answer. Check whether each value appears in the cited evidence.

### 14.6 Conflict Detection

When documents disagree, return both statements, their sources, their dates or versions, and a clear conflict notice.

### 14.7 Strict Refusal Path

```text
I could not find sufficient evidence in the uploaded documents to answer this question.
```

### 14.8 Optional Second Model Verifier

A separate local verifier can inspect the question, answer, evidence, and citations.

```json
{
  "supported": true,
  "unsupported_claims": [],
  "missing_citations": [],
  "conflicts": []
}
```

The verifier is an additional quality gate, not a guarantee.

---

## 15. Citation Design

Every source should include:

- Filename
- Document ID
- Page number
- Section title
- Chunk ID
- Supporting excerpt
- Bounding box when available

The frontend should allow the user to click a citation, open the PDF, jump to the exact page, highlight the supporting region, and compare extracted text with the original page.

---

## 16. Performance Design

### 16.1 Ingestion Performance

- Process pages incrementally.
- Do not load every PDF into memory.
- Batch embeddings.
- Skip unchanged documents.
- Cache parser output.
- Cache OCR output.
- Cache embeddings by content hash.
- Use selective OCR.
- Use a job queue.
- Persist intermediate results.
- Resume failed ingestion jobs.

### 16.2 Search Performance

- Use Qdrant HNSW indexes.
- Use payload indexes for filters.
- Tune HNSW using evaluation.
- Limit reranking to candidate passages.
- Cache repeated query embeddings.
- Cache retrieval results.
- Merge adjacent chunks before generation.
- Avoid unnecessary context.

### 16.3 Inference Performance

- Use quantized models.
- Use prefix caching where supported.
- Use batch inference.
- Use GPU acceleration when available.
- Tune llama.cpp threads and batch size on CPU.
- Stream answer tokens to the frontend.

### 16.4 Concurrency

For a single-user desktop system:

- One ingestion worker
- One generation request at a time
- Concurrent retrieval operations
- Background document processing

For multi-user use:

- Separate model services
- Queue-based generation
- Multiple workers
- Resource limits
- Request cancellation

---

## 17. Evaluation Framework

A system cannot be called accurate without testing.

### 17.1 Evaluation Set

Prepare 100 to 300 representative questions containing:

- Easy factual questions
- Exact identifier questions
- Cross-page questions
- Cross-document comparisons
- Table questions
- Scanned-page questions
- Ambiguous questions
- Unanswerable questions
- Questions with conflicting sources

For every question, store the expected answer, supporting document, supporting page, relevant chunks, and answerability label.

### 17.2 Retrieval Metrics

Measure:

- Recall@K
- Precision@K
- Mean Reciprocal Rank
- nDCG@K
- Hit rate
- Source page recall
- Document recall
- Reranker improvement

Evidence recall is the most important early metric. If the correct passage is not retrieved, the generator cannot produce a reliable answer.

### 17.3 Generation Metrics

Measure:

- Answer correctness
- Faithfulness
- Citation correctness
- Citation completeness
- Unsupported claim rate
- Refusal correctness
- Numeric accuracy
- Conflict detection accuracy

### 17.4 Latency Metrics

Measure:

- Parsing time
- OCR time per page
- Embedding time
- Retrieval latency
- Reranking latency
- Time to first token
- Total answer latency
- Memory usage

### 17.5 Regression Testing

Every change to the parser, chunking, embedding model, fusion method, reranker, prompt, or generator must run against the evaluation set.

Do not replace a component only because it scores better on a public benchmark. Replace it when it improves the actual PDF evaluation set.

---

## 18. Observability

Store retrieval traces for every question.

```json
{
  "query": "What is the leave limit?",
  "rewritten_queries": [],
  "dense_results": [],
  "sparse_results": [],
  "fusion_results": [],
  "reranked_results": [],
  "selected_context": [],
  "answer": "",
  "citations": [],
  "verification": {}
}
```

Create a retrieval inspector showing dense rank, sparse rank, fusion rank, reranker score, final inclusion, source page, and answer support.

---

## 19. Recommended API Design

```http
POST /api/documents
GET /api/documents
GET /api/documents/{document_id}/status
DELETE /api/documents/{document_id}
POST /api/query
GET /api/documents/{document_id}/pages/{page_number}
POST /api/debug/retrieve
POST /api/evaluations/run
```

Example query request:

```json
{
  "question": "Compare the leave carry-forward rules.",
  "document_ids": [],
  "search_mode": "hybrid",
  "include_debug": false
}
```

---

## 20. Proposed Project Structure

```text
local-pdf-rag/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── ingestion/
│   │   │   ├── parser/
│   │   │   ├── ocr/
│   │   │   ├── tables/
│   │   │   ├── cleaning/
│   │   │   └── chunking/
│   │   ├── indexing/
│   │   │   ├── dense/
│   │   │   ├── sparse/
│   │   │   ├── multivector/
│   │   │   └── qdrant/
│   │   ├── retrieval/
│   │   │   ├── query_analysis/
│   │   │   ├── rewriting/
│   │   │   ├── hybrid/
│   │   │   ├── fusion/
│   │   │   ├── reranking/
│   │   │   └── context_builder/
│   │   ├── generation/
│   │   ├── verification/
│   │   ├── evaluation/
│   │   ├── observability/
│   │   ├── database/
│   │   └── workers/
│   ├── tests/
│   ├── data/
│   └── pyproject.toml
├── frontend/
│   ├── app/
│   ├── components/
│   ├── features/
│   ├── lib/
│   └── package.json
├── infrastructure/
│   ├── docker/
│   ├── qdrant/
│   └── scripts/
├── evaluation/
│   ├── datasets/
│   ├── expected_answers/
│   └── reports/
└── docs/
```

---

## 21. Development Phases

### Phase 1: Retrieval Foundation

- Document upload
- Docling parsing
- Structured chunking
- Qdrant indexing
- Dense retrieval
- Source metadata
- Basic answer generation

### Phase 2: Accuracy Upgrade

- Sparse retrieval
- Hybrid fusion
- Qwen3 reranking
- Parent-child retrieval
- Query rewriting
- Duplicate control
- Table-aware chunks

### Phase 3: Hallucination Controls

- Answerability detection
- Claim-level citations
- Citation verification
- Numeric validation
- Conflict detection
- Strict refusal path

### Phase 4: Evaluation

- Ground-truth question set
- Retrieval metrics
- Generation metrics
- Latency benchmarks
- Regression tests
- Retrieval debugging interface

### Phase 5: Advanced Document Understanding

- Image-region extraction
- Multimodal retrieval
- Chart and diagram support
- Visual page retrieval
- Qwen3-VL embedding and reranking experiments

---

## 22. Final Recommended Stack

```text
Frontend:
Next.js, TypeScript, Tailwind CSS, PDF.js

Backend:
Python, FastAPI, Pydantic, SQLAlchemy

PDF parsing:
Docling primary
PyMuPDF fallback and rendering
Selective OCR
Accurate table extraction

Vector database:
Qdrant

Retrieval:
Qwen3-Embedding-0.6B or 4B
BM25 or learned sparse retrieval
Reciprocal Rank Fusion
Optional BGE-M3 multivector retrieval

Reranking:
Qwen3-Reranker-0.6B
Qwen3-Reranker-4B for stronger hardware

Generation:
Qwen3.5 9B for balanced local use
Qwen3.5 4B for faster CPU use
Larger model only when hardware supports it

Runtime:
Ollama for convenience
llama.cpp for optimized local deployment
vLLM for suitable GPU servers

Storage:
SQLite for single-user mode
PostgreSQL for future multi-user mode
Local filesystem for original PDFs and parser artifacts

Quality controls:
Answerability classification
Claim-level citations
Citation entailment checks
Numeric and date validation
Conflict detection
Strict refusal
Evaluation dataset
Regression tests
```

---

## 23. Key Architecture Decisions

### Replace FAISS with Qdrant

Qdrant supports persistent hybrid retrieval, filtering, sparse vectors, payloads, multivectors, fusion, deletion, and updates in one system.

### Replace basic extraction with Docling

RAG accuracy depends on reading order, tables, sections, and page structure. Plain text extraction can lose important meaning.

### Add a dedicated reranker

Initial retrieval produces candidates. The reranker determines which passages can actually answer the question.

### Use parent-child chunks

Small chunks improve retrieval precision. Parent chunks provide enough context for reliable generation.

### Add verification after generation

Prompt instructions alone cannot prevent unsupported claims.

### Build an evaluation dataset early

There is no universal best model or chunk size. The correct choice must be measured on the actual PDFs and questions.

---

## 24. Expected Outcome

The completed system will:

- Process 10 to 20 PDFs locally.
- Search thousands of pages quickly.
- Understand normal text, scans, layouts, and tables.
- Retrieve evidence using semantic and lexical search.
- Rerank results before generation.
- Answer using only supported evidence.
- Cite the exact file and page.
- Refuse when evidence is insufficient.
- Detect conflicting documents.
- Keep all data private.
- Require no paid model or online API.
- Support future scaling without replacing the complete architecture.

---

## 25. Research Basis

This proposal is based on current official documentation and primary model publications reviewed in June 2026.

1. Qdrant Hybrid Queries  
   https://qdrant.tech/documentation/search/hybrid-queries/

2. Qdrant Documentation  
   https://qdrant.tech/documentation/

3. Docling CLI and PDF Pipeline Options  
   https://docling-project.github.io/docling/reference/cli/

4. Docling Full-Page OCR Example  
   https://docling-project.github.io/docling/_generated/examples/full_page_ocr/

5. Qwen3 Embedding and Reranking Paper  
   https://arxiv.org/abs/2506.05176

6. Qwen3-Embedding-0.6B Model Card  
   https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

7. Qwen3-Reranker-0.6B Model Card  
   https://huggingface.co/Qwen/Qwen3-Reranker-0.6B

8. BGE-M3 Model Card  
   https://huggingface.co/BAAI/bge-m3

9. llama.cpp Server Documentation  
   https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

10. Ollama Qwen3.5 Model Library  
    https://registry.ollama.com/library/qwen3.5

11. vLLM Documentation  
    https://docs.vllm.ai/

12. Qwen3-VL Embedding and Reranking Paper  
    https://arxiv.org/abs/2601.04720

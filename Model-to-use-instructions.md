# Model-to-use-instructions.md

## Purpose

This file defines which AI models the Local PDF RAG system should use, what each model is responsible for, and when to use each one.

The target machine has:

```text
RAM: 24 GB DDR5 4800 MHz
GPU: NVIDIA RTX 3050, 4 GB VRAM, 95 W TGP
Storage: 100 GB available on Gen 5 SSD
Runtime: Ollama for local generation and embeddings
```

The system must remain fully local and should not use paid APIs.

---

# 1. Final Recommended Model Stack

Use this model stack for the first production version:

```text
Answer generation:
qwen3.5:9b

Fast development and fallback generation:
qwen3.5:4b

Dense embeddings:
qwen3-embedding:4b

Reranking:
Qwen/Qwen3-Reranker-0.6B

Sparse retrieval:
BM25 or Qdrant sparse vectors

PDF parsing:
Docling

Vector database:
Qdrant
```

---

# 2. Models to Download Through Ollama

Run:

```powershell
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:4b
```

Verify:

```powershell
ollama list
```

Expected approximate storage:

```text
qwen3.5:9b             6.6 GB
qwen3.5:4b             3.4 GB
qwen3-embedding:4b     2.5 GB
--------------------------------
Approximate total      12.5 GB
```

Do not download unnecessarily large models before the base system is working.

---

# 3. Main Answer Model

## Model

```text
qwen3.5:9b
```

## Responsibility

Use this model to generate the final user-facing answer after retrieval and reranking have completed.

It should receive:

- The user's question
- The final selected evidence chunks
- Source IDs
- Document names
- Page numbers
- Strict answer-generation instructions

## Use it when

Use `qwen3.5:9b` for:

- Final production answers
- Questions that require explanation
- Multi-document comparisons
- Questions involving several evidence chunks
- Policy interpretation based on retrieved evidence
- Summaries based on selected source sections
- Answers where higher quality matters more than speed
- Final accuracy testing
- Demonstrations and release builds

## Do not use it for

Do not use it for:

- Creating embeddings
- Searching vectors
- Reranking every retrieved chunk
- OCR
- PDF parsing
- Exact duplicate detection
- Simple metadata filtering
- Every development request during rapid testing

## Recommended settings

Start with:

```json
{
  "model": "qwen3.5:9b",
  "options": {
    "temperature": 0.1,
    "top_p": 0.9,
    "num_ctx": 16384
  }
}
```

Recommended context range:

```text
Development default: 8,192 tokens
Production default: 16,384 tokens
Maximum only after testing: 32,768 tokens
```

Do not use the full advertised model context by default. A larger context increases RAM use and reduces speed.

## Prompt rule

The answer model must be told:

```text
Use only the supplied evidence.
Do not answer from general knowledge.
Cite every important factual claim.
If the evidence is insufficient, say that the answer could not be found.
Do not invent names, dates, amounts, rules, filenames, or page numbers.
```

---

# 4. Fast Development Model

## Model

```text
qwen3.5:4b
```

## Responsibility

Use this model while building and debugging the application.

It provides faster responses and uses less memory than the 9B model.

## Use it when

Use `qwen3.5:4b` for:

- Backend API development
- Frontend integration testing
- Streaming-response testing
- Prompt-format testing
- Citation-output format testing
- Unit and integration testing
- Retrieval debugging
- Low-latency mode
- Systems where the 9B model is too slow
- Temporary fallback when memory is under pressure

## Do not use it when

Avoid using it as the final model for:

- Complex comparisons
- Long synthesis answers
- Questions requiring subtle reasoning
- Final accuracy benchmarks
- High-risk document interpretation

## Recommended settings

```json
{
  "model": "qwen3.5:4b",
  "options": {
    "temperature": 0.1,
    "top_p": 0.9,
    "num_ctx": 8192
  }
}
```

## Model selection rule

Use this configuration variable:

```env
RAG_GENERATION_MODEL=qwen3.5:9b
RAG_DEVELOPMENT_MODEL=qwen3.5:4b
```

In development mode:

```env
RAG_ACTIVE_MODEL=qwen3.5:4b
```

In production mode:

```env
RAG_ACTIVE_MODEL=qwen3.5:9b
```

---

# 5. Dense Embedding Model

## Model

```text
qwen3-embedding:4b
```

## Responsibility

Use this model to convert text into dense vectors.

It must embed:

- Every document chunk during ingestion
- Every user query during retrieval
- Optional document and section summaries
- Optional parent chunks

The same embedding model and configuration must be used for both document chunks and queries.

## Use it when

Use `qwen3-embedding:4b` for:

- Semantic document search
- Similar-meaning retrieval
- Multilingual retrieval
- English and Hindi document collections
- Searching when the question wording differs from the document wording
- Building the dense vector index
- Query embedding before Qdrant search

## Do not use it for

Do not use it for:

- Generating answers
- Reranking query and passage pairs
- OCR
- Keyword matching
- Chat responses
- Source verification

## Example use

```python
import ollama

response = ollama.embed(
    model="qwen3-embedding:4b",
    input=[
        "Employees may carry forward up to 30 days of earned leave."
    ],
)

embedding = response["embeddings"][0]
```

## Batch size

Start with:

```text
Embedding batch size: 4 to 8 chunks
```

Increase carefully after monitoring RAM and response time.

Because the RTX 3050 has only 4 GB VRAM, the embedding model may use both GPU and system RAM.

## Chunk length

Recommended initial chunk size:

```text
Child chunks: 250 to 500 tokens
Parent chunks: 800 to 1,500 tokens
```

Do not embed complete PDFs as one vector.

## Index consistency rule

If the embedding model changes:

```text
All existing vectors must be regenerated.
The Qdrant collection must be rebuilt or versioned.
```

Store the model name with the index:

```json
{
  "embedding_model": "qwen3-embedding:4b",
  "embedding_version": "v1"
}
```

---

# 6. Reranker Model

## Model

```text
Qwen/Qwen3-Reranker-0.6B
```

## Runtime

Use Python with `sentence-transformers`.

Install:

```powershell
uv add sentence-transformers torch transformers
```

## Responsibility

The reranker receives the user's query and a set of candidate passages.

It assigns a relevance score to every query-passage pair.

The reranker should run after dense and sparse retrieval.

## Use it when

Use the reranker for:

- Improving the order of retrieved chunks
- Removing semantically similar but irrelevant results
- Selecting the strongest evidence
- Cross-document questions
- Questions with many possible matches
- Questions involving exact policies, definitions, or conditions
- Final production retrieval
- Accuracy evaluation

## Do not use it for

Do not use it for:

- Searching the entire vector database
- Processing every stored chunk
- Generating embeddings
- Producing final answers
- OCR
- Parsing documents

## Recommended retrieval flow

```text
Dense retrieval: top 40
Sparse retrieval: top 40
Fusion: top 40 to 50 unique candidates
Reranker: score top 30 to 40 candidates
Final context: keep top 6 to 10 chunks
```

## GPU and batch settings

Start with:

```text
Device: CUDA when it fits
Batch size: 1 to 4
Maximum passage length: 1,024 to 2,048 tokens
```

If CUDA runs out of memory:

```text
Move the reranker to CPU.
Reduce batch size to 1.
Reduce maximum passage length.
```

## Important rule

Do not load the reranker and the 9B generation model on the GPU at the same time if this creates memory pressure.

Recommended request sequence:

```text
1. Run retrieval.
2. Run reranker.
3. Release or offload reranker resources if needed.
4. Run final generation model.
```

---

# 7. Sparse Retrieval

## Model

No generative model is needed for the first version.

Use:

```text
BM25
```

or:

```text
Qdrant sparse vectors
```

## Responsibility

Sparse retrieval finds exact terms.

Use it for:

- Names
- Dates
- Amounts
- Order numbers
- Section numbers
- IDs
- Acronyms
- Legal clauses
- Rare words
- Product codes
- Exact quotations

## Use it together with dense retrieval

Never rely only on dense embeddings.

Recommended flow:

```text
Dense semantic search
+
Sparse exact-term search
+
Reciprocal Rank Fusion
+
Reranking
```

---

# 8. Optional Future Retrieval Model

## Model

```text
BAAI/bge-m3
```

## Status

Do not add this in the first implementation.

Evaluate it later.

## Use it when

Consider BGE-M3 when the basic system is stable and you want to test:

- Dense vectors
- Learned sparse vectors
- ColBERT-style multivectors
- Multilingual retrieval
- A three-channel retrieval architecture

## Rule

Do not replace `qwen3-embedding:4b` only because BGE-M3 has more retrieval modes.

Create an evaluation dataset and compare:

- Recall@10
- Recall@20
- MRR
- nDCG
- Retrieval latency
- Memory consumption

Use whichever model performs better on the project's own PDFs.

---

# 9. Optional Vision Model

## Current decision

Do not add a separate vision model in the first version.

Use Docling and OCR first.

## Use a vision model later when

Add local vision processing only when the system must understand:

- Charts
- Diagrams
- Figures
- Images containing text
- Complex visual forms
- Scanned tables that OCR cannot reconstruct
- Questions whose answers depend on page layout

## Candidate

The current generation model family is multimodal, but visual document processing should remain a separate tested pipeline.

Do not send every PDF page image to the generation model. This will be slow and memory intensive.

Use vision only for pages identified as visually important.

---

# 10. Model Use by Pipeline Stage

| Pipeline stage | Model or tool | Purpose |
|---|---|---|
| PDF parsing | Docling | Extract text, layout, tables, and structure |
| OCR | Docling OCR backend | Read scanned pages |
| Text cleaning | Python rules | Remove repeated headers, footers, and noise |
| Chunking | Python and Docling structure | Create child and parent chunks |
| Dense embedding | `qwen3-embedding:4b` | Generate semantic vectors |
| Sparse indexing | BM25 or Qdrant sparse | Index exact terms |
| Dense search | `qwen3-embedding:4b` | Embed user query |
| Hybrid fusion | Qdrant RRF | Merge dense and sparse rankings |
| Reranking | `Qwen/Qwen3-Reranker-0.6B` | Select strongest evidence |
| Final answer | `qwen3.5:9b` | Generate grounded answer |
| Development answer | `qwen3.5:4b` | Faster testing |
| Claim verification | Rules first, optional local model later | Check citations, numbers, and support |

---

# 11. Which Model to Use for Each Question Type

## Simple factual question

Example:

```text
What is the maximum earned leave balance?
```

Use:

```text
Embedding: qwen3-embedding:4b
Sparse retrieval: enabled
Reranker: Qwen3-Reranker-0.6B
Generator: qwen3.5:4b or qwen3.5:9b
```

The 4B generator is acceptable when the evidence is short and direct.

## Exact identifier lookup

Example:

```text
What does order MPSEDC/HR/2026/104 state?
```

Use:

```text
Sparse retrieval: highest importance
Dense retrieval: enabled
Reranker: enabled
Generator: qwen3.5:4b
```

## Multi-document comparison

Example:

```text
Compare earned leave rules across all uploaded policy documents.
```

Use:

```text
Dense retrieval: enabled
Sparse retrieval: enabled
Document diversity: enabled
Reranker: enabled
Generator: qwen3.5:9b
Context: 16K tokens
```

## Summary question

Example:

```text
Summarize the employee promotion policy.
```

Use:

```text
Section routing: enabled
Parent chunks: enabled
Reranker: enabled
Generator: qwen3.5:9b
```

## Table question

Example:

```text
Which department received the highest budget?
```

Use:

```text
Docling table extraction
Table-specific chunks
Dense and sparse retrieval
Reranker
qwen3.5:9b
Numeric validation
```

## Follow-up question

Example:

```text
What about contract employees?
```

Use:

```text
Query rewriting
Previous question context
Dense and sparse retrieval
Reranker
qwen3.5:9b
```

Rewrite it into a standalone query before retrieval.

## Unanswerable question

Use:

```text
Retrieval threshold
Answerability check
No generation when evidence is insufficient
```

Return:

```text
I could not find sufficient evidence in the uploaded documents to answer this question.
```

---

# 12. Model Loading Strategy

The system should not keep every model active on the GPU.

Recommended sequence:

```text
During ingestion:
Load embedding model
Process chunks in batches
Unload when ingestion completes if required

During question answering:
Embed query
Run Qdrant retrieval
Run reranker
Release reranker GPU memory if required
Run generation model
```

Ollama may keep models loaded temporarily.

Use `keep_alive` carefully.

For systems with limited memory:

```text
Embedding keep_alive: short
Generation keep_alive: longer during active chat
Reranker: load only during reranking
```

---

# 13. Recommended Environment Variables

```env
OLLAMA_BASE_URL=http://localhost:11434

RAG_GENERATION_MODEL=qwen3.5:9b
RAG_DEVELOPMENT_MODEL=qwen3.5:4b
RAG_EMBEDDING_MODEL=qwen3-embedding:4b
RAG_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B

RAG_GENERATION_CONTEXT=16384
RAG_DEVELOPMENT_CONTEXT=8192

RAG_DENSE_TOP_K=40
RAG_SPARSE_TOP_K=40
RAG_FUSION_TOP_K=50
RAG_RERANK_TOP_K=30
RAG_FINAL_CONTEXT_CHUNKS=8

RAG_TEMPERATURE=0.1
RAG_EMBEDDING_BATCH_SIZE=4
RAG_RERANK_BATCH_SIZE=2
```

These values are starting points. Tune them using the evaluation dataset.

---

# 14. Model Fallback Rules

## If `qwen3.5:9b` is too slow

Use:

```text
qwen3.5:4b
```

## If the system runs out of RAM

Perform these steps:

```text
1. Reduce generation context.
2. Reduce final context chunks.
3. Use qwen3.5:4b.
4. Run reranker on CPU.
5. Reduce embedding and reranker batch sizes.
6. Prevent simultaneous model loading.
```

## If embedding ingestion is too slow

Perform these steps:

```text
1. Reduce embedding batch size if memory is the issue.
2. Process ingestion as a background job.
3. Cache embeddings by content hash.
4. Test qwen3-embedding:0.6b as a speed fallback.
5. Compare retrieval quality before permanently switching.
```

## If retrieval quality is weak

Do not immediately increase the generation model size.

Check:

```text
PDF parsing
Reading order
OCR quality
Chunk boundaries
Embedding consistency
Sparse retrieval
Hybrid fusion
Reranker
Retrieval thresholds
```

A larger answer model cannot recover evidence that was never retrieved.

---

# 15. Models Not Recommended for This Machine

Avoid using these as the default:

```text
qwen3.5:27b
qwen3.5:35b
qwen3.5:122b
qwen3-embedding:8b
Qwen3-Reranker-4B
Qwen3-Reranker-8B
```

Reasons:

- High RAM use
- Slow CPU inference
- Context memory pressure
- Limited 4 GB GPU VRAM
- Reduced responsiveness
- Little practical benefit before retrieval is properly evaluated

They may technically run with heavy system-memory use, but they are not appropriate for the first local production version.

---

# 16. Final Decision

Use:

```text
Primary answer model:
qwen3.5:9b

Development and fast fallback model:
qwen3.5:4b

Embedding model:
qwen3-embedding:4b

Reranker:
Qwen/Qwen3-Reranker-0.6B

Sparse retrieval:
BM25 or Qdrant sparse vectors
```

The default production pipeline should be:

```text
User question
    |
Query rewrite when required
    |
qwen3-embedding:4b query embedding
    |
Qdrant dense search
    |
BM25 or Qdrant sparse search
    |
Reciprocal Rank Fusion
    |
Qwen3-Reranker-0.6B
    |
Select 6 to 10 evidence chunks
    |
qwen3.5:9b
    |
Claim and citation validation
    |
Final answer with source pages
```

---

# 17. Sources Reviewed

- Ollama Qwen3.5 model library  
  https://registry.ollama.com/library/qwen3.5

- Ollama Qwen3 Embedding model library  
  https://registry.ollama.com/library/qwen3-embedding

- Qwen3 Embedding 4B model card  
  https://huggingface.co/Qwen/Qwen3-Embedding-4B

- Qwen3 Reranker 0.6B model card  
  https://huggingface.co/Qwen/Qwen3-Reranker-0.6B

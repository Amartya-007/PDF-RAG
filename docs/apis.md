# PDF-RAG API Reference

Everything a frontend needs to talk to the PDF-RAG backend. This document
describes what is **actually implemented and tested** in
`backend/app/api/main.py` as of this writing (backend test suite: 89/89
passing) — not an aspirational spec.

- **Base URL (dev):** `http://127.0.0.1:8000`
- **Start the server:** `uvicorn backend.app.api.main:app --reload` (see `docs/RUNNING.md`)
- **Interactive docs:** `http://127.0.0.1:8000/docs` (auto-generated Swagger UI — use this to try requests live)
- **Auth:** none. This is a local, single-user, offline app. Do not expose it directly to the internet.
- **Content type:** all request/response bodies are JSON except document upload, which is `multipart/form-data`.
- **CORS:** allowed origin is read from `RAG_FRONTEND_ORIGIN` (default `http://localhost:3000`) **once, at server startup**. Set it before starting the backend if your frontend runs elsewhere.

## Conventions

### Sessions

The backend organizes documents and conversations into **sessions**
(chat/workspace scoping). There is always a `"default"` session available
without creating one first. Most endpoints take an optional `session_id`;
when omitted, the server's current active session is used (`"default"`
until you switch it via `set_session`/create a new one).

### Errors

Errors come back as a standard FastAPI error body:

```json
{ "detail": "human-readable message" }
```

| Status | Meaning |
| --- | --- |
| `400` | Bad request (e.g. missing filename, unsupported file extension, OKF import failure) |
| `404` | Resource not found (e.g. unknown job ID) |
| `422` | Validation error (bad request body) or a known domain error (`RagError`) during ingestion |
| `500` | Unexpected server error during ingestion |

### IDs

- `document_id`, `session_id`, `node_id` are stable strings (hash-derived), safe to use as React keys / URL params.
- `job_id`, `query_id` are UUIDs, generated fresh per request/job.

---

## Health & status

### `GET /api/health`

Liveness check.

**Response `200`**
```json
{ "status": "ok" }
```

### `GET /api/status`

Snapshot of the current session's state — useful for a dashboard/sidebar.

**Response `200`**
```json
{
  "documents": 3,
  "chunks": 128,
  "concepts": 0,
  "ollama_ready": false,
  "ollama_message": "Extractive mode"
}
```

---

## Sessions

### `GET /api/sessions`

List all chat sessions.

**Response `200`** — `SessionOut[]`
```json
[
  { "session_id": "default", "title": "Welcome", "created_at": "2026-07-12T17:17:47" },
  { "session_id": "session_17123bec0c2a422d92454579784c1926", "title": "My Docs", "created_at": "2026-07-12T17:17:47" }
]
```

### `POST /api/sessions`

Create a new session. **This also switches the server's active session** to the newly created one (affects subsequent calls that omit `session_id`).

**Request body**
```json
{ "title": "My Docs" }
```

**Response `201`** — `SessionOut`

### `PATCH /api/sessions/{session_id}`

Rename a session.

**Request body**
```json
{ "title": "New title" }
```

**Response `200`**
```json
{ "session_id": "session_...", "title": "New title" }
```

### `DELETE /api/sessions/{session_id}`

Deletes the session **and every document/index entry in it** (irreversible). Falls back to the `"default"` session if you delete whichever session is currently active.

**Response `204`** — no body.

---

## Documents

### `GET /api/documents?session_id=...`

List documents. `session_id` is optional (defaults to the active session).

**Response `200`** — `DocumentOut[]`
```json
[
  {
    "document_id": "doc_4f1b37d1044ebecf",
    "filename": "policy.pdf",
    "status": "ready",
    "session_id": "session_17123bec0c2a422d92454579784c1926",
    "sha256": "604332b43d4e61c8f5e4fa4a39623acbb9eadf0a77267ae208490ee93192f40e"
  }
]
```

`status` is one of `"pending"`, `"processing"`, `"ready"`, `"failed"`.

### `POST /api/documents`

Upload and ingest a document. `multipart/form-data`, field name `file`. Optional query param `session_id`.

```
POST /api/documents?session_id=session_...
Content-Type: multipart/form-data

file: <binary>
```

Accepted extensions: `.pdf`, `.txt`, `.md` (configurable — see `GET /api/settings` → `allowed_file_extensions`).

**⚠️ Important — this call is synchronous.** The HTTP response only
arrives once ingestion has **fully finished** (parsing → cleaning →
structure detection → node building → FTS/heading indexing). For a small
document this is usually well under a second; for a large multi-hundred-page
PDF it can take longer. FastAPI runs the handler in a worker thread, so it
doesn't block *other* requests — but the upload request itself does hold
open until done. There is no "queued, come back later" response today.

If you want to show live progress, you can poll `GET /api/jobs/{job_id}`
**from a separate request** while the upload is in flight (see below) —
concurrent requests are served on different worker threads, so this
actually reflects live intermediate stages. You just can't get the
`job_id` until the upload response itself comes back, which somewhat
limits the usefulness of this for a progress bar on the *same* upload.
A future version may return `{job_id, document_id, status: "queued"}`
immediately and move ingestion to a true background task — the
`DocumentUploadResponse` schema for that already exists in
`backend/app/api/schemas/document.py` but isn't wired up yet.

**Response `200`** — `DocumentOut` (the finished document record)

**Errors:** `400` (bad/missing filename, disallowed extension), `422` (empty/unparseable document), `500` (unexpected ingestion failure).

### `DELETE /api/documents/{document_id}`

Delete a document and remove it from all indexes.

**Response `204`** — no body.

---

## Jobs

Every ingestion is tracked as a job with a stage history. Useful for an
"upload history" / activity log, or for concurrent polling per the note
above.

### `GET /api/jobs/{job_id}`

**Response `200`** — `JobOut`
```json
{
  "job_id": "910b7325-995b-4a7e-bd6d-0a070ab7310a",
  "document_id": "doc_4f1b37d1044ebecf",
  "status": "completed",
  "progress_message": "Indexed 4 nodes"
}
```

`status` is one of: `queued`, `parsing`, `cleaning`, `detecting_structure`, `building_nodes`, `indexing_fts`, `indexing_headings`, `completed`, `failed`, `cancelled`.

**Errors:** `404` if the job ID doesn't exist.

### `GET /api/documents/{document_id}/jobs`

All jobs ever run for a document (usually one, more if re-ingested).

**Response `200`** — `JobOut[]`

---

## Query

### `POST /api/query`

Ask a question. Retrieval is always lexical (SQLite FTS5 + BM25 + structural tree navigation) — never vector-based. Answering is either fast extractive pattern-matching or, if enabled, Ollama-generated synthesis with an extractive fallback on any Ollama failure.

**Request body** — `QueryRequest`
```json
{
  "question": "How many earned leave days can employees carry forward?",
  "session_id": "session_...",
  "include_debug": false
}
```
- `question` — required, non-empty.
- `session_id` — optional, defaults to `"default"`.
- `include_debug` — optional, default `false`. When `true`, populates `debug` with retrieval diagnostics (useful for a "why this answer" panel during development).

**Response `200`** — `AnswerResponse`
```json
{
  "answer": "Employees may carry forward up to 30 days of earned leave. [S1]",
  "answerable": true,
  "strategy": "EXTRACTIVE",
  "session_id": "session_...",
  "query_id": "e86fad65-d231-4693-ac47-872af76f381d",
  "citations": [
    {
      "document_id": "doc_4f1b37d1044ebecf",
      "document_name": "policy.pdf",
      "page_start": 1,
      "page_end": 1,
      "heading_path": [],
      "excerpt": "Employees may carry forward up to 30 days of earned leave.",
      "node_id": "node_ab54ceb86a4bd76049b76d29dab72de2"
    }
  ],
  "debug": null
}
```

- `strategy` — `"EXTRACTIVE"` (pattern/fuzzy-match answer, no LLM), `"GENERATE"` (Ollama-synthesized), or `"INSUFFICIENT"` (no answer found).
- `answerable: false` means the system found no adequate evidence — `answer` will contain a standard "could not find sufficient evidence" message. Render this distinctly from a real answer (e.g. don't attach citations UI to it).
- Answer text embeds citation markers like `[S1]`, `[S2]` inline — these correspond to `citations[0]`, `citations[1]`, etc. in order. Render the answer text and let the user click `[S1]` to jump to that citation.
- `citations[].heading_path` is the ancestor section titles from the document root down to (but not including) the cited node — e.g. `["Chapter 3: Leave Policy", "3.2 Carry-Forward Rules"]`. Currently populated when the node came from the newer structural pipeline; may be `[]` for some sources.
- `debug` is `null` unless you asked for it. Its shape is intentionally not stabilized/documented — treat it as opaque diagnostic data for a dev-only panel, not something to build production UI around.

### `POST /api/debug/retrieve`

Same request shape as `/api/query`, but returns the raw retrieved chunks and retrieval diagnostics instead of a generated answer. Useful for building a "sources considered" debug view.

**Request body** — `QueryRequest` (same as above)

**Response `200`**
```json
{ "chunks": [ /* raw Chunk objects */ ], "debug": { /* retrieval diagnostics */ } }
```

Treat both fields as opaque/debug-only — not a stable contract like `AnswerResponse`.

---

## Settings

### `GET /api/settings`

**Response `200`** — `SettingsOut`
```json
{
  "ollama_base_url": "http://localhost:11434/",
  "generation_model": "llama3.2",
  "use_ollama": false,
  "max_upload_size_mb": 100,
  "allowed_file_extensions": [".pdf", ".txt", ".md"],
  "backend_host": "127.0.0.1",
  "backend_port": 8000,
  "frontend_origin": "http://localhost:3000/",
  "debug_mode": false
}
```

### `PATCH /api/settings`

Updates a small, safe subset of settings **in the running process** (not persisted to disk/env — resets on restart).

**Request body** — all fields optional
```json
{ "generation_model": "llama3.2", "use_ollama": true, "debug_mode": true }
```

**Response `200`** — `SettingsOut` (full, updated settings)

Fields *not* patchable here (paths, ports, CORS origin, upload limits) require restarting the server with different `RAG_*` environment variables — see `docs/RUNNING.md`.

---

## OKF (Open Knowledge Format) — optional/advanced

OKF is a supplementary Markdown-based knowledge layer (concept files with cross-links) that can be generated from ingested documents or imported from an externally-authored bundle. Most frontends can ignore this section entirely.

### `POST /api/okf/validate`

Validates an OKF markdown bundle directory without importing it.

**Request body**
```json
{ "path": "/absolute/path/to/okf-bundle" }
```

**Response `200`**
```json
{ "issues": [ { "path": "...", "severity": "error", "message": "No Markdown concept files found." } ] }
```
Empty `issues` array means the bundle is valid.

### `POST /api/okf/import`

Imports a validated OKF bundle so its concepts participate in retrieval.

**Request body** — same as validate.

**Response `200`**
```json
{ "imported_concepts": 12, "concept_ids": ["concept-slug-1", "concept-slug-2"] }
```

**Errors:** `400` if the bundle fails to import (invalid structure, etc.)

---

## Data model reference

### `DocumentOut`
| Field | Type | Notes |
| --- | --- | --- |
| `document_id` | string | |
| `filename` | string | original upload filename |
| `status` | `"pending" \| "processing" \| "ready" \| "failed"` | |
| `session_id` | string | |
| `sha256` | string | content hash, used for de-duplication |

### `SessionOut`
| Field | Type |
| --- | --- |
| `session_id` | string |
| `title` | string |
| `created_at` | ISO-8601 datetime |

### `JobOut`
| Field | Type |
| --- | --- |
| `job_id` | string |
| `document_id` | string |
| `status` | see job status list above |
| `progress_message` | string, ≤500 chars |

### `AnswerResponse`
| Field | Type |
| --- | --- |
| `answer` | string |
| `answerable` | boolean |
| `strategy` | `"EXTRACTIVE" \| "GENERATE" \| "INSUFFICIENT"` |
| `session_id` | string |
| `query_id` | string (UUID) |
| `citations` | `CitationOut[]` |
| `debug` | object or `null` (opaque) |

### `CitationOut`
| Field | Type | Notes |
| --- | --- | --- |
| `document_id` | string | |
| `document_name` | string | source filename |
| `page_start` / `page_end` | int, ≥1 | 1-based, inclusive |
| `heading_path` | string[] | ancestor section titles; may be empty |
| `excerpt` | string | truncated source text |
| `node_id` | string | stable identifier of the cited passage |

### `SettingsOut`
| Field | Type |
| --- | --- |
| `ollama_base_url` | URL string |
| `generation_model` | string |
| `use_ollama` | boolean |
| `max_upload_size_mb` | int |
| `allowed_file_extensions` | string[] |
| `backend_host` | string |
| `backend_port` | int |
| `frontend_origin` | URL string |
| `debug_mode` | boolean |

---

## Known limitations (be aware of these when building the frontend)

- **Upload is synchronous** (see the note under `POST /api/documents`) — design your upload UI around a loading spinner for the duration of the request, not a progress bar driven by polling the same upload.
- **No authentication.** This is a local single-user app.
- **`PATCH /api/settings` doesn't persist** across a server restart.
- **`debug` fields are intentionally unstable** — fine for a dev-only inspector panel, not a contract to build core UI on.
- **Single active session per server process** — `session_id` defaults to whatever was last set via `POST/PATCH` session calls if omitted. Always pass `session_id` explicitly from the frontend once the user has more than one session, rather than relying on the server's notion of "current" session.

# Smart Journal

Increment 8 baseline for the Smart Journal knowledge base (with API hardening).

## What is implemented

- project skeleton with `src/` layout
- contract definitions for:
  - `BlobStore`, `MetaStore`, `VectorIndex`, `JobQueue`
  - `Extractor`, `EmbeddingProvider`, `LLMProvider`
- SQLite `MetaStore` schema v5 with base tables:
  - `nodes`, `revisions`, `content_items`, `tags`, `groups`, `edges` (+ join tables)
- ingestion extensions in MetaStore:
  - extraction status/text for `content_items`
  - `chunks` table with checksum per text chunk
  - `embeddings` table with vector BLOBs per `chunk_id` + `model_id`
  - `vector_index_ops` journal (`pending/applied`) for index consistency
- local filesystem `BlobStore` with SHA-256 content-addressed storage
- node CRUD + revisions + file attach/detach metadata
- in-process `JobQueue` with explicit statuses: `queued/running/completed/failed`
- extractor backend `basic_v1`:
  - plain text / markdown
  - PDF -> text
  - image -> thumbnail metadata + optional OCR text (best-effort)
  - audio -> metadata + optional ASR transcript (best-effort)
  - video -> metadata-only (video OCR/ASR tracked in `TODO.md`)
- ingestion pipeline with chunking (`chunk_size`, `chunk_overlap`) + SHA-256 checksum
- incremental embedding sync by chunk checksum (recompute only changed chunks)
- real embedding backend `multilingual_e5_small` (Sentence Transformers)
- file-backed `usearch_file` VectorIndex (`indexes/.../usearch.index` + `manifest.json`)
- replay of pending `vector_index_ops` at startup (`smart-journal run`)
- FastAPI web backend (`smart-journal serve`) for graphs/nodes/search/ingestion/vector query
- React web UI scaffold in `ui/` (Vite) connected to FastAPI API
- dual-mode web UI:
  - existing control/table panel for CRUD/search/ingestion workflows
  - new interactive 2D graph canvas (nodes/groups/tags + retrieval overlays)
- embedder warmup at API startup (default enabled) to avoid first-query latency
- FTS prefix search (`te` matches `test`)
- vector query payload enriched with chunk/node/content context
- graph/node detail API + group/tag APIs (without edge CRUD yet)
- graph/node detail and topology APIs now include edge payloads and aggregated edge counters
- node-details API now includes relationship direction (`incoming`/`outgoing`/`self`)
- Explore mode API (`/api/explore/run`) with implication provenance and optional synthesis node
- edge status APIs (`/api/edges/{edge_id}/accept|reject|patch`)
- startup auto-rebuild of non-durable vector index backends (e.g. `in_memory`) from embeddings
- FTS5 full-text search index in SQLite meta store
- Search API with scope filters: graph, group, tags
- CRUD for tags/groups and node-to-tag/node-to-group relations
- optional local real LLM provider backend: `ollama_chat` (kept optional; default remains `mock_chat`)
- optional cloud LLM provider backend: `openai_chat` (OpenAI SDK)
- config-driven provider selection through factories
- mock/in-memory providers kept for testing and fallback
- CLI that can list providers and start an app shell
- CI workflow + lint/type-check config

## Quick start

```powershell
python -m pip install -e .[dev]
smart-journal providers
smart-journal run
```

Run web API:

```powershell
python -m pip install -e .[ui]
smart-journal serve --host 127.0.0.1 --port 8000
```

Run React UI in development mode:

```powershell
cd ui
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

Build UI and let FastAPI serve static files from `ui/dist`:

```powershell
cd ui
npm run build
cd ..
smart-journal serve
```

Optional env vars for embedder warmup:

```powershell
$env:SMART_JOURNAL_PRELOAD_EMBEDDER = "1"          # default: enabled
$env:SMART_JOURNAL_PRELOAD_EMBEDDER_STRICT = "0"   # if 1: startup fails on warmup error
```

Without installation:

```powershell
$env:PYTHONPATH="src"
python -m smart_journal providers --json
python -m smart_journal run --json
```

## Config

Default config file: `smart-journal.toml`.

```toml
[blob_store]
backend = "local_cas"
root = "./data/blobs"

[meta_store]
backend = "sqlite"
path = "./data/meta.db"

[vector_index]
backend = "in_memory"

[job_queue]
backend = "in_process"

[extractor]
backend = "basic_v1"

# Optional multimodal extraction controls
enable_image_ocr = true
enable_audio_asr = true
ocr_lang = "eng"
asr_model = "base"
# asr_language = "en"
# asr_device = "cpu"

[embedding_provider]
backend = "multilingual_e5_small"
device = "cpu"
text_prefix = "passage"
batch_size = 32

[llm_provider]
backend = "mock_chat"
```

Optional local Ollama backend:

```toml
[llm_provider]
backend = "ollama_chat"
base_url = "http://127.0.0.1:11434"
model = "llama3.1:8b-instruct"
timeout_seconds = 60
```

Optional OpenAI backend:

```toml
[llm_provider]
backend = "openai_chat"
model = "gpt-4.1-mini"
api_key = "sk-..."  # or use OPENAI_API_KEY env var
timeout_seconds = 60
```

## OpenAI Smoke Check

Prerequisites:

```powershell
python -m pip install -e .[dev,ui]
```

Set API key (or pass it as `--api-key`):

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

Run one-command smoke (chat + structured + Explore):

```powershell
$env:PYTHONPATH="src"
python scripts/smoke_openai.py --model gpt-4.1-mini
```

Expected result: JSON with `"ok": true` and non-empty `chat_preview` / `structured_payload`.

Optional:

```powershell
python scripts/smoke_openai.py --model gpt-4.1-mini --base-url https://api.openai.com/v1
python scripts/smoke_openai.py --model gpt-4.1-mini --query "What risks connect these notes?"
```

## Backlog

Pending multimodal work (video + medium/heavy levels) is tracked in `TODO.md`.

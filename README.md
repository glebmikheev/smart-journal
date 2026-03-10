# Smart Journal

Increment 5 baseline for the Smart Journal knowledge base.

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
  - image -> thumbnail metadata (no OCR)
  - audio/video -> metadata-only (alpha)
- ingestion pipeline with chunking (`chunk_size`, `chunk_overlap`) + SHA-256 checksum
- incremental embedding sync by chunk checksum (recompute only changed chunks)
- real embedding backend `multilingual_e5_small` (Sentence Transformers)
- file-backed `usearch_file` VectorIndex (`indexes/.../usearch.index` + `manifest.json`)
- replay of pending `vector_index_ops` at startup (`smart-journal run`)
- FastAPI web backend (`smart-journal serve`) for graphs/nodes/search/ingestion/vector query
- React web UI scaffold in `ui/` (Vite) connected to FastAPI API
- FTS5 full-text search index in SQLite meta store
- Search API with scope filters: graph, group, tags
- CRUD for tags/groups and node-to-tag/node-to-group relations
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

[embedding_provider]
backend = "multilingual_e5_small"
device = "cpu"
text_prefix = "passage"
batch_size = 32

[llm_provider]
backend = "mock_chat"
```

# Smart Journal

Increment 2 baseline for the Smart Journal knowledge base.

## What is implemented

- project skeleton with `src/` layout
- contract definitions for:
  - `BlobStore`, `MetaStore`, `VectorIndex`, `JobQueue`
  - `Extractor`, `EmbeddingProvider`, `LLMProvider`
- SQLite `MetaStore` schema v1 with base tables:
  - `nodes`, `revisions`, `content_items`, `tags`, `groups`, `edges` (+ join tables)
- ingestion extensions in MetaStore:
  - extraction status/text for `content_items`
  - `chunks` table with checksum per text chunk
- local filesystem `BlobStore` with SHA-256 content-addressed storage
- node CRUD + revisions + file attach/detach metadata
- in-process `JobQueue` with explicit statuses: `queued/running/completed/failed`
- extractor backend `basic_v1`:
  - plain text / markdown
  - PDF -> text
  - image -> thumbnail metadata (no OCR)
  - audio/video -> metadata-only (alpha)
- ingestion pipeline with chunking (`chunk_size`, `chunk_overlap`) + SHA-256 checksum
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
backend = "mock_text"

[llm_provider]
backend = "mock_chat"
```

# Smart Journal

Increment 0 scaffold for the Smart Journal knowledge base.

## What is implemented

- project skeleton with `src/` layout
- contract definitions for:
  - `BlobStore`, `MetaStore`, `VectorIndex`, `JobQueue`
  - `Extractor`, `EmbeddingProvider`, `LLMProvider`
- config-driven provider selection through factories
- mock/in-memory providers with `capabilities()` and `version()`
- CLI that can list providers and start an empty app shell
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
backend = "in_memory"

[meta_store]
backend = "in_memory"

[vector_index]
backend = "in_memory"

[job_queue]
backend = "in_process"

[extractor]
backend = "plain_text"

[embedding_provider]
backend = "mock_text"

[llm_provider]
backend = "mock_chat"
```

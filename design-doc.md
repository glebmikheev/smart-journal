Below is the **alpha version of the Smart Journal design document (smart-journal)**: a graph knowledge base with multimodal content, semantic links, Explore mode, and support for local LLMs/embeddings (with an option for cloud providers via contracts).

---

## 1) Summary

**Smart Journal** is a local-first knowledge base where:

* A **Node** (Record/Node) contains **multimodal content** (a set of attachments: text, PDF, images, audio, video, binary files, etc.) + metadata.
* Nodes live inside a **graph/catalog** (Graph/Catalog). Inside a graph, **groups** are possible (logical grouping for the UI).
* There are **links (edges)** of three types between nodes:

  1. **Association**: a manual link.
  2. **Semantic Relevance**: a semantic-similarity link (RAG-like), created automatically/manually from candidates.
  3. **Implication**: links/inferences built in Explore mode using an LLM/reasoning pipeline.

The system can create **aggregating nodes (Synthesis Nodes)** based on links and/or content fragments from source nodes.

---

## 2) Goals and Non-goals

### Goals

* A unified store of multimodal "records" with a graph of links.
* Fast search and navigation: by text, tags, links, and semantics.
* Automatic semantic-link suggestions with **user control** (approve/reject).
* Explore mode: an LLM builds inferences (Implication) with a **transparent evidence base** (provenance).
* Incremental updates on edits: recompute **only affected** parts/indexes.
* Extensibility: new content types, new embedding providers, and new LLM providers.

### Non-goals (for alpha)

* Multi-user access rights / enterprise-level ACL.
* Error-free "truth" from an LLM (we introduce trust/status mechanisms, not guarantees).
* "Smart" cross-device sync like Notion/Obsidian Sync (we can lay groundwork).

---

## 3) Terms and Entities

* **Graph (Catalog)**: a container for nodes and links, a separate "database".
* **Group**: logical grouping of nodes for visualization (board/cluster/folder in UI), does not directly affect semantics.
* **Node**: a record containing a set of **Content Items**.
* **Content Item**: an attachment (file/text/media) with type (MIME), hash, size, and storage path.
* **Chunk**: a content fragment suitable for embedding (text chunk, keyframe, audio segment, etc.).
* **Edge (Link)**: a connection between nodes. Has a type, weight/confidence, and provenance.
* **Provenance**: why the link exists: which chunks/frames/transcript/prompt/model were used.
* **Synthesis Node**: a node created from a set of edges/nodes/fragments as a "summary / synthesis / inference node".

---

## 4) Core User Scenarios

1. **Add a node** with multiple files (pdf+md+mp3+png).
2. The system extracts representations (text/frames/transcript), builds embeddings, suggests **N** semantically close nodes -> the user approves some.
3. The user manually adds an **Association**.
4. The user edits node content -> the system recomputes only changed chunks and updates relevant Semantic edges.
5. The user runs **Explore**: asks a question -> the system finds relevant nodes/chunks, the LLM builds inferences -> **Implication edges** are created + (optionally) a **Synthesis Node**.
6. Later, the user opens a node and sees that some links are **stale** due to changes -> can run targeted "Recompute".

---

## 5) Architecture (Logical Modules)

### 5.1. Core Engine

* Management of graphs, nodes, content, and versions.
* API for UI (local IPC/HTTP/gRPC).

### 5.2. Storage Layer

* **Metadata/graph**: tables for nodes, edges, tags, groups, revisions, provenance.
* **Content**: file storage for attachments (preferably content-addressed).
* **Indexes**:

  * full-text index (FTS),
  * vector index (ANN),
  * tag/attribute index.

### 5.3. Ingestion + Extractors (plugins)

* For each MIME type: extraction of "derived artifacts":

  * text (PDF->text, DOCX->text, MD->text),
  * images (thumbnails, OCR optional),
  * audio (ASR transcript),
  * video (keyframes + audio transcript + metadata),
  * binaries (minimal metadata: size/hash/name; no "understanding" at alpha).

### 5.4. Embedding Providers (contract)

* Text embeddings.
* Multimodal embeddings (shared space for text/image/audio/video) - see options below.
* Ability to plug in local and cloud implementations.

### 5.5. Linker

* Generate Semantic Relevance candidates.
* Support statuses: suggested/pending/accepted/rejected/stale.
* Incremental updates.

### 5.6. Explore Engine

* Retrieval (search relevant chunks/nodes).
* Reasoning (LLM) -> Implication edges (+ provenance).
* Synthesis Node construction.

---

## 6) Data Model (Proposal)

### 6.1. Node

Minimal fields:

* `node_id` (UUID)
* `graph_id`
* `title`
* `created_at`, `updated_at`
* `group_id?`
* `tags[]` (see section 10)
* `attributes` (JSON: arbitrary system/user fields)
* `current_revision_id`

### 6.2. Node Revision (Versioning)

* `revision_id`
* `node_id`
* `parent_revision_id?`
* `created_at`
* `change_summary` (text + optional LLM summary)
* `content_manifest` (list of content_item_id + order + annotations)

### 6.3. Content Item

* `content_item_id`
* `node_id`
* `mime_type`
* `original_filename`
* `size_bytes`
* `sha256`
* `storage_ref` (path/CAS key)
* `created_at`
* `extract_status` (pending/done/error)
* `derived_refs` (links to transcript, keyframes, text from PDF, etc.)

### 6.4. Chunk

* `chunk_id`
* `content_item_id`
* `node_id`
* `modality` (text/image/audio/video)
* `chunk_checksum` (for incrementality)
* `span` (offset/timecode/frame range)
* `text?` / `frame_ref?` / `audio_ref?`
* `embedding_ref` (or store embedding inline)
* `embedding_model_id`

### 6.5. Edge

* `edge_id`
* `graph_id`
* `from_node_id`, `to_node_id`
* `type` = association | semantic | implication
* `subtype` (for semantic: auto/aux; for implication: explore_session_id)
* `directional` (bool)
* `weight` (float, 0..1)
* `status` = pending/accepted/rejected/stale
* `created_at`, `updated_at`
* `created_by` (user/system/llm)
* `provenance` (JSON: chunk_ids, prompt hashes, model ids, etc.)

---

## 7) Semantic Relevance: Multimodal Similarity (Options)

The key problem: **how to compare different content types** (text vs video vs audio vs images).

### Option A (recommended for alpha): one shared multimodal embedding space

Use a model/approach that provides embeddings in one space for multiple modalities. For example, **ImageBind** claims joint embedding for multiple modalities (including text/audio/video) in one space. ([GitHub][1])
Pros:

* Immediate cross-modal retrieval: text can find video/audio/images.
* Semantic-link computation is simpler (single ANN index).
  Cons:
* More resource-intensive (especially video).
* Quality for individual modalities may be worse than specialized models.

**Combination for video/audio:**

* Video -> (1) keyframes -> image embeddings, (2) audio track -> audio embedding, (3) ASR -> text embedding.
* Then aggregate into a "node embedding profile".

### Option B: multi-index (by modality) + score fusion

* Separate embeddings and ANN indexes: text-text, image-image, audio-audio, video-video.
* Cross-modal is handled through "bridges": for example image<->text via OpenCLIP-like models (`open_clip` is an OSS CLIP implementation). ([GitHub][2])
* Final score: weighted sum / RRF / top-k merge.
  Pros:
* Allows selecting the best model per modality.
* Flexible weight tuning.
  Cons:
* More complex implementation and debugging.
* Cross-modal becomes "partially available" and depends on bridges.

### Option C: "everything to text" (cheap, but limited)

* Images -> OCR/Captioning,
* Audio/video -> ASR,
* Then only text embeddings.
  Pros:
* Fast and simple.
  Cons:
* Loses visual/audio semantics, captioning/OCR can fail.

**Recommendation:** for alpha, use **A or a hybrid A+B**, where A provides baseline cross-modal, and B can be added later for quality.

---

## 8) Semantic Link Generation (Auto and Auxiliary)

### 8.1. Chunk-level -> node-level

1. For each new/changed chunk `c` of node `N`:

   * run ANN query `topK` against vector index,
   * get candidates (chunks from other nodes).
2. Reduce candidates to a set of **candidate nodes** `M`.
3. Compute `node_similarity(N, M)`:

   * **Max pooling**: maximum over chunk pairs,
   * or **Mean of top-t** (more stable),
   * plus modality-diversity penalties/bonuses (if both text and image match -> +).

### 8.2. Auto Semantic Relevance (on add/update)

* User parameter: `N_suggestions` (for example 5..30).
* System creates edges with `pending` status and shows UI:

  * "accept all / choose several / reject".
* Accepted -> `accepted`, rejected -> `rejected` (and preferably not suggested again without significant changes).

### 8.3. Auxiliary Semantic Relevance (on request)

* "Find related" command:

  * shows candidates, including previously rejected ones (optionally via filter).
* User selects -> `accepted`.

---

## 9) Implication in Explore Mode

### 9.1. Explore Contract

Input:

* `query` (text)
* `scope` (entire graph / group / subgraph / selected nodes)
* parameters (max nodes/chunks, response style, evidence strictness)

Pipeline:

1. Retrieval: find relevant chunks/nodes (semantic + full-text).
2. Reasoning: LLM builds:

   * list of inferences (implications),
   * for each inference: links to supporting evidence (chunk_ids/node_ids),
   * confidence/hallucination risk (heuristics).
3. Persist results:

   * Implication edges: `from` = sources, `to` = node/idea/or existing node
   * or create a new Synthesis Node, and edges link sources -> synthesis.

### 9.2. Provenance and Factual-Grounding Controls

Required fields:

* model_id, prompt_hash, retrieval_snapshot (which chunks were provided),
* quotes (short) or pointers to spans/timecodes,
* default status: `accepted?` **no** -> better `pending` (user "approves" the inference).

### 9.3. Improvement over "just mark stale"

**Proposal:** store implication dependencies on chunks:

* edge.provenance includes `chunk_checksums` (or revision_id).
* When a node changes:

  * if chunk(s) used as evidence were affected -> mark implication edge as `stale`.
  * optionally: button "Re-run Explore for this implication" (targeted recompute).

---

## 10) Tags and Metadata: Options

### Question: "are tags content or a system attribute?"

**Option A (recommended): tags as system attributes**

* Tags live in a separate table + m2m.
* Indexed (fast filtering/facets).
* Can participate in ranking (boost), but are not "part of content".
  Pros: clean model, fast queries.
  Cons: if "tag semantics" are required, tags must be embedded separately.

**Option B: tags as content (added to embedding text)**

* Add a line like `Tags: ...` during text embedding.
  Pros: tags influence semantic search out of the box.
  Cons: tags can "distort" content meaning, harder to control weight.

**Option C: hybrid**

* Tags are system attributes,
* but there is an option: "include tags into embeddings (weight=...)".
  Pros: flexibility.
  Cons: slightly more complex pipeline.

**Recommendation:** C (or A for alpha, with a quick path to C).

### Other node metadata (proposal)

* `source` (user/import/webclip)
* `language` (detected)
* `importance` / `pin`
* `created_by_model` (for synthesis/implication)
* `visibility` (future: private/shared)
* `quality_flags` (ocr_low_confidence, asr_low_confidence, ...)

---

## 11) Editing and Incremental Recomputation

### 11.1. General principle

* **Content items are immutable** (by hash). Editing creates a new node revision and/or new content items.
* Extracted artifacts and chunks are cached by `(content_hash, extractor_version)`.

### 11.2. Association

* No recompute.
* On node change: mark edge as `possibly_stale` (or `stale_reason=content_changed`), but do not break the link.

### 11.3. Semantic Relevance

* Determine by chunk checksums:

  * which chunks were removed/added/changed,
* recompute embeddings only for changed chunks,
* update ANN index incrementally,
* recompute candidate links only for affected areas:

  * new chunks -> find new neighbors,
  * removed chunks -> decrease weights of existing links (or mark for revalidation).

**Subtle point:** "which links may no longer be relevant?"

* Store chunk contributions to edge.weight (for example top contributors).
* If key contributors disappear -> edge becomes `stale` or weight drops below threshold.

### 11.4. Implication

Alpha baseline: mark as stale.
Improvement (see 9.3): if evidence chunks changed -> stale; otherwise "still valid under current data".

---

## 12) Synthesis Nodes (Aggregation)

### Two modes

1. **Edge-centric synthesis**: summary of links (Association/Semantic/Implication) among a set of nodes.
2. **Content-centric synthesis**: summary of content across multiple nodes + source links.

Model:

* A Synthesis Node stores:

  * list of sources (node_ids),
  * "recipe" (build parameters),
  * provenance (model, date, retrieval snapshot),
  * optional "frozen copy" of quotes/fragments.

---

## 13) Contracts for LLM and Embeddings (to easily swap providers)

### 13.1. EmbeddingProvider API

* `capabilities()` -> {text, image, audio, video}
* `embed_text(chunks[]) -> vectors[]`
* `embed_image(frames[]) -> vectors[]`
* `embed_audio(segments[]) -> vectors[]`
* `embed_video(keyframes/segments[]) -> vectors[]` (optional)
* `model_id`, `dim`, `normalize`, `batch_limits`

### 13.2. LLMProvider API

* `generate_structured(prompt, schema) -> json`
* `chat(messages, tools?)`
* `model_id`, `context_window`, `supports_vision?`

**Local priority:** local LLM execution can be built around llama.cpp (GGUF) as one of the adapters. ([GitHub][3])
UI/engine should not depend on a specific runtime.

---

## 14) Storage and Indexes: Technology Options (cross-platform)

Below are 3 realistic options. Start with one and keep abstraction layers for migration.

### Option 1 (recommended): Embedded-first

* **Metadata + Graph:** SQLite (tables for nodes/edges/revisions/tags + FTS5).
* **Vector search:** SQLite extension or built-in ANN:

  * minimal option: `sqlite-vec` (as "runs anywhere SQLite runs"). ([GitHub][4])
  * or a separate ANN (HNSW) inside core (Rust).
* **Content store:** file-based CAS folder (sha256/xxhash).
  Pros:
* Maximum portability (including potential Android support).
* Simple installation (single DB file + content folder).
  Cons:
* ANN updates/performance must be implemented carefully.

### Option 2: Sidecar Vector DB (desktop-friendly)

* SQLite for metadata/graph
* **Qdrant** as local sidecar for vectors (Rust, Apache-2.0). ([GitHub][5])
  Pros:
* Fast path to high-quality ANN + payload filters.
  Cons:
* Separate process/service; harder on Android.

### Option 3: LanceDB as embedded vector store

* SQLite for graph/meta (or everything in LanceDB/tables)
* **LanceDB** as embedded vector DB, Apache-2.0 (as stated in integration docs). ([GitHub][6])
  Pros:
* Embedded approach optimized for vectors/multimodality.
  Cons:
* Must carefully design interoperability with graph/revisions.

---

## 15) UI / Cross-platform (proposal)

Target platforms: **Linux/Windows + potential Android**.

* **Tauri 2.x** as the UI shell (Rust backend + web frontend) - claims desktop and mobile builds. ([GitHub][7])
  Alternatives: Flutter (faster mobile-first), Electron (heavier).

---

## 16) Requirements (templates + initial set)

### 16.1. Functional Requirement (FR) Template

* **ID:** FR-XXX
* **Name:** ...
* **Description:** ...
* **Priority:** P0/P1/P2
* **Actors:** ...
* **Preconditions:** ...
* **Main flow:** ...
* **Exceptions/errors:** ...
* **Acceptance Criteria:** (Given/When/Then)

### 16.2. Non-Functional Requirement (NFR) Template

* **ID:** NFR-XXX
* **Category:** performance/security/usability/portability/...
* **Requirement:** ... (measurable)
* **Metric/Threshold:** ...
* **Validation method:** test/profiling/audit
* **Priority:** P0/P1/P2

---

### 16.3. Functional Requirements (initial set)

**FR-001 Graph Management (P0)**
Create/open/delete a graph (catalog), import/export.

**FR-002 Node CRUD (P0)**
Create a node with multiple attachments, edit, delete, restore from history.

**FR-003 Content Attachment (P0)**
Support adding files of any type; for unknown types, store as opaque binary + metadata.

**FR-004 Extractors Pipeline (P0)**
After content is added, run extraction of derived artifacts (text/frames/transcript) via plugins.

**FR-005 Full-Text Search (P0)**
Search over node text, transcripts, extracted text from PDF/DOCX.

**FR-006 Tags (P0)**
Add/remove tags; filtering and faceted search.

**FR-007 Groups for Visualization (P1)**
Create groups, move nodes, display clusters.

**FR-008 Manual Association Links (P0)**
User creates/deletes Association links, adds description/reason.

**FR-009 Auto Semantic Suggestions (P0)**
When adding/updating a node, suggest N semantically close nodes (pending -> accepted/rejected).

**FR-010 Auxiliary Semantic Linking (P1)**
"Find similar" command and manual approval of candidates.

**FR-011 Explore Mode (P0)**
Build Implication inferences on user request and save them as edges (+ provenance).

**FR-012 Synthesis Node Creation (P1)**
Create an aggregating node from a set of edges/nodes with source citations.

**FR-013 Revision History (P0)**
Node change history: revision list, attachment-manifest diff, rollback.

**FR-014 Staleness Tracking (P0)**
Mark edges as stale/possibly_stale when sources change (by edge-type rules).

**FR-015 Provider Plugins (P0)**
Connect LLMProvider and EmbeddingProvider via a unified contract.

---

### 16.4. Non-Functional Requirements (initial set)

**NFR-001 Portability (P0)**
Desktop: Windows + Linux. Architecture must not exclude Android.

**NFR-002 Local-first & Offline (P0)**
All basic operations (CRUD, search, graph, semantics when local models are available) work offline.

**NFR-003 Performance: Ingestion (P1)**
Adding a node must not block UI: extraction/indexing runs as jobs with progress.

**NFR-004 Performance: Search Latency (P0)**
FTS query: < 200ms on medium graphs (roughly up to 50k nodes) on desktop; ANN: < 500ms (target thresholds can be refined).

**NFR-005 Reliability & Crash Safety (P0)**
Metadata transactional integrity, resilience to power/process interruption.

**NFR-006 Explainability (P0)**
Every Implication must include provenance (sources/chunks/time/model).

**NFR-007 Privacy (P0)**
Content does not leave the device by default. Cloud providers are used only when explicitly enabled.

**NFR-008 Security at Rest (P1)**
Optional encryption for DB and content.

**NFR-009 Extensibility (P0)**
Add new extractor/modality without core migrations (via plugin API and versioning).

**NFR-010 Model Output Risk (P1)**
Transcripts/LLM outputs are marked as "derived"; the system must not replace source content with them. (ASR can produce errors/insertions - this is a known risk for such models.) ([GitHub][8])

---

## 17) Open Questions and Branching Decisions

1. **Single index vs multi-index by modality**

* For alpha, a **single (ImageBind-like)** index is better for cross-modal, then improve quality with multi-indexes.

2. **Where to store vectors**

* For a fast desktop start -> Qdrant sidecar. ([GitHub][5])
* For "everything in one file" and a path to Android -> SQLite + sqlite-vec/built-in HNSW. ([GitHub][4])
* For embedded vectors with AI-data focus -> LanceDB. ([docs.langchain.com][9])

3. **Degree of Explore "autonomy"**

* For alpha: save everything as `pending` (user confirms).
* Later: trust rules (for example, "if based on >=3 independent sources + low answer entropy").

4. **Semantic links: store only node-node or also chunk-chunk?**

* Recommend storing node-node as primary, but include "top contributing chunk pairs" in provenance - this gives explainability and incremental recompute.

---

## 18) Proposed "default" startup stack (most pragmatic)

**Core:** Rust (portability, performance, single binary)
**UI:** Tauri 2 + any web frontend (React/Svelte/Vue) ([GitHub][7])
**DB:** SQLite (meta/graph/revisions/tags + FTS5)
**Vectors:** SQLite-vec or built-in HNSW (with option to switch via adapter to Qdrant/LanceDB) ([GitHub][4])
**Local LLM adapter:** llama.cpp (GGUF) as one provider ([GitHub][3])
**Multimodal embeddings:** ImageBind adapter as cross-modal baseline ([GitHub][1])
**ASR:** Whisper adapter ([GitHub][8])

---

[1]: https://github.com/facebookresearch/ImageBind?utm_source=chatgpt.com "ImageBind One Embedding Space to Bind Them All"
[2]: https://github.com/mlfoundations/open_clip?utm_source=chatgpt.com "mlfoundations/open_clip: An open source implementation ..."
[3]: https://github.com/ggml-org/llama.cpp?utm_source=chatgpt.com "ggml-org/llama.cpp: LLM inference in C/C++"
[4]: https://github.com/asg017/sqlite-vec?utm_source=chatgpt.com "asg017/sqlite-vec: A vector search ..."
[5]: https://github.com/qdrant/qdrant?utm_source=chatgpt.com "GitHub - qdrant/qdrant: Qdrant - High-performance, ..."
[6]: https://github.com/lancedb/lancedb?utm_source=chatgpt.com "lancedb/lancedb: Developer-friendly OSS embedded ..."
[7]: https://github.com/tauri-apps/tauri?utm_source=chatgpt.com "tauri-apps/tauri: Build smaller, faster, and more secure ..."
[8]: https://github.com/openai/whisper?utm_source=chatgpt.com "openai/whisper: Robust Speech Recognition"
[9]: https://docs.langchain.com/oss/javascript/integrations/vectorstores/lancedb?utm_source=chatgpt.com "LanceDB integration - Docs by LangChain"


---

## Updates to the Design Document (v0.1-alpha, additional sections)

### A. Architectural principles for seamless modularity

**Requirement:** any module can be replaced **without changing the rest of the system** as long as it satisfies the contract.

**How we ensure this:**

1. **Contracts + capability negotiation**

   * each provider/module declares `capabilities()` and `version()`
   * core makes no assumptions about concrete implementation (no "if SQLite"/"if S3" in domain logic)

2. **No leaky abstractions**

   * private details do not leak through contracts (for example `rowid`, SQL queries, S3 ETag as mandatory field, etc.)

3. **Stable identifiers**

   * in-domain only UUID/ULID; any internal integer ids (for ANN) are purely technical and hidden behind contracts

4. **Config-driven provider selection**

   * `BlobStoreFactory`, `MetaStoreFactory`, `VectorIndexFactory` choose implementation by config, not code paths

---

## 1) Tiered storage: BlobStore + Cache + Sync (added/clarified)

### 1.1. BlobStore contract

**Goal:** replace `local CAS folder` with `S3/GCS/etc` without business-logic changes.

**Types**

* `BlobKey = sha256:<hex>` (content-addressed key)
* `BlobRef = { scheme, key, size, hash, version? }`

  * scheme: `localcas | s3 | gcs | ...`

**Methods (logical)**

* `put(stream, opts) -> BlobRef`
* `open(blobRef, range?) -> stream`
* `stat(blobRef) -> {size, hash, modified, ...}`
* `exists(blobRef) -> bool`
* `delete(blobRef)` (optional, can be soft-delete)
* `verify(blobRef) -> bool` (rehash when needed)

### 1.2. Tiered implementation (recommended)

* `TieredBlobStore(remote: BlobStore, cache: BlobStore, policy: CachePolicy, sync: SyncQueue)`

  * write path: `cache.put` -> enqueue upload -> (optionally) confirm remote
  * read path: try cache -> if miss: remote.open + cache.put (read-through)
  * `CachePolicy`: max size, LRU, pinning for "important" graphs, prefetch

### 1.3. Sync states (add to ContentItem metadata)

* `sync_state`: `local_only | queued_upload | synced | queued_download | error`
* `last_sync_error`
* `remote_ref?` (if remote is enabled)

> For alpha, enable only `local_only` and queue infrastructure; add full remote sync later as an increment.

---

## 2) Abstraction over SQLite: Domain API + MetaStore (clarification)

### 2.1. Domain layer (SQL-agnostic)

* `NodeService`
* `LinkService`
* `RevisionService`
* `IngestionService`
* `ExploreService`

They work through interfaces:

* `MetaStore` (metadata/graph/revisions/attribute search)
* `BlobStore` (content)
* `VectorIndex` (ANN)
* `JobQueue` (async jobs inside the application)

### 2.2. MetaStore contract (right-sized)

**Important:** MetaStore should not be a "universal ORM." It should cover *only* required operations.

Example operation groups:

* Nodes: create/read/update/delete, list by graph/group/tag, get revisions
* Revisions: commit revision, diff manifest, rollback
* ContentItems: attach/detach, update extraction status, derived refs
* Chunks: upsert by checksum, mark deleted, list by node/revision
* Edges: upsert semantic candidates, accept/reject, mark stale, query neighborhood
* Search:

  * `search_fulltext(query, scope, limit)`
  * `filter_nodes(tags, attrs, time_range, group)`

**Backend v1:** SQLite (FTS5 for full text, regular indexes for metadata).
**Backend v2:** Postgres/other - added without changing domain services.

### 2.3. Transactionality guarantees

* MetaStore must support transactions for "atomic" operations:

  * commit revision + update manifest + persist job statuses
* For MetaStore <-> VectorIndex coupling, use an **operations log** (see below) to survive crashes.

---

## 3) ANN as a separate file + USearch (added/clarified)

### 3.1. VectorIndex contract

**Goal:** replace USearch with FAISS/HNSWlib/Qdrant/LanceDB without changing domain logic.

Minimal contract:

* `upsert(vectors: [{external_id, vector, metadata?}])`
* `delete(external_ids[])` *(can be best-effort)*
* `query(vector, top_k, filter?) -> [{external_id, score}]`
* `save()/load()`
* `rebuild()` (optional)
* `capabilities(): {supports_delete, supports_filter, metric_types, ...}`

**external_id** is `chunk_id` (UUID) or a stable chunk key.

### 3.2. Physical storage

* Vectors *can* be stored:

  * either in MetaStore (`embeddings` table as BLOB) + index stores only structure,
  * or inside the index file (if backend does that),
  * or hybrid.

**Recommendation for "seamless" replacement and recovery:**

* store the "source of truth" in MetaStore:

  * `embeddings(chunk_id, model_id, dim, vector_blob, checksum, created_at)`
* USearch index stores ANN structure for speed; on corruption it can be **rebuilt** from the embeddings table.

### 3.3. MetaStore <-> VectorIndex consistency (important)

Problem: 2 stores -> risk of desynchronization.

Solution (simple and robust):

* maintain `vector_index_ops` in MetaStore (journal):

  * `op_id, op_type(upsert/delete), chunk_id, model_id, status(pending/applied), created_at`
* domain logic:

  1. writes embeddings + ops(pending) in a MetaStore transaction
  2. worker applies ops to VectorIndex
  3. marks ops as `applied`
* on app startup: "replay pending ops"

This provides seamlessness and crash resilience, and makes USearch replacement easier.

---

## 4) Clarification on FTS5 and where it is used

* **FTS5** is used only inside SQLite MetaStore backend as the `search_fulltext` implementation.
* When migrating MetaStore to Postgres, this can be replaced with `tsvector`/GIN without changing `NodeService/SearchService`.

---

## 5) Data-model updates (minimal changes)

### ContentItem (add)

* `blob_ref` (string/JSON: `scheme+key+version`)
* `sync_state`, `remote_blob_ref?`, `cache_state?`
* `extraction_status`, `derived_refs`

### Embeddings table (if we choose "source of truth in MetaStore")

* `chunk_id`, `model_id`, `vector_blob`, `dim`, `metric`, `checksum`, `created_at`

### Vector index artifacts

* `indexes/<graph_id>/<model_id>/usearch.index`
* `indexes/<graph_id>/<model_id>/manifest.json` (version, metric, dim, build params)

---

# Work plan: incremental development model (alpha)

Below are **increments** (each ends with a working state and demo scenarios). No time binding.

---

## Increment 0 - Project skeleton and contracts

**Goal:** lock module boundaries.

Deliverables:

* repository, build, CI, linters
* contract definitions:

  * `BlobStore`, `MetaStore`, `VectorIndex`, `JobQueue`
  * `Extractor`, `EmbeddingProvider`, `LLMProvider`
* config system + factories (provider selection)

Acceptance:

* launch an "empty app/CLI" and verify which providers are available via `capabilities()`.

---

## Increment 1 - MetaStore(SQLite) + basic Nodes/Revisions + Local CAS BlobStore

**Goal:** minimal offline knowledge base.

Deliverables:

* SQLite schema v1 (nodes, revisions, content_items, tags, groups, edges skeleton)
* Local CAS BlobStore (`sha256` + blobs folder)
* Node CRUD + attach files (without extraction)

Acceptance:

* create a graph, add a node with multiple files, open/read blob back, delete node (soft-delete is OK).

---

## Increment 2 - Ingestion pipeline + Extractors (minimum) + Chunking

**Goal:** chunks and derived artifacts appear.

Deliverables:

* JobQueue (in-process task queue + statuses)
* Extractors v1:

  * plain text / markdown
  * PDF -> text (via library if needed)
  * image -> thumbnail (without OCR)
  * audio/video: metadata only for now (alpha minimum)
* Chunking for text (rule + checksum per chunk)

Acceptance:

* add PDF/MD -> extracted text and chunks appear.

---

## Increment 3 - Full-text search (FTS5) + tags/groups

**Goal:** basic search and organization.

Deliverables:

* FTS5 index and extracted-text -> FTS sync
* Search API: scope (graph/group/tags)
* CRUD for tags and groups

Acceptance:

* search words inside PDF/MD, filter by tags and groups.

---

## Increment 4 - EmbeddingProvider v1 + Embeddings store

**Goal:** vectorization-ready without ANN.

Deliverables:

* EmbeddingProvider contract + mock/local implementation (at least text)
* embeddings table in MetaStore
* incremental embedding recompute by chunk checksum

Acceptance:

* add/modify text -> embeddings recomputed only for changed chunks (verified by logs/counters).

---

## Increment 5 - VectorIndex(USearch) as separate file + ops journal

**Goal:** fast ANN and resilience.

Deliverables:

* VectorIndex backend: USearch
* index storage in `indexes/...`
* `vector_index_ops` journal + startup replay
* initial `query(topK)` implementation

Acceptance:

* after app restart, index recovers (load + replay pending ops), search returns chunk_ids.

---

## Increment 6 - Semantic Relevance (Auto + Auxiliary) + edge statuses

**Goal:** truly useful links.

Deliverables:

* node similarity calculation from chunk-neighbor candidates
* creation of `pending` semantic edges (Auto)
* UI/CLI operations: accept/reject, "find related" (Aux)
* staleness for semantic links on changes (via chunk checksum + contributors)

Acceptance:

* add a new node -> get suggestions -> approve -> graph neighborhood appears.

---

## Increment 7 - Revision history + targeted recompute + staleness rules

**Goal:** edits without "recompute everything".

Deliverables:

* node revisions, rollback, attachment-manifest diff
* staleness rules:

  * Association: possibly_stale
  * Semantic: stale if contributors disappeared/weight dropped
  * Implication: stale if evidence chunks changed (preparation)

Acceptance:

* edit content -> entire graph does not collapse, only part becomes stale/recomputed.

---

## Increment 8 - Explore mode (minimal) + Implication edges + Synthesis Node (P0-lite)

**Goal:** system "magic" appears, but under control.

Deliverables:

* ExploreService:

  * retrieval: top chunks via ANN + FTS fallback
  * LLMProvider contract + local adapter (can be later)
  * structured output: implications + evidence
* persistence of Implication edges (default pending)
* Synthesis Node creation (summary + source links)

Acceptance:

* ask a question -> get 2-5 inferences, each with sources (chunk refs), can save as node.

---

## Increment 9 - Tiered BlobStore: remote backend (optional for alpha)

**Goal:** "cloud-ready" without breaking architecture.

Deliverables:

* TieredBlobStore (cache + remote) + SyncQueue
* sync_state statuses, retries, basic error telemetry
* (if S3 is not needed immediately) build a `RemoteMock` backend to test the contract end-to-end

Acceptance:

* with remote enabled: new blobs move to queued_upload -> synced; if local blob is absent, it is fetched and cached.

---

# Important risks and how to mitigate them in alpha

* **Two-store consistency (MetaStore + VectorIndex):** mitigate with ops journal + rebuild from embeddings.
* **Multimodality:** in alpha, start with text (and transcript) as "must have," add image/video as extractors/embeddings mature.
* **Cloud without pain:** tiered storage must include cache and queue; this architectural decision is already assumed.

---

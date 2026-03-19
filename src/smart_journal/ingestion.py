from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from smart_journal.contracts import (
    BlobRef,
    BlobStore,
    EmbeddingProvider,
    Extractor,
    JobQueue,
    MetaStore,
)


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    chunk_index: int
    text: str
    checksum: str


@dataclass(frozen=True, slots=True)
class EmbeddingSyncStats:
    total_chunks: int
    reused_chunks: int
    computed_chunks: int
    model_id: str | None = None
    upserted_chunk_ids: tuple[str, ...] = ()


def split_text_into_chunks(
    text: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[ChunkDraft]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")

    chunks: list[ChunkDraft] = []
    start = 0
    index = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        if end < len(normalized):
            split_point = normalized.rfind(" ", start, end)
            if split_point > start + max(8, chunk_size // 4):
                end = split_point
        chunk_text = normalized[start:end].strip()
        if not chunk_text:
            break
        checksum = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        chunks.append(ChunkDraft(chunk_index=index, text=chunk_text, checksum=checksum))
        if end >= len(normalized):
            break
        next_start = end - chunk_overlap
        start = max(next_start, start + 1)
        index += 1
    return chunks


class IngestionPipeline:
    def __init__(
        self,
        *,
        meta_store: MetaStore,
        blob_store: BlobStore,
        extractor: Extractor,
        job_queue: JobQueue,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_metric: str = "cosine",
        chunk_size: int = 500,
        chunk_overlap: int = 80,
    ) -> None:
        self._meta_store = meta_store
        self._blob_store = blob_store
        self._extractor = extractor
        self._job_queue = job_queue
        self._embedding_provider = embedding_provider
        self._embedding_metric = embedding_metric
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def enqueue_content_item(self, content_item_id: str) -> str:
        return self._job_queue.enqueue(
            "ingest_content_item",
            {"content_item_id": content_item_id},
        )

    def process_next(self) -> str | None:
        job_id = self._job_queue.run_next()
        if job_id is None:
            return None
        job = self._job_queue.get_job(job_id)
        if job is None:
            self._job_queue.set_status(job_id, "failed", error="Job payload not found.")
            return job_id

        payload_raw = job.get("payload")
        if not isinstance(payload_raw, Mapping):
            self._job_queue.set_status(job_id, "failed", error="Invalid job payload.")
            return job_id
        content_item_id = payload_raw.get("content_item_id")
        if not isinstance(content_item_id, str):
            self._job_queue.set_status(job_id, "failed", error="Missing content_item_id.")
            return job_id

        try:
            self._ingest_content_item(content_item_id)
            self._job_queue.set_status(job_id, "completed")
        except Exception as error:  # noqa: BLE001
            self._job_queue.set_status(job_id, "failed", error=str(error))
        return job_id

    def ingest_content_item_now(self, content_item_id: str) -> None:
        self._ingest_content_item(content_item_id)

    def _ingest_content_item(self, content_item_id: str) -> None:
        item = self._meta_store.get_content_item(content_item_id)
        if item is None:
            raise KeyError(f"Content item not found: {content_item_id}")

        blob_ref = BlobRef(
            scheme=str(item["blob_scheme"]),
            key=str(item["blob_key"]),
            size=int(item["blob_size"]),
            hash=str(item["blob_hash"]),
            version=(str(item["blob_version"]) if item["blob_version"] is not None else None),
        )
        mime_type = str(item["mime_type"] or "application/octet-stream")
        blob_payload = self._blob_store.open(blob_ref)

        self._meta_store.set_content_item_extraction(content_item_id, status="running")
        try:
            artifact = self._extractor.extract(blob_payload, mime_type=mime_type)
            previous_chunk_ids = {
                str(row["chunk_id"]) for row in self._meta_store.list_chunks(content_item_id)
            }
            previous_embeddings_by_checksum = self._snapshot_embeddings_by_checksum(content_item_id)

            extracted_text = artifact.text or ""
            chunk_drafts = split_text_into_chunks(
                extracted_text,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
            chunk_rows: list[dict[str, str | int]] = [
                {
                    "chunk_index": draft.chunk_index,
                    "text": draft.text,
                    "checksum": draft.checksum,
                }
                for draft in chunk_drafts
            ]
            self._meta_store.replace_content_item_chunks(content_item_id, chunk_rows)
            embedding_sync_stats = self._sync_text_embeddings(
                content_item_id,
                cached_vectors_by_checksum=previous_embeddings_by_checksum,
            )
            self._enqueue_vector_index_ops(
                previous_chunk_ids=previous_chunk_ids,
                sync_stats=embedding_sync_stats,
            )
            self._meta_store.set_content_item_extraction(
                content_item_id,
                status="done",
                extracted_text=extracted_text,
                metadata=(artifact.metadata or {}),
                error=None,
            )
        except Exception as error:  # noqa: BLE001
            self._meta_store.set_content_item_extraction(
                content_item_id,
                status="failed",
                error=str(error),
            )
            self._meta_store.replace_content_item_chunks(content_item_id, [])
            raise

    def _snapshot_embeddings_by_checksum(self, content_item_id: str) -> dict[str, list[float]]:
        if self._embedding_provider is None:
            return {}
        model_id = self._embedding_provider.model_id()
        expected_dim = self._embedding_provider.dim()
        rows = self._meta_store.list_chunk_embeddings(content_item_id, model_id=model_id)
        payload: dict[str, list[float]] = {}
        for row in rows:
            checksum = str(row["checksum"])
            if checksum in payload:
                continue
            vector = _coerce_vector(row.get("vector"))
            if len(vector) != expected_dim:
                continue
            payload[checksum] = vector
        return payload

    def _sync_text_embeddings(
        self,
        content_item_id: str,
        *,
        cached_vectors_by_checksum: Mapping[str, Sequence[float]] | None = None,
    ) -> EmbeddingSyncStats:
        if self._embedding_provider is None:
            return EmbeddingSyncStats(
                total_chunks=0,
                reused_chunks=0,
                computed_chunks=0,
                model_id=None,
                upserted_chunk_ids=(),
            )

        model_id = self._embedding_provider.model_id()
        expected_dim = self._embedding_provider.dim()
        current_chunks = self._meta_store.list_chunks(content_item_id)
        if not current_chunks:
            return EmbeddingSyncStats(
                total_chunks=0,
                reused_chunks=0,
                computed_chunks=0,
                model_id=model_id,
                upserted_chunk_ids=(),
            )

        cache: dict[str, list[float]] = {
            str(checksum): [float(value) for value in vector]
            for checksum, vector in (cached_vectors_by_checksum or {}).items()
        }

        embeddings_to_upsert: list[dict[str, Any]] = []
        pending_chunks: list[tuple[str, str, str]] = []
        reused_chunks = 0
        for chunk in current_chunks:
            checksum = str(chunk["checksum"])
            cached_vector = cache.get(checksum)
            if cached_vector is None:
                pending_chunks.append((str(chunk["chunk_id"]), str(chunk["text"]), checksum))
                continue
            if len(cached_vector) != expected_dim:
                pending_chunks.append((str(chunk["chunk_id"]), str(chunk["text"]), checksum))
                continue
            reused_chunks += 1
            embeddings_to_upsert.append(
                {
                    "chunk_id": str(chunk["chunk_id"]),
                    "model_id": model_id,
                    "dim": expected_dim,
                    "metric": self._embedding_metric,
                    "vector": list(cached_vector),
                    "checksum": checksum,
                }
            )

        computed_chunks = 0
        if pending_chunks:
            vectors = self._embedding_provider.embed_text([row[1] for row in pending_chunks])
            if len(vectors) != len(pending_chunks):
                raise ValueError(
                    "EmbeddingProvider returned unexpected vector count: "
                    f"expected {len(pending_chunks)}, got {len(vectors)}."
                )
            for (chunk_id, _, checksum), raw_vector in zip(pending_chunks, vectors, strict=True):
                vector = _coerce_vector(raw_vector)
                if len(vector) != expected_dim:
                    raise ValueError(
                        f"Embedding dim mismatch for chunk_id={chunk_id}: "
                        f"expected {expected_dim}, got {len(vector)}."
                    )
                embeddings_to_upsert.append(
                    {
                        "chunk_id": chunk_id,
                        "model_id": model_id,
                        "dim": expected_dim,
                        "metric": self._embedding_metric,
                        "vector": vector,
                        "checksum": checksum,
                    }
                )
                computed_chunks += 1

        self._meta_store.upsert_chunk_embeddings(embeddings_to_upsert)
        upserted_chunk_ids = tuple(str(row["chunk_id"]) for row in embeddings_to_upsert)
        return EmbeddingSyncStats(
            total_chunks=len(current_chunks),
            reused_chunks=reused_chunks,
            computed_chunks=computed_chunks,
            model_id=model_id,
            upserted_chunk_ids=upserted_chunk_ids,
        )

    def _enqueue_vector_index_ops(
        self,
        *,
        previous_chunk_ids: set[str],
        sync_stats: EmbeddingSyncStats,
    ) -> None:
        model_id = sync_stats.model_id
        if model_id is None:
            return

        upsert_ids = {str(chunk_id) for chunk_id in sync_stats.upserted_chunk_ids}
        delete_ids = previous_chunk_ids - upsert_ids
        ops: list[dict[str, str]] = [
            {
                "op_type": "upsert",
                "chunk_id": chunk_id,
                "model_id": model_id,
            }
            for chunk_id in sorted(upsert_ids)
        ]
        ops.extend(
            {
                "op_type": "delete",
                "chunk_id": chunk_id,
                "model_id": model_id,
            }
            for chunk_id in sorted(delete_ids)
        )
        if ops:
            self._meta_store.enqueue_vector_index_ops(ops)


def _coerce_vector(raw: Any) -> list[float]:
    if isinstance(raw, Sequence) and not isinstance(raw, str | bytes | bytearray):
        return [float(value) for value in raw]
    raise TypeError("Embedding vector must be a sequence of floats.")


def build_default_ingestion_pipeline(
    *,
    meta_store: MetaStore,
    blob_store: BlobStore,
    extractor: Extractor,
    job_queue: JobQueue,
    embedding_provider: EmbeddingProvider | None = None,
    options: Mapping[str, Any] | None = None,
) -> IngestionPipeline:
    options = options or {}
    chunk_size = int(options.get("chunk_size", 500))
    chunk_overlap = int(options.get("chunk_overlap", 80))
    embedding_metric = str(options.get("embedding_metric", "cosine"))
    return IngestionPipeline(
        meta_store=meta_store,
        blob_store=blob_store,
        extractor=extractor,
        job_queue=job_queue,
        embedding_provider=embedding_provider,
        embedding_metric=embedding_metric,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

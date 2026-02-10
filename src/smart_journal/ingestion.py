from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from smart_journal.contracts import BlobRef, BlobStore, Extractor, JobQueue, MetaStore


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    chunk_index: int
    text: str
    checksum: str


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
        chunk_size: int = 500,
        chunk_overlap: int = 80,
    ) -> None:
        self._meta_store = meta_store
        self._blob_store = blob_store
        self._extractor = extractor
        self._job_queue = job_queue
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
        except Exception as error:  # noqa: BLE001
            self._meta_store.set_content_item_extraction(
                content_item_id,
                status="failed",
                error=str(error),
            )
            self._meta_store.replace_content_item_chunks(content_item_id, [])
            raise

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
        self._meta_store.set_content_item_extraction(
            content_item_id,
            status="done",
            extracted_text=extracted_text,
            metadata=(artifact.metadata or {}),
            error=None,
        )


def build_default_ingestion_pipeline(
    *,
    meta_store: MetaStore,
    blob_store: BlobStore,
    extractor: Extractor,
    job_queue: JobQueue,
    options: Mapping[str, Any] | None = None,
) -> IngestionPipeline:
    options = options or {}
    chunk_size = int(options.get("chunk_size", 500))
    chunk_overlap = int(options.get("chunk_overlap", 80))
    return IngestionPipeline(
        meta_store=meta_store,
        blob_store=blob_store,
        extractor=extractor,
        job_queue=job_queue,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

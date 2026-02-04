from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from smart_journal.contracts import BlobInfo, BlobRef, ExtractedArtifact, VectorResult


class InMemoryBlobStore:
    def __init__(self, _: Mapping[str, Any] | None = None) -> None:
        self._blobs: dict[str, bytes] = {}

    def provider_id(self) -> str:
        return "in_memory"

    def version(self) -> str:
        return "0.1.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "content_addressed": True,
            "schemes": ["memcas"],
            "supports_delete": True,
            "supports_verify": True,
        }

    def put(self, data: bytes, *, content_type: str | None = None) -> BlobRef:
        _ = content_type
        digest = hashlib.sha256(data).hexdigest()
        key = f"sha256:{digest}"
        self._blobs[key] = data
        return BlobRef(scheme="memcas", key=key, size=len(data), hash=digest)

    def open(self, blob_ref: BlobRef) -> bytes:
        return self._blobs[blob_ref.key]

    def stat(self, blob_ref: BlobRef) -> BlobInfo:
        data = self._blobs[blob_ref.key]
        return BlobInfo(size=len(data), hash=blob_ref.hash)

    def exists(self, blob_ref: BlobRef) -> bool:
        return blob_ref.key in self._blobs

    def delete(self, blob_ref: BlobRef) -> None:
        self._blobs.pop(blob_ref.key, None)

    def verify(self, blob_ref: BlobRef) -> bool:
        data = self._blobs.get(blob_ref.key)
        if data is None:
            return False
        digest = hashlib.sha256(data).hexdigest()
        return digest == blob_ref.hash


class InMemoryMetaStore:
    def __init__(self, _: Mapping[str, Any] | None = None) -> None:
        self._graphs: dict[str, dict[str, Any]] = {}
        self._nodes: dict[str, dict[str, Any]] = {}
        self._revisions_by_node: dict[str, list[dict[str, Any]]] = {}
        self._content_items: dict[str, dict[str, Any]] = {}

    def provider_id(self) -> str:
        return "in_memory"

    def version(self) -> str:
        return "0.1.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "transactions": False,
            "fulltext_backend": "none",
            "durable": False,
            "schema_version": 1,
        }

    def begin_transaction(self) -> None:
        return None

    def create_graph(self, title: str) -> str:
        graph_id = str(uuid4())
        timestamp = _utc_now()
        self._graphs[graph_id] = {
            "graph_id": graph_id,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
            "deleted_at": None,
        }
        return graph_id

    def get_graph(
        self, graph_id: str, *, include_deleted: bool = False
    ) -> Mapping[str, Any] | None:
        graph = self._graphs.get(graph_id)
        if graph is None:
            return None
        if not include_deleted and graph["deleted_at"] is not None:
            return None
        return dict(graph)

    def create_node(self, graph_id: str, title: str, body: str = "") -> str:
        if self.get_graph(graph_id) is None:
            raise KeyError(f"Graph not found or deleted: {graph_id}")
        node_id = str(uuid4())
        timestamp = _utc_now()
        revision_id = str(uuid4())
        self._nodes[node_id] = {
            "node_id": node_id,
            "graph_id": graph_id,
            "title": title,
            "body": body,
            "created_at": timestamp,
            "updated_at": timestamp,
            "deleted_at": None,
            "current_revision_id": revision_id,
        }
        self._revisions_by_node[node_id] = [
            {
                "revision_id": revision_id,
                "node_id": node_id,
                "revision_no": 1,
                "title": title,
                "body": body,
                "created_at": timestamp,
                "comment": "create",
            }
        ]
        return node_id

    def update_node(
        self, node_id: str, *, title: str | None = None, body: str | None = None
    ) -> None:
        node = self._nodes.get(node_id)
        if node is None or node["deleted_at"] is not None:
            raise KeyError(f"Node not found or deleted: {node_id}")
        if title is not None:
            node["title"] = title
        if body is not None:
            node["body"] = body
        timestamp = _utc_now()
        node["updated_at"] = timestamp

        revisions = self._revisions_by_node.setdefault(node_id, [])
        revision_id = str(uuid4())
        revisions.append(
            {
                "revision_id": revision_id,
                "node_id": node_id,
                "revision_no": len(revisions) + 1,
                "title": node["title"],
                "body": node["body"],
                "created_at": timestamp,
                "comment": "update",
            }
        )
        node["current_revision_id"] = revision_id

    def delete_node(self, node_id: str, *, soft_delete: bool = True) -> None:
        node = self._nodes.get(node_id)
        if node is None:
            return
        if soft_delete:
            timestamp = _utc_now()
            if node["deleted_at"] is None:
                node["deleted_at"] = timestamp
                node["updated_at"] = timestamp
            return

        self._nodes.pop(node_id, None)
        self._revisions_by_node.pop(node_id, None)
        for content_item_id, item in list(self._content_items.items()):
            if item["node_id"] == node_id:
                self._content_items.pop(content_item_id, None)

    def list_nodes(
        self, graph_id: str, *, include_deleted: bool = False
    ) -> list[Mapping[str, Any]]:
        nodes: list[Mapping[str, Any]] = [
            dict(node)
            for node in self._nodes.values()
            if node["graph_id"] == graph_id and (include_deleted or node["deleted_at"] is None)
        ]
        nodes.sort(key=lambda item: str(item["created_at"]))
        return nodes

    def get_node(self, node_id: str, *, include_deleted: bool = False) -> Mapping[str, Any] | None:
        node = self._nodes.get(node_id)
        if node is None:
            return None
        if not include_deleted and node["deleted_at"] is not None:
            return None
        return dict(node)

    def list_revisions(self, node_id: str) -> list[Mapping[str, Any]]:
        revisions = self._revisions_by_node.get(node_id, [])
        return [dict(revision) for revision in revisions]

    def attach_content_item(
        self,
        node_id: str,
        blob_ref: BlobRef,
        *,
        mime_type: str | None = None,
        filename: str | None = None,
    ) -> str:
        if self.get_node(node_id) is None:
            raise KeyError(f"Node not found or deleted: {node_id}")
        content_item_id = str(uuid4())
        self._content_items[content_item_id] = {
            "content_item_id": content_item_id,
            "node_id": node_id,
            "blob_scheme": blob_ref.scheme,
            "blob_key": blob_ref.key,
            "blob_hash": blob_ref.hash,
            "blob_size": blob_ref.size,
            "blob_version": blob_ref.version,
            "mime_type": mime_type,
            "filename": filename,
            "created_at": _utc_now(),
            "deleted_at": None,
        }
        return content_item_id

    def list_content_items(
        self, node_id: str, *, include_deleted: bool = False
    ) -> list[Mapping[str, Any]]:
        items: list[Mapping[str, Any]] = [
            dict(item)
            for item in self._content_items.values()
            if item["node_id"] == node_id and (include_deleted or item["deleted_at"] is None)
        ]
        items.sort(key=lambda item: str(item["created_at"]))
        return items

    def detach_content_item(self, content_item_id: str, *, soft_delete: bool = True) -> None:
        item = self._content_items.get(content_item_id)
        if item is None:
            return
        if soft_delete:
            item["deleted_at"] = _utc_now()
            return
        self._content_items.pop(content_item_id, None)


class InMemoryVectorIndex:
    def __init__(self, _: Mapping[str, Any] | None = None) -> None:
        self._vectors: dict[str, list[float]] = {}

    def provider_id(self) -> str:
        return "in_memory"

    def version(self) -> str:
        return "0.1.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "supports_delete": True,
            "supports_filter": False,
            "metric_types": ["cosine"],
            "durable": False,
        }

    def upsert(self, vectors: Sequence[tuple[str, Sequence[float]]]) -> None:
        for external_id, vector in vectors:
            self._vectors[external_id] = [float(v) for v in vector]

    def delete(self, external_ids: Sequence[str]) -> None:
        for external_id in external_ids:
            self._vectors.pop(external_id, None)

    def query(self, vector: Sequence[float], top_k: int) -> list[VectorResult]:
        query_vector = [float(v) for v in vector]
        scored = [
            VectorResult(external_id=external_id, score=_cosine_similarity(query_vector, candidate))
            for external_id, candidate in self._vectors.items()
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def save(self) -> None:
        return None

    def load(self) -> None:
        return None


class InProcessJobQueue:
    def __init__(self, _: Mapping[str, Any] | None = None) -> None:
        self._jobs: deque[tuple[str, str, Mapping[str, Any]]] = deque()

    def provider_id(self) -> str:
        return "in_process"

    def version(self) -> str:
        return "0.1.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "durable": False,
            "priority_levels": 1,
        }

    def enqueue(self, job_name: str, payload: Mapping[str, Any]) -> str:
        job_id = str(uuid4())
        self._jobs.append((job_id, job_name, dict(payload)))
        return job_id

    def run_next(self) -> str | None:
        if not self._jobs:
            return None
        job_id, _, _ = self._jobs.popleft()
        return job_id


class PlainTextExtractor:
    def __init__(self, _: Mapping[str, Any] | None = None) -> None:
        self._supported_mime_types = {"text/plain", "text/markdown"}

    def provider_id(self) -> str:
        return "plain_text"

    def version(self) -> str:
        return "0.1.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "mime_types": sorted(self._supported_mime_types),
            "outputs": ["text"],
        }

    def supports_mime(self, mime_type: str) -> bool:
        return mime_type in self._supported_mime_types

    def extract(self, content: bytes, *, mime_type: str) -> ExtractedArtifact:
        if not self.supports_mime(mime_type):
            raise ValueError(f"Unsupported MIME type: {mime_type}")
        text = content.decode("utf-8", errors="replace")
        return ExtractedArtifact(
            content_type="text/plain",
            text=text,
            metadata={"source_mime_type": mime_type},
        )


class MockEmbeddingProvider:
    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        options = options or {}
        self._dim = int(options.get("dim", 8))
        self._normalize = bool(options.get("normalize", True))

    def provider_id(self) -> str:
        return "mock_text"

    def version(self) -> str:
        return "0.1.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "text": True,
            "image": False,
            "audio": False,
            "video": False,
            "dim": self._dim,
            "batch_limits": 128,
        }

    def model_id(self) -> str:
        return "mock-text-embed-v1"

    def dim(self) -> int:
        return self._dim

    def normalize(self) -> bool:
        return self._normalize

    def embed_text(self, chunks: Sequence[str]) -> list[list[float]]:
        return [self._vector_from_payload(chunk.encode("utf-8")) for chunk in chunks]

    def embed_image(self, frames: Sequence[bytes]) -> list[list[float]]:
        if frames:
            raise NotImplementedError("Image embeddings are not supported by mock_text.")
        return []

    def embed_audio(self, segments: Sequence[bytes]) -> list[list[float]]:
        if segments:
            raise NotImplementedError("Audio embeddings are not supported by mock_text.")
        return []

    def embed_video(self, segments: Sequence[bytes]) -> list[list[float]]:
        if segments:
            raise NotImplementedError("Video embeddings are not supported by mock_text.")
        return []

    def _vector_from_payload(self, payload: bytes) -> list[float]:
        digest = hashlib.sha256(payload).digest()
        raw = [((digest[index % len(digest)] / 255.0) * 2.0) - 1.0 for index in range(self._dim)]
        if not self._normalize:
            return raw
        norm = math.sqrt(sum(value * value for value in raw))
        if norm == 0.0:
            return raw
        return [value / norm for value in raw]


class MockLLMProvider:
    def __init__(self, _: Mapping[str, Any] | None = None) -> None:
        pass

    def provider_id(self) -> str:
        return "mock_chat"

    def version(self) -> str:
        return "0.1.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "supports_vision": False,
            "structured_output": True,
            "tools": False,
        }

    def model_id(self) -> str:
        return "mock-llm-v1"

    def context_window(self) -> int:
        return 4096

    def supports_vision(self) -> bool:
        return False

    def generate_structured(self, prompt: str, schema: Mapping[str, Any]) -> Mapping[str, Any]:
        response: dict[str, Any] = {key: None for key in schema}
        response["prompt_preview"] = prompt[:80]
        response["provider"] = self.provider_id()
        return response

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        if not messages:
            return "mock_chat: no messages"
        last = messages[-1].get("content", "")
        return f"mock_chat: {last[:120]}"


def _cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if not lhs or not rhs or len(lhs) != len(rhs):
        return 0.0
    dot = sum(a * b for a, b in zip(lhs, rhs, strict=False))
    lhs_norm = math.sqrt(sum(a * a for a in lhs))
    rhs_norm = math.sqrt(sum(b * b for b in rhs))
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 0.0
    return dot / (lhs_norm * rhs_norm)


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")

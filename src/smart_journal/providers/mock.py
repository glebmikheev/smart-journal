from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import Mapping, Sequence
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
        self._nodes: dict[str, dict[str, str]] = {}

    def provider_id(self) -> str:
        return "in_memory"

    def version(self) -> str:
        return "0.1.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "transactions": False,
            "fulltext_backend": "none",
            "durable": False,
        }

    def begin_transaction(self) -> None:
        return None

    def create_node(self, graph_id: str, title: str) -> str:
        node_id = str(uuid4())
        self._nodes[node_id] = {"node_id": node_id, "graph_id": graph_id, "title": title}
        return node_id

    def get_node(self, node_id: str) -> Mapping[str, Any] | None:
        node = self._nodes.get(node_id)
        return dict(node) if node is not None else None


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

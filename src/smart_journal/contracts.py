from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

CapabilityValue = bool | int | float | str | list[str]
Capabilities = Mapping[str, CapabilityValue]


@dataclass(frozen=True, slots=True)
class BlobRef:
    scheme: str
    key: str
    size: int
    hash: str
    version: str | None = None


@dataclass(frozen=True, slots=True)
class BlobInfo:
    size: int
    hash: str
    modified_unix: float | None = None


@dataclass(frozen=True, slots=True)
class ExtractedArtifact:
    content_type: str
    text: str | None = None
    metadata: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class VectorResult:
    external_id: str
    score: float


@runtime_checkable
class ProviderInfo(Protocol):
    def provider_id(self) -> str: ...

    def version(self) -> str: ...

    def capabilities(self) -> Capabilities: ...


@runtime_checkable
class BlobStore(ProviderInfo, Protocol):
    def put(self, data: bytes, *, content_type: str | None = None) -> BlobRef: ...

    def open(self, blob_ref: BlobRef) -> bytes: ...

    def stat(self, blob_ref: BlobRef) -> BlobInfo: ...

    def exists(self, blob_ref: BlobRef) -> bool: ...

    def delete(self, blob_ref: BlobRef) -> None: ...

    def verify(self, blob_ref: BlobRef) -> bool: ...


@runtime_checkable
class MetaStore(ProviderInfo, Protocol):
    def begin_transaction(self) -> None: ...

    def create_node(self, graph_id: str, title: str) -> str: ...

    def get_node(self, node_id: str) -> Mapping[str, Any] | None: ...


@runtime_checkable
class VectorIndex(ProviderInfo, Protocol):
    def upsert(self, vectors: Sequence[tuple[str, Sequence[float]]]) -> None: ...

    def delete(self, external_ids: Sequence[str]) -> None: ...

    def query(self, vector: Sequence[float], top_k: int) -> list[VectorResult]: ...

    def save(self) -> None: ...

    def load(self) -> None: ...


@runtime_checkable
class JobQueue(ProviderInfo, Protocol):
    def enqueue(self, job_name: str, payload: Mapping[str, Any]) -> str: ...

    def run_next(self) -> str | None: ...


@runtime_checkable
class Extractor(ProviderInfo, Protocol):
    def supports_mime(self, mime_type: str) -> bool: ...

    def extract(self, content: bytes, *, mime_type: str) -> ExtractedArtifact: ...


@runtime_checkable
class EmbeddingProvider(ProviderInfo, Protocol):
    def model_id(self) -> str: ...

    def dim(self) -> int: ...

    def normalize(self) -> bool: ...

    def embed_text(self, chunks: Sequence[str]) -> list[list[float]]: ...

    def embed_image(self, frames: Sequence[bytes]) -> list[list[float]]: ...

    def embed_audio(self, segments: Sequence[bytes]) -> list[list[float]]: ...

    def embed_video(self, segments: Sequence[bytes]) -> list[list[float]]: ...


@runtime_checkable
class LLMProvider(ProviderInfo, Protocol):
    def model_id(self) -> str: ...

    def context_window(self) -> int: ...

    def supports_vision(self) -> bool: ...

    def generate_structured(self, prompt: str, schema: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str: ...


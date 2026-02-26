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

    def create_graph(self, title: str) -> str: ...

    def get_graph(
        self, graph_id: str, *, include_deleted: bool = False
    ) -> Mapping[str, Any] | None: ...

    def create_node(self, graph_id: str, title: str, body: str = "") -> str: ...

    def update_node(
        self, node_id: str, *, title: str | None = None, body: str | None = None
    ) -> None: ...

    def delete_node(self, node_id: str, *, soft_delete: bool = True) -> None: ...

    def list_nodes(
        self, graph_id: str, *, include_deleted: bool = False
    ) -> list[Mapping[str, Any]]: ...

    def get_node(
        self, node_id: str, *, include_deleted: bool = False
    ) -> Mapping[str, Any] | None: ...

    def list_revisions(self, node_id: str) -> list[Mapping[str, Any]]: ...

    def attach_content_item(
        self,
        node_id: str,
        blob_ref: BlobRef,
        *,
        mime_type: str | None = None,
        filename: str | None = None,
    ) -> str: ...

    def list_content_items(
        self, node_id: str, *, include_deleted: bool = False
    ) -> list[Mapping[str, Any]]: ...

    def get_content_item(
        self, content_item_id: str, *, include_deleted: bool = False
    ) -> Mapping[str, Any] | None: ...

    def detach_content_item(
        self, content_item_id: str, *, soft_delete: bool = True
    ) -> None: ...

    def set_content_item_extraction(
        self,
        content_item_id: str,
        *,
        status: str,
        extracted_text: str | None = None,
        metadata: Mapping[str, str] | None = None,
        error: str | None = None,
    ) -> None: ...

    def replace_content_item_chunks(
        self,
        content_item_id: str,
        chunks: Sequence[Mapping[str, str | int]],
    ) -> list[str]: ...

    def list_chunks(
        self, content_item_id: str, *, include_deleted: bool = False
    ) -> list[Mapping[str, Any]]: ...

    def upsert_chunk_embeddings(
        self,
        embeddings: Sequence[Mapping[str, Any]],
    ) -> None: ...

    def list_chunk_embeddings(
        self,
        content_item_id: str,
        *,
        model_id: str | None = None,
    ) -> list[Mapping[str, Any]]: ...

    def create_tag(self, graph_id: str, name: str) -> str: ...

    def list_tags(
        self, graph_id: str, *, include_deleted: bool = False
    ) -> list[Mapping[str, Any]]: ...

    def delete_tag(self, tag_id: str, *, soft_delete: bool = True) -> None: ...

    def add_node_tag(self, node_id: str, tag_id: str) -> None: ...

    def remove_node_tag(self, node_id: str, tag_id: str) -> None: ...

    def list_node_tags(self, node_id: str) -> list[Mapping[str, Any]]: ...

    def create_group(self, graph_id: str, name: str) -> str: ...

    def list_groups(
        self, graph_id: str, *, include_deleted: bool = False
    ) -> list[Mapping[str, Any]]: ...

    def delete_group(self, group_id: str, *, soft_delete: bool = True) -> None: ...

    def add_node_to_group(self, node_id: str, group_id: str) -> None: ...

    def remove_node_from_group(self, node_id: str, group_id: str) -> None: ...

    def list_node_groups(self, node_id: str) -> list[Mapping[str, Any]]: ...

    def search_fulltext(
        self,
        query: str,
        *,
        graph_id: str | None = None,
        group_id: str | None = None,
        tag_ids: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[Mapping[str, Any]]: ...


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

    def set_status(self, job_id: str, status: str, *, error: str | None = None) -> None: ...

    def get_job(self, job_id: str) -> Mapping[str, Any] | None: ...


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

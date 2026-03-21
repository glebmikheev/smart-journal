from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from smart_journal.config import load_config
from smart_journal.contracts import ProviderInfo
from smart_journal.explore import ExploreService
from smart_journal.factories import ComponentBundle, ComponentFactory
from smart_journal.ingestion import IngestionPipeline, build_default_ingestion_pipeline
from smart_journal.registry import ProviderRegistry, build_default_registry
from smart_journal.semantic import SemanticLinker
from smart_journal.vector_ops import (
    VectorIndexOpsReplayer,
    VectorIndexReplayStats,
    rebuild_vector_index_from_embeddings,
)

UPLOAD_FILE = File(...)
FORM_INGEST_NOW = Form(default=True)
FORM_REPLAY_VECTOR_OPS = Form(default=True)
QUERY_INCLUDE_DELETED = Query(default=False)
QUERY_SEARCH_TEXT = Query(min_length=1, description="FTS query.")
QUERY_GRAPH_ID = Query(default=None)
QUERY_GROUP_ID = Query(default=None)
QUERY_TAG_IDS = Query(default=None)
QUERY_SEARCH_LIMIT = Query(default=20, ge=1, le=200)
QUERY_REPLAY_LIMIT = Query(default=1000, ge=1, le=50_000)
QUERY_EDGE_TYPE = Query(default=None)
QUERY_EDGE_STATUS = Query(default=None)
QUERY_EDGE_NODE_ID = Query(default=None)
QUERY_EDGE_LIMIT = Query(default=200, ge=1, le=2_000)


@dataclass(slots=True)
class AppRuntime:
    registry: ProviderRegistry
    bundle: ComponentBundle
    ingestion: IngestionPipeline


@dataclass(frozen=True, slots=True)
class EmbeddingPreloadStatus:
    enabled: bool
    ready: bool
    strict: bool
    elapsed_ms: int | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class VectorBootstrapRebuildStatus:
    enabled: bool
    durable_backend: bool
    replay_applied_ops: int
    scanned_chunks: int
    upserted_vectors: int
    missing_embeddings: int


class CreateGraphRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class CreateNodeRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    body: str = ""


class UpdateNodeRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    body: str | None = None


class IngestRequest(BaseModel):
    replay_vector_ops: bool = True


class VectorQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=200)


class NameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class SemanticSuggestRequest(BaseModel):
    top_k_per_chunk: int = Field(default=10, ge=1, le=200)
    max_suggestions: int = Field(default=10, ge=1, le=200)
    replay_vector_ops: bool = True


class SemanticRecomputeRequest(BaseModel):
    top_k_per_chunk: int = Field(default=10, ge=1, le=200)
    max_suggestions: int = Field(default=10, ge=1, le=200)
    replay_vector_ops: bool = True


class EdgeStatusRequest(BaseModel):
    status: str = Field(min_length=1, max_length=32)


class CreateAssociationEdgeRequest(BaseModel):
    from_node_id: str = Field(min_length=1)
    to_node_id: str = Field(min_length=1)
    status: str = Field(default="accepted", min_length=1, max_length=32)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    note: str | None = Field(default=None, max_length=2_000)


class UpdateAssociationEdgeRequest(BaseModel):
    status: str | None = Field(default=None, min_length=1, max_length=32)
    weight: float | None = Field(default=None, ge=0.0, le=1.0)
    note: str | None = Field(default=None, max_length=2_000)
    clear_note: bool = False


class ExploreRunRequest(BaseModel):
    query: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    group_id: str | None = None
    top_k_chunks: int = Field(default=12, ge=1, le=200)
    max_inferences: int = Field(default=5, ge=1, le=50)
    create_synthesis: bool = True
    replay_vector_ops: bool = True


def create_app(config_path: Path | None = None) -> FastAPI:
    resolved_config_path = _resolve_config_path(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        registry = build_default_registry()
        config = load_config(resolved_config_path)
        bundle = ComponentFactory(registry).create(config)
        bundle.vector_index.load()
        replay_stats = _replay_vector_index(bundle=bundle, limit=10_000)
        ingestion = build_default_ingestion_pipeline(
            meta_store=bundle.meta_store,
            blob_store=bundle.blob_store,
            extractor=bundle.extractor,
            job_queue=bundle.job_queue,
            embedding_provider=bundle.embedding_provider,
        )
        preload_status = _preload_embedding_provider(bundle.embedding_provider)
        rebuild_status = _rebuild_index_if_needed(
            bundle=bundle,
            replay_stats=replay_stats,
        )

        app.state.runtime = AppRuntime(
            registry=registry,
            bundle=bundle,
            ingestion=ingestion,
        )
        app.state.vector_replay_bootstrap = _replay_stats_payload(replay_stats)
        app.state.embedding_preload = _embedding_preload_payload(preload_status)
        app.state.vector_rebuild_bootstrap = _vector_rebuild_payload(rebuild_status)
        try:
            yield
        finally:
            _close_bundle_resources(bundle)

    app = FastAPI(
        title="Smart Journal API",
        version="0.6.0",
        lifespan=lifespan,
    )

    cors_origins = _cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api = APIRouter(prefix="/api")

    @api.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return {
            "status": "ok",
            "bootstrap": dict(getattr(request.app.state, "vector_replay_bootstrap", {})),
            "vector_rebuild": dict(getattr(request.app.state, "vector_rebuild_bootstrap", {})),
            "embedding_preload": dict(getattr(request.app.state, "embedding_preload", {})),
        }

    @api.get("/providers/available")
    def list_available_providers(request: Request) -> dict[str, list[dict[str, Any]]]:
        runtime = _runtime_from_request(request)
        payload: dict[str, list[dict[str, Any]]] = {}
        for category, descriptors in runtime.registry.available_all().items():
            payload[category] = [
                {
                    "provider_id": descriptor.provider_id,
                    "version": descriptor.version,
                    "capabilities": dict(descriptor.capabilities),
                }
                for descriptor in descriptors
            ]
        return payload

    @api.get("/providers/selected")
    def list_selected_providers(request: Request) -> dict[str, dict[str, Any]]:
        runtime = _runtime_from_request(request)
        return _selected_providers_payload(runtime.bundle)

    @api.get("/graphs")
    def list_graphs(
        request: Request,
        include_deleted: bool = QUERY_INCLUDE_DELETED,
    ) -> list[dict[str, Any]]:
        runtime = _runtime_from_request(request)
        return [_to_dict(graph) for graph in runtime.bundle.meta_store.list_graphs(
            include_deleted=include_deleted
        )]

    @api.post("/graphs", status_code=201)
    def create_graph(payload: CreateGraphRequest, request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        title = payload.title.strip()
        if not title:
            raise HTTPException(status_code=400, detail="Graph title must not be empty.")
        graph_id = runtime.bundle.meta_store.create_graph(title)
        graph = runtime.bundle.meta_store.get_graph(graph_id, include_deleted=True)
        if graph is None:
            raise HTTPException(status_code=500, detail="Graph creation failed.")
        return _to_dict(graph)

    @api.get("/graphs/{graph_id}")
    def get_graph(
        graph_id: str,
        request: Request,
        include_deleted: bool = QUERY_INCLUDE_DELETED,
    ) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        graph = runtime.bundle.meta_store.get_graph(graph_id, include_deleted=include_deleted)
        if graph is None:
            raise HTTPException(status_code=404, detail=f"Graph not found: {graph_id}")
        return _to_dict(graph)

    @api.get("/graphs/{graph_id}/details")
    def get_graph_details(
        graph_id: str,
        request: Request,
        include_deleted: bool = QUERY_INCLUDE_DELETED,
    ) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        graph = runtime.bundle.meta_store.get_graph(graph_id, include_deleted=include_deleted)
        if graph is None:
            raise HTTPException(status_code=404, detail=f"Graph not found: {graph_id}")
        try:
            nodes = runtime.bundle.meta_store.list_nodes(graph_id, include_deleted=include_deleted)
            groups = runtime.bundle.meta_store.list_groups(
                graph_id,
                include_deleted=include_deleted,
            )
            tags = runtime.bundle.meta_store.list_tags(graph_id, include_deleted=include_deleted)
            edges = runtime.bundle.meta_store.list_edges(
                graph_id=graph_id,
                include_deleted=include_deleted,
                limit=1_000,
            )
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)

        group_counts = {str(group["group_id"]): 0 for group in groups}
        tag_counts = {str(tag["tag_id"]): 0 for tag in tags}
        for node in nodes:
            node_id = str(node["node_id"])
            for group in runtime.bundle.meta_store.list_node_groups(node_id):
                group_id = str(group["group_id"])
                if group_id in group_counts:
                    group_counts[group_id] += 1
            for tag in runtime.bundle.meta_store.list_node_tags(node_id):
                tag_id = str(tag["tag_id"])
                if tag_id in tag_counts:
                    tag_counts[tag_id] += 1

        return {
            "graph": _to_dict(graph),
            "nodes": [_to_dict(node) for node in nodes],
            "groups": [
                {
                    **_to_dict(group),
                    "node_count": group_counts.get(str(group["group_id"]), 0),
                }
                for group in groups
            ],
            "tags": [
                {
                    **_to_dict(tag),
                    "node_count": tag_counts.get(str(tag["tag_id"]), 0),
                }
                for tag in tags
            ],
            "edges": {
                "supported": True,
                "count": len(edges),
                "by_type": _count_by_key(edges, "edge_type"),
                "by_status": _count_by_key(edges, "status"),
                "items": [_edge_payload(edge) for edge in edges],
            },
        }

    @api.get("/graphs/{graph_id}/topology")
    def get_graph_topology(
        graph_id: str,
        request: Request,
        include_deleted: bool = QUERY_INCLUDE_DELETED,
    ) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        graph = runtime.bundle.meta_store.get_graph(graph_id, include_deleted=include_deleted)
        if graph is None:
            raise HTTPException(status_code=404, detail=f"Graph not found: {graph_id}")
        try:
            nodes = runtime.bundle.meta_store.list_nodes(graph_id, include_deleted=include_deleted)
            groups = runtime.bundle.meta_store.list_groups(
                graph_id,
                include_deleted=include_deleted,
            )
            tags = runtime.bundle.meta_store.list_tags(graph_id, include_deleted=include_deleted)
            edges = runtime.bundle.meta_store.list_edges(
                graph_id=graph_id,
                include_deleted=include_deleted,
                limit=1_000,
            )
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)

        group_counts = {str(group["group_id"]): 0 for group in groups}
        tag_counts = {str(tag["tag_id"]): 0 for tag in tags}
        group_links: list[dict[str, str]] = []
        tag_links: list[dict[str, str]] = []
        node_items: list[dict[str, Any]] = []

        for node in nodes:
            node_id = str(node["node_id"])
            node_groups = runtime.bundle.meta_store.list_node_groups(node_id)
            node_tags = runtime.bundle.meta_store.list_node_tags(node_id)
            group_ids = [str(group["group_id"]) for group in node_groups]
            tag_ids = [str(tag["tag_id"]) for tag in node_tags]
            for group_id in group_ids:
                if group_id in group_counts:
                    group_counts[group_id] += 1
                group_links.append({"group_id": group_id, "node_id": node_id})
            for tag_id in tag_ids:
                if tag_id in tag_counts:
                    tag_counts[tag_id] += 1
                tag_links.append({"tag_id": tag_id, "node_id": node_id})
            node_items.append(
                {
                    **_to_dict(node),
                    "group_ids": group_ids,
                    "tag_ids": tag_ids,
                }
            )

        return {
            "graph": _to_dict(graph),
            "nodes": node_items,
            "groups": [
                {
                    **_to_dict(group),
                    "node_count": group_counts.get(str(group["group_id"]), 0),
                }
                for group in groups
            ],
            "tags": [
                {
                    **_to_dict(tag),
                    "node_count": tag_counts.get(str(tag["tag_id"]), 0),
                }
                for tag in tags
            ],
            "links": {
                "group_membership": group_links,
                "tag_membership": tag_links,
            },
            "edges": {
                "supported": True,
                "count": len(edges),
                "by_type": _count_by_key(edges, "edge_type"),
                "by_status": _count_by_key(edges, "status"),
                "items": [_edge_payload(edge) for edge in edges],
            },
        }

    @api.get("/graphs/{graph_id}/edges")
    def list_graph_edges(
        graph_id: str,
        request: Request,
        edge_type: str | None = QUERY_EDGE_TYPE,
        status: str | None = QUERY_EDGE_STATUS,
        node_id: str | None = QUERY_EDGE_NODE_ID,
        include_deleted: bool = QUERY_INCLUDE_DELETED,
        limit: int = QUERY_EDGE_LIMIT,
    ) -> list[dict[str, Any]]:
        runtime = _runtime_from_request(request)
        graph = runtime.bundle.meta_store.get_graph(graph_id, include_deleted=include_deleted)
        if graph is None:
            raise HTTPException(status_code=404, detail=f"Graph not found: {graph_id}")
        try:
            edges = runtime.bundle.meta_store.list_edges(
                graph_id=graph_id,
                node_id=node_id,
                edge_type=edge_type,
                status=status,
                include_deleted=include_deleted,
                limit=limit,
            )
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return [_to_dict(edge) for edge in edges]

    @api.post("/graphs/{graph_id}/edges/association", status_code=201)
    def create_association_edge(
        graph_id: str,
        payload: CreateAssociationEdgeRequest,
        request: Request,
    ) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        from_node_id = payload.from_node_id.strip()
        to_node_id = payload.to_node_id.strip()
        if not from_node_id or not to_node_id:
            raise HTTPException(
                status_code=400,
                detail="from_node_id/to_node_id must not be empty.",
            )
        note = payload.note.strip() if payload.note is not None else ""
        provenance = {"note": note} if note else None
        try:
            edge_id = runtime.bundle.meta_store.create_edge(
                graph_id=graph_id,
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                edge_type="association",
                status=payload.status.strip().lower(),
                weight=float(payload.weight),
                provenance=provenance,
                created_by="user",
            )
            edge = runtime.bundle.meta_store.get_edge(edge_id=edge_id, include_deleted=True)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        if edge is None:
            raise HTTPException(status_code=500, detail="Association edge creation failed.")
        return _edge_payload(edge)

    @api.get("/graphs/{graph_id}/nodes")
    def list_graph_nodes(
        graph_id: str,
        request: Request,
        include_deleted: bool = QUERY_INCLUDE_DELETED,
    ) -> list[dict[str, Any]]:
        runtime = _runtime_from_request(request)
        try:
            return [
                _to_dict(node)
                for node in runtime.bundle.meta_store.list_nodes(
                    graph_id,
                    include_deleted=include_deleted,
                )
            ]
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)

    @api.get("/graphs/{graph_id}/groups")
    def list_graph_groups(
        graph_id: str,
        request: Request,
        include_deleted: bool = QUERY_INCLUDE_DELETED,
    ) -> list[dict[str, Any]]:
        runtime = _runtime_from_request(request)
        try:
            groups = runtime.bundle.meta_store.list_groups(
                graph_id,
                include_deleted=include_deleted,
            )
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return [_to_dict(group) for group in groups]

    @api.post("/graphs/{graph_id}/groups", status_code=201)
    def create_graph_group(graph_id: str, payload: NameRequest, request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Group name must not be empty.")
        try:
            group_id = runtime.bundle.meta_store.create_group(graph_id, name)
            groups = runtime.bundle.meta_store.list_groups(graph_id, include_deleted=True)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        for group in groups:
            if str(group["group_id"]) == group_id:
                return _to_dict(group)
        raise HTTPException(status_code=500, detail="Group creation failed.")

    @api.get("/graphs/{graph_id}/tags")
    def list_graph_tags(
        graph_id: str,
        request: Request,
        include_deleted: bool = QUERY_INCLUDE_DELETED,
    ) -> list[dict[str, Any]]:
        runtime = _runtime_from_request(request)
        try:
            tags = runtime.bundle.meta_store.list_tags(graph_id, include_deleted=include_deleted)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return [_to_dict(tag) for tag in tags]

    @api.post("/graphs/{graph_id}/tags", status_code=201)
    def create_graph_tag(graph_id: str, payload: NameRequest, request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Tag name must not be empty.")
        try:
            tag_id = runtime.bundle.meta_store.create_tag(graph_id, name)
            tags = runtime.bundle.meta_store.list_tags(graph_id, include_deleted=True)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        for tag in tags:
            if str(tag["tag_id"]) == tag_id:
                return _to_dict(tag)
        raise HTTPException(status_code=500, detail="Tag creation failed.")

    @api.post("/graphs/{graph_id}/nodes", status_code=201)
    def create_node(graph_id: str, payload: CreateNodeRequest, request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        title = payload.title.strip()
        if not title:
            raise HTTPException(status_code=400, detail="Node title must not be empty.")
        try:
            node_id = runtime.bundle.meta_store.create_node(
                graph_id=graph_id,
                title=title,
                body=payload.body,
            )
            node = runtime.bundle.meta_store.get_node(node_id, include_deleted=True)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        if node is None:
            raise HTTPException(status_code=500, detail="Node creation failed.")
        return _to_dict(node)

    @api.get("/nodes/{node_id}")
    def get_node(
        node_id: str,
        request: Request,
        include_deleted: bool = QUERY_INCLUDE_DELETED,
    ) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        node = runtime.bundle.meta_store.get_node(node_id, include_deleted=include_deleted)
        if node is None:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
        return _to_dict(node)

    @api.get("/nodes/{node_id}/details")
    def get_node_details(
        node_id: str,
        request: Request,
        include_deleted: bool = QUERY_INCLUDE_DELETED,
    ) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        node = runtime.bundle.meta_store.get_node(node_id, include_deleted=include_deleted)
        if node is None:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
        try:
            revisions = runtime.bundle.meta_store.list_revisions(node_id)
            content_items = runtime.bundle.meta_store.list_content_items(
                node_id,
                include_deleted=include_deleted,
            )
            tags = runtime.bundle.meta_store.list_node_tags(node_id)
            groups = runtime.bundle.meta_store.list_node_groups(node_id)
            edges = runtime.bundle.meta_store.list_edges(
                graph_id=str(node["graph_id"]),
                node_id=node_id,
                include_deleted=include_deleted,
                limit=1_000,
            )
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)

        content_with_chunks: list[dict[str, Any]] = []
        for content_item in content_items:
            chunks = runtime.bundle.meta_store.list_chunks(
                str(content_item["content_item_id"]),
                include_deleted=include_deleted,
            )
            content_with_chunks.append(
                {
                    "content_item": _to_dict(content_item),
                    "chunks": [_to_dict(chunk) for chunk in chunks],
                }
            )

        return {
            "node": _to_dict(node),
            "revisions": [_to_dict(revision) for revision in revisions],
            "content_items": [_to_dict(item) for item in content_items],
            "content_chunks": content_with_chunks,
            "tags": [_to_dict(tag) for tag in tags],
            "groups": [_to_dict(group) for group in groups],
            "relationships": {
                "supported": True,
                "count": len(edges),
                "by_type": _count_by_key(edges, "edge_type"),
                "by_status": _count_by_key(edges, "status"),
                "items": [_node_relationship_payload(edge=edge, node_id=node_id) for edge in edges],
            },
        }

    @api.get("/nodes/{node_id}/edges")
    def list_node_edges(
        node_id: str,
        request: Request,
        edge_type: str | None = QUERY_EDGE_TYPE,
        status: str | None = QUERY_EDGE_STATUS,
        include_deleted: bool = QUERY_INCLUDE_DELETED,
        limit: int = QUERY_EDGE_LIMIT,
    ) -> list[dict[str, Any]]:
        runtime = _runtime_from_request(request)
        node = runtime.bundle.meta_store.get_node(node_id, include_deleted=include_deleted)
        if node is None:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
        try:
            edges = runtime.bundle.meta_store.list_edges(
                graph_id=str(node["graph_id"]),
                node_id=node_id,
                edge_type=edge_type,
                status=status,
                include_deleted=include_deleted,
                limit=limit,
            )
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return [_to_dict(edge) for edge in edges]

    @api.post("/nodes/{node_id}/semantic/suggest")
    def suggest_semantic_links(
        node_id: str,
        payload: SemanticSuggestRequest,
        request: Request,
    ) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        node = runtime.bundle.meta_store.get_node(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
        replay_stats: dict[str, Any] | None = None
        try:
            if payload.replay_vector_ops:
                replay_stats = _replay_stats_payload(
                    _replay_vector_index(bundle=runtime.bundle, limit=10_000)
                )
            linker = SemanticLinker(
                meta_store=runtime.bundle.meta_store,
                vector_index=runtime.bundle.vector_index,
                model_id=runtime.bundle.embedding_provider.model_id(),
            )
            suggestions = linker.suggest_for_node(
                node_id=node_id,
                top_k_per_chunk=payload.top_k_per_chunk,
                max_suggestions=payload.max_suggestions,
            )
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return {
            "node_id": node_id,
            "graph_id": str(node["graph_id"]),
            "suggestions": [
                {
                    "edge_id": suggestion.edge_id,
                    "from_node_id": suggestion.from_node_id,
                    "to_node_id": suggestion.to_node_id,
                    "status": suggestion.status,
                    "weight": float(suggestion.weight),
                    "supporting_chunk_hits": int(suggestion.supporting_chunk_hits),
                }
                for suggestion in suggestions
            ],
            "vector_replay": replay_stats,
        }

    @api.post("/nodes/{node_id}/semantic/recompute")
    def recompute_semantic_links(
        node_id: str,
        payload: SemanticRecomputeRequest,
        request: Request,
    ) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        node = runtime.bundle.meta_store.get_node(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
        replay_stats: dict[str, Any] | None = None
        try:
            if payload.replay_vector_ops:
                replay_stats = _replay_stats_payload(
                    _replay_vector_index(bundle=runtime.bundle, limit=10_000)
                )
            linker = SemanticLinker(
                meta_store=runtime.bundle.meta_store,
                vector_index=runtime.bundle.vector_index,
                model_id=runtime.bundle.embedding_provider.model_id(),
            )
            result = linker.recompute_for_node(
                node_id=node_id,
                top_k_per_chunk=payload.top_k_per_chunk,
                max_suggestions=payload.max_suggestions,
            )
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return {
            "node_id": node_id,
            "graph_id": str(node["graph_id"]),
            "suggestions": [
                {
                    "edge_id": suggestion.edge_id,
                    "from_node_id": suggestion.from_node_id,
                    "to_node_id": suggestion.to_node_id,
                    "status": suggestion.status,
                    "weight": float(suggestion.weight),
                    "supporting_chunk_hits": int(suggestion.supporting_chunk_hits),
                }
                for suggestion in result.suggestions
            ],
            "stale_edge_ids": list(result.stale_edge_ids),
            "stale_edge_count": int(result.stale_edge_count),
            "vector_replay": replay_stats,
        }

    @api.post("/explore/run")
    def run_explore(payload: ExploreRunRequest, request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        replay_stats: dict[str, Any] | None = None
        try:
            if payload.replay_vector_ops:
                replay_stats = _replay_stats_payload(
                    _replay_vector_index(bundle=runtime.bundle, limit=10_000)
                )
            service = ExploreService(
                meta_store=runtime.bundle.meta_store,
                vector_index=runtime.bundle.vector_index,
                embedding_provider=runtime.bundle.embedding_provider,
                llm_provider=runtime.bundle.llm_provider,
            )
            result = service.run(
                graph_id=payload.graph_id.strip(),
                query=payload.query,
                group_id=(payload.group_id.strip() if payload.group_id else None),
                top_k_chunks=payload.top_k_chunks,
                max_inferences=payload.max_inferences,
                create_synthesis=payload.create_synthesis,
            )
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return {
            "query": result.query,
            "graph_id": result.graph_id,
            "explore_session_id": result.explore_session_id,
            "prompt_hash": result.prompt_hash,
            "retrieval": [
                {
                    "chunk_id": row.chunk_id,
                    "node_id": row.node_id,
                    "content_item_id": row.content_item_id,
                    "score": float(row.score),
                    "text_preview": _preview_text(row.text),
                    "source": row.source,
                }
                for row in result.retrieval
            ],
            "inferences": [
                {
                    "edge_id": row.edge_id,
                    "from_node_id": row.from_node_id,
                    "to_node_id": row.to_node_id,
                    "weight": float(row.weight),
                    "statement": row.statement,
                    "evidence_chunk_ids": list(row.evidence_chunk_ids),
                    "provenance": dict(row.provenance),
                }
                for row in result.inferences
            ],
            "synthesis_node_id": result.synthesis_node_id,
            "llm_payload": dict(result.llm_payload),
            "vector_replay": replay_stats,
        }

    @api.get("/nodes/{node_id}/tags")
    def list_node_tags(node_id: str, request: Request) -> list[dict[str, Any]]:
        runtime = _runtime_from_request(request)
        try:
            tags = runtime.bundle.meta_store.list_node_tags(node_id)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return [_to_dict(tag) for tag in tags]

    @api.post("/nodes/{node_id}/tags/{tag_id}")
    def add_node_tag(node_id: str, tag_id: str, request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        try:
            runtime.bundle.meta_store.add_node_tag(node_id, tag_id)
            tags = runtime.bundle.meta_store.list_node_tags(node_id)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return {"node_id": node_id, "tags": [_to_dict(tag) for tag in tags]}

    @api.delete("/nodes/{node_id}/tags/{tag_id}")
    def remove_node_tag(node_id: str, tag_id: str, request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        try:
            runtime.bundle.meta_store.remove_node_tag(node_id, tag_id)
            tags = runtime.bundle.meta_store.list_node_tags(node_id)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return {"node_id": node_id, "tags": [_to_dict(tag) for tag in tags]}

    @api.get("/nodes/{node_id}/groups")
    def list_node_groups(node_id: str, request: Request) -> list[dict[str, Any]]:
        runtime = _runtime_from_request(request)
        try:
            groups = runtime.bundle.meta_store.list_node_groups(node_id)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return [_to_dict(group) for group in groups]

    @api.post("/nodes/{node_id}/groups/{group_id}")
    def add_node_to_group(node_id: str, group_id: str, request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        try:
            runtime.bundle.meta_store.add_node_to_group(node_id, group_id)
            groups = runtime.bundle.meta_store.list_node_groups(node_id)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return {"node_id": node_id, "groups": [_to_dict(group) for group in groups]}

    @api.delete("/nodes/{node_id}/groups/{group_id}")
    def remove_node_from_group(node_id: str, group_id: str, request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        try:
            runtime.bundle.meta_store.remove_node_from_group(node_id, group_id)
            groups = runtime.bundle.meta_store.list_node_groups(node_id)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return {"node_id": node_id, "groups": [_to_dict(group) for group in groups]}

    @api.patch("/nodes/{node_id}")
    def update_node(node_id: str, payload: UpdateNodeRequest, request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        title = payload.title.strip() if payload.title is not None else None
        if title is not None and not title:
            raise HTTPException(status_code=400, detail="Node title must not be empty.")
        if title is None and payload.body is None:
            raise HTTPException(
                status_code=400,
                detail="At least one of 'title' or 'body' must be provided.",
            )
        try:
            runtime.bundle.meta_store.update_node(node_id, title=title, body=payload.body)
            node = runtime.bundle.meta_store.get_node(node_id, include_deleted=True)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        if node is None:
            raise HTTPException(status_code=500, detail="Node update failed.")
        return _to_dict(node)

    @api.get("/nodes/{node_id}/revisions")
    def list_revisions(node_id: str, request: Request) -> list[dict[str, Any]]:
        runtime = _runtime_from_request(request)
        try:
            revisions = runtime.bundle.meta_store.list_revisions(node_id)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return [_to_dict(revision) for revision in revisions]

    @api.get("/nodes/{node_id}/revisions/diff")
    def diff_node_revisions(
        node_id: str,
        request: Request,
        from_revision_id: str = Query(min_length=1),
        to_revision_id: str = Query(min_length=1),
    ) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        try:
            payload = runtime.bundle.meta_store.diff_revisions(
                node_id=node_id,
                from_revision_id=from_revision_id,
                to_revision_id=to_revision_id,
            )
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return _to_dict(payload)

    @api.post("/nodes/{node_id}/revisions/{revision_id}/rollback")
    def rollback_node_revision(node_id: str, revision_id: str, request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        try:
            new_revision_id = runtime.bundle.meta_store.rollback_node_to_revision(
                node_id=node_id,
                revision_id=revision_id,
            )
            node = runtime.bundle.meta_store.get_node(node_id, include_deleted=True)
            revisions = runtime.bundle.meta_store.list_revisions(node_id)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        if node is None:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
        return {
            "node": _to_dict(node),
            "current_revision_id": new_revision_id,
            "revisions": [_to_dict(revision) for revision in revisions],
        }

    @api.post("/nodes/{node_id}/edges/mark-stale")
    def mark_node_edges_stale(node_id: str, request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        try:
            changed = runtime.bundle.meta_store.mark_node_edges_stale(node_id)
            node = runtime.bundle.meta_store.get_node(node_id)
            graph_id = str(node["graph_id"]) if node is not None else None
            edges = runtime.bundle.meta_store.list_edges(
                graph_id=graph_id,
                node_id=node_id,
                include_deleted=False,
                limit=500,
            ) if graph_id is not None else []
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return {
            "node_id": node_id,
            "changed_edges": int(changed),
            "edges": [_to_dict(edge) for edge in edges],
        }

    @api.get("/nodes/{node_id}/content-items")
    def list_content_items(
        node_id: str,
        request: Request,
        include_deleted: bool = QUERY_INCLUDE_DELETED,
    ) -> list[dict[str, Any]]:
        runtime = _runtime_from_request(request)
        try:
            return [
                _to_dict(item)
                for item in runtime.bundle.meta_store.list_content_items(
                    node_id,
                    include_deleted=include_deleted,
                )
            ]
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)

    @api.post("/nodes/{node_id}/content-items", status_code=201)
    async def attach_content_item(
        node_id: str,
        request: Request,
        file: UploadFile = UPLOAD_FILE,
        ingest_now: bool = FORM_INGEST_NOW,
        replay_vector_ops: bool = FORM_REPLAY_VECTOR_OPS,
    ) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        mime_type = file.content_type or "application/octet-stream"
        filename = file.filename or "upload.bin"
        try:
            blob_ref = runtime.bundle.blob_store.put(payload, content_type=mime_type)
            content_item_id = runtime.bundle.meta_store.attach_content_item(
                node_id=node_id,
                blob_ref=blob_ref,
                mime_type=mime_type,
                filename=filename,
            )
            replay_stats: dict[str, Any] | None = None
            if ingest_now:
                runtime.ingestion.ingest_content_item_now(content_item_id)
                if replay_vector_ops:
                    replay_stats = _replay_stats_payload(
                        _replay_vector_index(bundle=runtime.bundle, limit=10_000)
                    )
            content_item = runtime.bundle.meta_store.get_content_item(
                content_item_id,
                include_deleted=True,
            )
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        if content_item is None:
            raise HTTPException(status_code=500, detail="Content item creation failed.")
        return {
            "content_item": _to_dict(content_item),
            "ingested": ingest_now,
            "vector_replay": replay_stats,
        }

    @api.post("/content-items/{content_item_id}/ingest")
    def ingest_content_item(
        content_item_id: str,
        payload: IngestRequest,
        request: Request,
    ) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        try:
            runtime.ingestion.ingest_content_item_now(content_item_id)
            content_item = runtime.bundle.meta_store.get_content_item(
                content_item_id,
                include_deleted=True,
            )
            replay_stats = (
                _replay_stats_payload(_replay_vector_index(bundle=runtime.bundle, limit=10_000))
                if payload.replay_vector_ops
                else None
            )
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        if content_item is None:
            raise HTTPException(
                status_code=404,
                detail=f"Content item not found: {content_item_id}",
            )
        return {
            "content_item": _to_dict(content_item),
            "vector_replay": replay_stats,
        }

    @api.get("/content-items/{content_item_id}/chunks")
    def list_content_item_chunks(
        content_item_id: str,
        request: Request,
        include_deleted: bool = QUERY_INCLUDE_DELETED,
    ) -> list[dict[str, Any]]:
        runtime = _runtime_from_request(request)
        try:
            return [
                _to_dict(chunk)
                for chunk in runtime.bundle.meta_store.list_chunks(
                    content_item_id,
                    include_deleted=include_deleted,
                )
            ]
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)

    @api.get("/chunks/{chunk_id}")
    def get_chunk(
        chunk_id: str,
        request: Request,
        include_deleted: bool = QUERY_INCLUDE_DELETED,
    ) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        chunk = runtime.bundle.meta_store.get_chunk(chunk_id, include_deleted=include_deleted)
        if chunk is None:
            raise HTTPException(status_code=404, detail=f"Chunk not found: {chunk_id}")
        node = runtime.bundle.meta_store.get_node(
            str(chunk["node_id"]),
            include_deleted=include_deleted,
        )
        content_item = runtime.bundle.meta_store.get_content_item(
            str(chunk["content_item_id"]),
            include_deleted=include_deleted,
        )
        return {
            "chunk": {
                **_to_dict(chunk),
                "text_preview": _preview_text(str(chunk.get("text", ""))),
            },
            "node": _node_summary(node),
            "content_item": _content_item_summary(content_item),
        }

    @api.post("/jobs/process-next")
    def process_next_job(request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        job_id = runtime.ingestion.process_next()
        if job_id is None:
            return {"processed": False, "job": None}
        job = runtime.bundle.job_queue.get_job(job_id)
        replay_stats = _replay_stats_payload(
            _replay_vector_index(bundle=runtime.bundle, limit=10_000)
        )
        return {
            "processed": True,
            "job": _to_dict(job) if job is not None else None,
            "vector_replay": replay_stats,
        }

    @api.get("/jobs/{job_id}")
    def get_job(job_id: str, request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        job = runtime.bundle.job_queue.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return _to_dict(job)

    @api.get("/edges/{edge_id}")
    def get_edge(
        edge_id: str,
        request: Request,
        include_deleted: bool = QUERY_INCLUDE_DELETED,
    ) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        edge = runtime.bundle.meta_store.get_edge(edge_id=edge_id, include_deleted=include_deleted)
        if edge is None:
            raise HTTPException(status_code=404, detail=f"Edge not found: {edge_id}")
        return _edge_payload(edge)

    @api.patch("/edges/{edge_id}/association")
    def patch_association_edge(
        edge_id: str,
        payload: UpdateAssociationEdgeRequest,
        request: Request,
    ) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        has_changes = (
            payload.status is not None
            or payload.weight is not None
            or payload.note is not None
            or payload.clear_note
        )
        if not has_changes:
            raise HTTPException(status_code=400, detail="At least one field must be provided.")
        edge = runtime.bundle.meta_store.get_edge(edge_id=edge_id, include_deleted=True)
        if edge is None or edge.get("deleted_at") is not None:
            raise HTTPException(status_code=404, detail=f"Edge not found: {edge_id}")
        if str(edge.get("edge_type")) != "association":
            raise HTTPException(
                status_code=400,
                detail="Association patch endpoint supports only association edges.",
            )
        next_provenance: dict[str, Any] | None = None
        if payload.note is not None or payload.clear_note:
            next_provenance = _mapping_to_dict(edge.get("provenance"))
            if payload.clear_note:
                next_provenance.pop("note", None)
            if payload.note is not None:
                note = payload.note.strip()
                if note:
                    next_provenance["note"] = note
                else:
                    next_provenance.pop("note", None)
        try:
            runtime.bundle.meta_store.update_edge(
                edge_id=edge_id,
                status=(payload.status.strip().lower() if payload.status is not None else None),
                weight=payload.weight,
                provenance=next_provenance,
            )
            updated = runtime.bundle.meta_store.get_edge(edge_id=edge_id, include_deleted=True)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Edge not found: {edge_id}")
        return _edge_payload(updated)

    @api.delete("/edges/{edge_id}/association")
    def delete_association_edge(
        edge_id: str,
        request: Request,
        soft_delete: bool = Query(default=True),
    ) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        edge = runtime.bundle.meta_store.get_edge(edge_id=edge_id, include_deleted=True)
        if edge is None:
            raise HTTPException(status_code=404, detail=f"Edge not found: {edge_id}")
        if str(edge.get("edge_type")) != "association":
            raise HTTPException(
                status_code=400,
                detail="Association delete endpoint supports only association edges.",
            )
        try:
            runtime.bundle.meta_store.delete_edge(edge_id=edge_id, soft_delete=soft_delete)
            deleted_edge = runtime.bundle.meta_store.get_edge(edge_id=edge_id, include_deleted=True)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return {
            "edge_id": edge_id,
            "deleted": True,
            "soft_delete": soft_delete,
            "edge": _edge_payload(deleted_edge) if deleted_edge is not None else None,
        }

    @api.patch("/edges/{edge_id}")
    def patch_edge(
        edge_id: str,
        payload: EdgeStatusRequest,
        request: Request,
    ) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        try:
            runtime.bundle.meta_store.update_edge(
                edge_id=edge_id,
                status=payload.status.strip().lower(),
            )
            edge = runtime.bundle.meta_store.get_edge(edge_id, include_deleted=True)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        if edge is None:
            raise HTTPException(status_code=404, detail=f"Edge not found: {edge_id}")
        return _to_dict(edge)

    @api.post("/edges/{edge_id}/accept")
    def accept_edge(edge_id: str, request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        try:
            runtime.bundle.meta_store.update_edge(edge_id=edge_id, status="accepted")
            edge = runtime.bundle.meta_store.get_edge(edge_id, include_deleted=True)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        if edge is None:
            raise HTTPException(status_code=404, detail=f"Edge not found: {edge_id}")
        return _to_dict(edge)

    @api.post("/edges/{edge_id}/reject")
    def reject_edge(edge_id: str, request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        try:
            runtime.bundle.meta_store.update_edge(edge_id=edge_id, status="rejected")
            edge = runtime.bundle.meta_store.get_edge(edge_id, include_deleted=True)
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        if edge is None:
            raise HTTPException(status_code=404, detail=f"Edge not found: {edge_id}")
        return _to_dict(edge)

    @api.get("/search")
    def search_nodes(
        request: Request,
        q: str = QUERY_SEARCH_TEXT,
        graph_id: str | None = QUERY_GRAPH_ID,
        group_id: str | None = QUERY_GROUP_ID,
        tag_ids: list[str] | None = QUERY_TAG_IDS,
        limit: int = QUERY_SEARCH_LIMIT,
    ) -> list[dict[str, Any]]:
        runtime = _runtime_from_request(request)
        try:
            rows = runtime.bundle.meta_store.search_fulltext(
                q,
                graph_id=graph_id,
                group_id=group_id,
                tag_ids=tag_ids,
                limit=limit,
            )
        except Exception as error:  # noqa: BLE001
            _raise_http_error(error)
        return [_to_dict(row) for row in rows]

    @api.post("/vector/query")
    def vector_query(payload: VectorQueryRequest, request: Request) -> dict[str, Any]:
        runtime = _runtime_from_request(request)
        vectors = runtime.bundle.embedding_provider.embed_text([payload.query])
        if not vectors:
            return {
                "query": payload.query,
                "model_id": runtime.bundle.embedding_provider.model_id(),
                "results": [],
            }
        results = runtime.bundle.vector_index.query(vectors[0], top_k=payload.top_k)
        enriched_results: list[dict[str, Any]] = []
        for result in results:
            chunk_id = result.external_id
            chunk = runtime.bundle.meta_store.get_chunk(chunk_id)
            if chunk is None:
                enriched_results.append(
                    {
                        "external_id": chunk_id,
                        "chunk_id": chunk_id,
                        "score": float(result.score),
                        "chunk": None,
                        "node": None,
                        "content_item": None,
                    }
                )
                continue
            node = runtime.bundle.meta_store.get_node(str(chunk["node_id"]))
            content_item = runtime.bundle.meta_store.get_content_item(str(chunk["content_item_id"]))
            enriched_results.append(
                {
                    "external_id": chunk_id,
                    "chunk_id": chunk_id,
                    "score": float(result.score),
                    "chunk": {
                        **_to_dict(chunk),
                        "text_preview": _preview_text(str(chunk.get("text", ""))),
                    },
                    "node": _node_summary(node),
                    "content_item": _content_item_summary(content_item),
                }
            )
        return {
            "query": payload.query,
            "model_id": runtime.bundle.embedding_provider.model_id(),
            "results": enriched_results,
        }

    @api.post("/vector/replay")
    def replay_vector_ops(
        request: Request,
        limit: int = QUERY_REPLAY_LIMIT,
    ) -> dict[str, int]:
        runtime = _runtime_from_request(request)
        stats = _replay_vector_index(bundle=runtime.bundle, limit=limit)
        return _replay_stats_payload(stats)

    app.include_router(api)

    frontend_dist = _frontend_dist_path()
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    else:
        @app.get("/")
        def root() -> dict[str, str]:
            return {
                "message": (
                    "Smart Journal API is running. Frontend build was not found at ui/dist."
                )
            }

    return app


def _runtime_from_request(request: Request) -> AppRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if not isinstance(runtime, AppRuntime):
        raise HTTPException(status_code=500, detail="Application runtime is not initialized.")
    return runtime


def _raise_http_error(error: Exception) -> NoReturn:
    if isinstance(error, HTTPException):
        raise error
    if isinstance(error, KeyError):
        raise HTTPException(status_code=404, detail=_error_text(error)) from error
    if isinstance(error, ValueError):
        raise HTTPException(status_code=400, detail=_error_text(error)) from error
    raise HTTPException(status_code=500, detail=_error_text(error)) from error


def _error_text(error: Exception) -> str:
    if error.args:
        return str(error.args[0])
    return str(error)


def _to_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in payload.items()}


def _mapping_to_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    return {str(key): value for key, value in payload.items()}


def _count_by_key(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if not normalized:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def _edge_payload(edge: Mapping[str, Any]) -> dict[str, Any]:
    payload = _to_dict(edge)
    payload["provenance"] = _mapping_to_dict(payload.get("provenance"))
    return payload


def _node_relationship_payload(*, edge: Mapping[str, Any], node_id: str) -> dict[str, Any]:
    payload = _edge_payload(edge)
    from_node_id = str(payload.get("from_node_id", ""))
    to_node_id = str(payload.get("to_node_id", ""))
    if from_node_id == node_id and to_node_id == node_id:
        direction = "self"
        other_node_id = node_id
    elif from_node_id == node_id:
        direction = "outgoing"
        other_node_id = to_node_id
    else:
        direction = "incoming"
        other_node_id = from_node_id
    payload["direction"] = direction
    payload["other_node_id"] = other_node_id
    return payload


def _selected_providers_payload(bundle: ComponentBundle) -> dict[str, dict[str, Any]]:
    return {
        "blob_store": _provider_payload(bundle.blob_store),
        "meta_store": _provider_payload(bundle.meta_store),
        "vector_index": _provider_payload(bundle.vector_index),
        "job_queue": _provider_payload(bundle.job_queue),
        "extractor": _provider_payload(bundle.extractor),
        "embedding_provider": _provider_payload(bundle.embedding_provider),
        "llm_provider": _provider_payload(bundle.llm_provider),
    }


def _provider_payload(provider: ProviderInfo) -> dict[str, Any]:
    return {
        "provider_id": provider.provider_id(),
        "version": provider.version(),
        "capabilities": dict(provider.capabilities()),
    }


def _node_summary(node: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if node is None:
        return None
    return {
        "node_id": str(node.get("node_id")),
        "graph_id": str(node.get("graph_id")),
        "title": str(node.get("title")),
        "updated_at": node.get("updated_at"),
    }


def _content_item_summary(content_item: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if content_item is None:
        return None
    return {
        "content_item_id": str(content_item.get("content_item_id")),
        "node_id": str(content_item.get("node_id")),
        "filename": content_item.get("filename"),
        "mime_type": content_item.get("mime_type"),
        "extraction_status": content_item.get("extraction_status"),
    }


def _preview_text(text: str, *, limit: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(1, limit - 3)].rstrip() + "..."


def _replay_vector_index(*, bundle: ComponentBundle, limit: int) -> VectorIndexReplayStats:
    replayer = VectorIndexOpsReplayer(
        meta_store=bundle.meta_store,
        vector_index=bundle.vector_index,
        model_id=bundle.embedding_provider.model_id(),
    )
    return replayer.replay_pending(limit=limit)


def _replay_stats_payload(stats: VectorIndexReplayStats) -> dict[str, int]:
    return {
        "pending_ops": stats.pending_ops,
        "applied_ops": stats.applied_ops,
        "upserted_vectors": stats.upserted_vectors,
        "deleted_vectors": stats.deleted_vectors,
    }


def _preload_embedding_provider(provider: Any) -> EmbeddingPreloadStatus:
    enabled = _read_bool_env("SMART_JOURNAL_PRELOAD_EMBEDDER", default=True)
    strict = _read_bool_env("SMART_JOURNAL_PRELOAD_EMBEDDER_STRICT", default=False)
    if not enabled:
        return EmbeddingPreloadStatus(
            enabled=False,
            ready=False,
            strict=strict,
            elapsed_ms=None,
            error=None,
        )

    started = time.perf_counter()
    try:
        provider.embed_text(["startup warmup"])
    except Exception as error:  # noqa: BLE001
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if strict:
            raise RuntimeError("Embedding provider warmup failed during startup.") from error
        return EmbeddingPreloadStatus(
            enabled=True,
            ready=False,
            strict=False,
            elapsed_ms=elapsed_ms,
            error=_error_text(error),
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return EmbeddingPreloadStatus(
        enabled=True,
        ready=True,
        strict=strict,
        elapsed_ms=elapsed_ms,
        error=None,
    )


def _embedding_preload_payload(status: EmbeddingPreloadStatus) -> dict[str, Any]:
    return {
        "enabled": status.enabled,
        "ready": status.ready,
        "strict": status.strict,
        "elapsed_ms": status.elapsed_ms,
        "error": status.error,
    }


def _rebuild_index_if_needed(
    *,
    bundle: ComponentBundle,
    replay_stats: VectorIndexReplayStats,
) -> VectorBootstrapRebuildStatus:
    capabilities = bundle.vector_index.capabilities()
    durable_backend = bool(capabilities.get("durable", False))
    has_index_artifact = _has_index_artifact(capabilities)
    enabled = replay_stats.applied_ops == 0 and ((not durable_backend) or (not has_index_artifact))
    if not enabled:
        return VectorBootstrapRebuildStatus(
            enabled=False,
            durable_backend=durable_backend,
            replay_applied_ops=replay_stats.applied_ops,
            scanned_chunks=0,
            upserted_vectors=0,
            missing_embeddings=0,
        )

    stats = rebuild_vector_index_from_embeddings(
        meta_store=bundle.meta_store,
        vector_index=bundle.vector_index,
        model_id=bundle.embedding_provider.model_id(),
    )
    return VectorBootstrapRebuildStatus(
        enabled=True,
        durable_backend=False,
        replay_applied_ops=replay_stats.applied_ops,
        scanned_chunks=stats.scanned_chunks,
        upserted_vectors=stats.upserted_vectors,
        missing_embeddings=stats.missing_embeddings,
    )


def _vector_rebuild_payload(status: VectorBootstrapRebuildStatus) -> dict[str, Any]:
    return {
        "enabled": status.enabled,
        "durable_backend": status.durable_backend,
        "replay_applied_ops": status.replay_applied_ops,
        "scanned_chunks": status.scanned_chunks,
        "upserted_vectors": status.upserted_vectors,
        "missing_embeddings": status.missing_embeddings,
    }


def _has_index_artifact(capabilities: Mapping[str, Any]) -> bool:
    index_file = capabilities.get("index_file")
    if not isinstance(index_file, str) or not index_file.strip():
        return True
    return Path(index_file).exists()


def _resolve_config_path(config_path: Path | None) -> Path | None:
    if config_path is not None:
        return config_path
    env_path = os.getenv("SMART_JOURNAL_CONFIG", "").strip()
    if not env_path:
        return None
    return Path(env_path)


def _cors_origins() -> list[str]:
    raw = os.getenv("SMART_JOURNAL_CORS_ORIGINS", "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]


def _read_bool_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "on"}


def _close_bundle_resources(bundle: ComponentBundle) -> None:
    providers: Sequence[ProviderInfo] = (
        bundle.blob_store,
        bundle.meta_store,
        bundle.vector_index,
        bundle.job_queue,
        bundle.extractor,
        bundle.embedding_provider,
        bundle.llm_provider,
    )
    for provider in providers:
        closer = getattr(provider, "close", None)
        if callable(closer):
            closer()


def _frontend_dist_path() -> Path:
    return Path(__file__).resolve().parents[3] / "ui" / "dist"

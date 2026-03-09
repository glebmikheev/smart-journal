from __future__ import annotations

import os
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
from smart_journal.factories import ComponentBundle, ComponentFactory
from smart_journal.ingestion import IngestionPipeline, build_default_ingestion_pipeline
from smart_journal.registry import ProviderRegistry, build_default_registry
from smart_journal.vector_ops import VectorIndexOpsReplayer, VectorIndexReplayStats

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


@dataclass(slots=True)
class AppRuntime:
    registry: ProviderRegistry
    bundle: ComponentBundle
    ingestion: IngestionPipeline


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

        app.state.runtime = AppRuntime(
            registry=registry,
            bundle=bundle,
            ingestion=ingestion,
        )
        app.state.vector_replay_bootstrap = _replay_stats_payload(replay_stats)
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
        return {
            "query": payload.query,
            "model_id": runtime.bundle.embedding_provider.model_id(),
            "results": [
                {"external_id": result.external_id, "score": float(result.score)}
                for result in results
            ],
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

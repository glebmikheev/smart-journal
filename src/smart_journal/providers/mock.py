from __future__ import annotations

import hashlib
import json
import math
import re
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
        return "0.2.0"

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
        self._revision_content_manifest: dict[str, list[str]] = {}
        self._content_items: dict[str, dict[str, Any]] = {}
        self._chunks_by_content_item: dict[str, list[dict[str, Any]]] = {}
        self._chunk_embeddings: dict[tuple[str, str], dict[str, Any]] = {}
        self._vector_index_ops: dict[str, dict[str, Any]] = {}
        self._edges: dict[str, dict[str, Any]] = {}
        self._tags: dict[str, dict[str, Any]] = {}
        self._groups: dict[str, dict[str, Any]] = {}
        self._node_tags: dict[str, set[str]] = {}
        self._node_groups: dict[str, set[str]] = {}

    def provider_id(self) -> str:
        return "in_memory"

    def version(self) -> str:
        return "0.6.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "transactions": False,
            "fulltext_backend": "naive",
            "durable": False,
            "schema_version": 5,
            "supports_scope_filters": True,
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

    def list_graphs(self, *, include_deleted: bool = False) -> list[Mapping[str, Any]]:
        graphs: list[Mapping[str, Any]] = [
            dict(graph)
            for graph in self._graphs.values()
            if include_deleted or graph["deleted_at"] is None
        ]
        graphs.sort(key=lambda item: str(item["created_at"]))
        return graphs

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
        self._capture_revision_manifest(node_id=node_id, revision_id=revision_id)
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
        self._capture_revision_manifest(node_id=node_id, revision_id=revision_id)
        self.mark_node_edges_stale(node_id)

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
        removed_revisions = self._revisions_by_node.pop(node_id, [])
        for revision in removed_revisions:
            self._revision_content_manifest.pop(str(revision["revision_id"]), None)
        self._node_tags.pop(node_id, None)
        self._node_groups.pop(node_id, None)
        for edge_id, edge in list(self._edges.items()):
            if edge["from_node_id"] == node_id or edge["to_node_id"] == node_id:
                self._edges.pop(edge_id, None)
        for content_item_id, item in list(self._content_items.items()):
            if item["node_id"] == node_id:
                self._content_items.pop(content_item_id, None)
                removed_chunks = self._chunks_by_content_item.pop(content_item_id, [])
                self._delete_embeddings_for_chunks(
                    [str(chunk["chunk_id"]) for chunk in removed_chunks]
                )

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

    def get_revision_manifest(
        self,
        node_id: str,
        revision_id: str,
    ) -> list[Mapping[str, Any]]:
        _ = self._require_revision(node_id=node_id, revision_id=revision_id)
        content_item_ids = self._revision_content_manifest.get(revision_id, [])
        payload: list[Mapping[str, Any]] = []
        for position, content_item_id in enumerate(content_item_ids):
            item = self._content_items.get(content_item_id)
            if item is None:
                continue
            payload.append(
                {
                    "content_item_id": content_item_id,
                    "position": position,
                    "filename": item.get("filename"),
                    "mime_type": item.get("mime_type"),
                    "blob_hash": item.get("blob_hash"),
                    "blob_size": item.get("blob_size"),
                }
            )
        return payload

    def diff_revisions(
        self,
        node_id: str,
        from_revision_id: str,
        to_revision_id: str,
    ) -> Mapping[str, Any]:
        from_revision = self._require_revision(node_id=node_id, revision_id=from_revision_id)
        to_revision = self._require_revision(node_id=node_id, revision_id=to_revision_id)
        source_manifest = self.get_revision_manifest(node_id, from_revision_id)
        target_manifest = self.get_revision_manifest(node_id, to_revision_id)
        source_ids = {str(row["content_item_id"]) for row in source_manifest}
        target_ids = {str(row["content_item_id"]) for row in target_manifest}
        return {
            "node_id": node_id,
            "from_revision_id": from_revision_id,
            "to_revision_id": to_revision_id,
            "title_changed": str(from_revision["title"]) != str(to_revision["title"]),
            "body_changed": str(from_revision["body"]) != str(to_revision["body"]),
            "added_content_item_ids": sorted(target_ids - source_ids),
            "removed_content_item_ids": sorted(source_ids - target_ids),
        }

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
            "extraction_status": "pending",
            "extracted_text": "",
            "extraction_error": None,
            "created_at": _utc_now(),
            "deleted_at": None,
        }
        self._chunks_by_content_item.setdefault(content_item_id, [])
        self.mark_node_edges_stale(node_id)
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

    def get_content_item(
        self, content_item_id: str, *, include_deleted: bool = False
    ) -> Mapping[str, Any] | None:
        item = self._content_items.get(content_item_id)
        if item is None:
            return None
        if not include_deleted and item["deleted_at"] is not None:
            return None
        return dict(item)

    def detach_content_item(self, content_item_id: str, *, soft_delete: bool = True) -> None:
        item = self._content_items.get(content_item_id)
        if item is None:
            return
        if soft_delete:
            deleted_at = _utc_now()
            item["deleted_at"] = deleted_at
            for chunk in self._chunks_by_content_item.get(content_item_id, []):
                if chunk["deleted_at"] is None:
                    chunk["deleted_at"] = deleted_at
            self.mark_node_edges_stale(str(item["node_id"]))
            return
        self._content_items.pop(content_item_id, None)
        removed_chunks = self._chunks_by_content_item.pop(content_item_id, [])
        self._delete_embeddings_for_chunks([str(chunk["chunk_id"]) for chunk in removed_chunks])
        self.mark_node_edges_stale(str(item["node_id"]))

    def set_content_item_extraction(
        self,
        content_item_id: str,
        *,
        status: str,
        extracted_text: str | None = None,
        metadata: Mapping[str, str] | None = None,
        error: str | None = None,
    ) -> None:
        _ = metadata
        item = self._content_items.get(content_item_id)
        if item is None or item["deleted_at"] is not None:
            raise KeyError(f"Content item not found or deleted: {content_item_id}")
        item["extraction_status"] = status
        if extracted_text is not None:
            item["extracted_text"] = extracted_text
        item["extraction_error"] = error
        item["updated_at"] = _utc_now()

    def replace_content_item_chunks(
        self,
        content_item_id: str,
        chunks: Sequence[Mapping[str, str | int]],
    ) -> list[str]:
        item = self._content_items.get(content_item_id)
        if item is None or item["deleted_at"] is not None:
            raise KeyError(f"Content item not found or deleted: {content_item_id}")
        created_chunk_ids: list[str] = []
        existing_rows = self._chunks_by_content_item.get(content_item_id, [])
        new_rows: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_id = str(uuid4())
            created_chunk_ids.append(chunk_id)
            new_rows.append(
                {
                    "chunk_id": chunk_id,
                    "content_item_id": content_item_id,
                    "node_id": item["node_id"],
                    "chunk_index": int(chunk["chunk_index"]),
                    "text": str(chunk["text"]),
                    "checksum": str(chunk["checksum"]),
                    "created_at": _utc_now(),
                    "deleted_at": None,
                }
            )
        self._delete_embeddings_for_chunks([str(row["chunk_id"]) for row in existing_rows])
        self._chunks_by_content_item[content_item_id] = new_rows
        self.mark_node_edges_stale(str(item["node_id"]))
        return created_chunk_ids

    def list_chunks(
        self, content_item_id: str, *, include_deleted: bool = False
    ) -> list[Mapping[str, Any]]:
        chunks = self._chunks_by_content_item.get(content_item_id, [])
        payload: list[Mapping[str, Any]] = [
            dict(chunk)
            for chunk in chunks
            if include_deleted or chunk["deleted_at"] is None
        ]
        payload.sort(key=lambda chunk: int(chunk["chunk_index"]))
        return payload

    def get_chunk(
        self,
        chunk_id: str,
        *,
        include_deleted: bool = False,
    ) -> Mapping[str, Any] | None:
        for chunks in self._chunks_by_content_item.values():
            for chunk in chunks:
                if str(chunk["chunk_id"]) != chunk_id:
                    continue
                if not include_deleted and chunk["deleted_at"] is not None:
                    return None
                return dict(chunk)
        return None

    def upsert_chunk_embeddings(self, embeddings: Sequence[Mapping[str, Any]]) -> None:
        for embedding in embeddings:
            chunk_id = str(embedding["chunk_id"])
            model_id = str(embedding["model_id"])
            dim = int(embedding["dim"])
            metric = str(embedding.get("metric", "cosine"))
            vector = _coerce_vector(embedding.get("vector"))
            if len(vector) != dim:
                raise ValueError(
                    f"Vector dim mismatch for chunk_id={chunk_id}: "
                    f"expected {dim}, got {len(vector)}."
                )
            chunk = self._get_live_chunk(chunk_id)
            checksum_raw = embedding.get("checksum")
            checksum = str(chunk["checksum"]) if checksum_raw is None else str(checksum_raw)
            if checksum != str(chunk["checksum"]):
                raise ValueError(
                    "Embedding checksum does not match chunk checksum "
                    f"for chunk_id={chunk_id}."
                )
            self._chunk_embeddings[(chunk_id, model_id)] = {
                "chunk_id": chunk_id,
                "model_id": model_id,
                "dim": dim,
                "metric": metric,
                "vector": vector,
                "checksum": checksum,
                "created_at": _utc_now(),
            }

    def list_chunk_embeddings(
        self,
        content_item_id: str,
        *,
        model_id: str | None = None,
    ) -> list[Mapping[str, Any]]:
        chunk_by_id = {
            str(chunk["chunk_id"]): chunk
            for chunk in self._chunks_by_content_item.get(content_item_id, [])
            if chunk["deleted_at"] is None
        }
        payload: list[dict[str, Any]] = []
        for row in self._chunk_embeddings.values():
            chunk_id = str(row["chunk_id"])
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            if model_id is not None and str(row["model_id"]) != model_id:
                continue
            payload.append(
                {
                    "chunk_id": chunk_id,
                    "model_id": str(row["model_id"]),
                    "dim": int(row["dim"]),
                    "metric": str(row["metric"]),
                    "vector": [float(value) for value in row["vector"]],
                    "checksum": str(row["checksum"]),
                    "created_at": str(row["created_at"]),
                    "_chunk_index": int(chunk["chunk_index"]),
                }
            )
        payload.sort(key=lambda row: (int(row["_chunk_index"]), str(row["model_id"])))
        result: list[Mapping[str, Any]] = [
            {
                "chunk_id": str(row["chunk_id"]),
                "model_id": str(row["model_id"]),
                "dim": int(row["dim"]),
                "metric": str(row["metric"]),
                "vector": [float(value) for value in row["vector"]],
                "checksum": str(row["checksum"]),
                "created_at": str(row["created_at"]),
            }
            for row in payload
        ]
        return result

    def get_chunk_embedding(
        self,
        chunk_id: str,
        model_id: str,
    ) -> Mapping[str, Any] | None:
        row = self._chunk_embeddings.get((chunk_id, model_id))
        if row is None:
            return None
        try:
            chunk = self._get_live_chunk(chunk_id)
        except KeyError:
            return None
        return {
            "chunk_id": str(row["chunk_id"]),
            "model_id": str(row["model_id"]),
            "dim": int(row["dim"]),
            "metric": str(row["metric"]),
            "vector": [float(value) for value in row["vector"]],
            "checksum": str(chunk["checksum"]),
            "created_at": str(row["created_at"]),
        }

    def enqueue_vector_index_ops(
        self,
        ops: Sequence[Mapping[str, str]],
    ) -> list[str]:
        created_ids: list[str] = []
        for op in ops:
            op_type = str(op["op_type"])
            if op_type not in {"upsert", "delete"}:
                raise ValueError(f"Unsupported vector index op_type: {op_type}")
            op_id = str(uuid4())
            created_ids.append(op_id)
            timestamp = _utc_now()
            self._vector_index_ops[op_id] = {
                "op_id": op_id,
                "op_type": op_type,
                "chunk_id": str(op["chunk_id"]),
                "model_id": str(op["model_id"]),
                "status": "pending",
                "created_at": timestamp,
                "applied_at": None,
            }
        return created_ids

    def list_vector_index_ops(
        self,
        *,
        status: str = "pending",
        model_id: str | None = None,
        limit: int = 1000,
    ) -> list[Mapping[str, Any]]:
        if limit <= 0:
            return []
        rows = [
            dict(row)
            for row in self._vector_index_ops.values()
            if str(row["status"]) == status
            and (model_id is None or str(row["model_id"]) == model_id)
        ]
        rows.sort(key=lambda row: (str(row["created_at"]), str(row["op_id"])))
        result: list[Mapping[str, Any]] = []
        for row in rows[:limit]:
            result.append(row)
        return result

    def mark_vector_index_ops_applied(self, op_ids: Sequence[str]) -> None:
        if not op_ids:
            return
        timestamp = _utc_now()
        for op_id in op_ids:
            row = self._vector_index_ops.get(op_id)
            if row is None:
                continue
            row["status"] = "applied"
            row["applied_at"] = timestamp

    def create_edge(
        self,
        *,
        graph_id: str,
        from_node_id: str,
        to_node_id: str,
        edge_type: str,
        status: str = "pending",
        weight: float | None = None,
        subtype: str | None = None,
        provenance: Mapping[str, Any] | None = None,
        created_by: str | None = None,
    ) -> str:
        if from_node_id == to_node_id:
            raise ValueError("Self-loop edges are not supported in alpha.")
        graph = self.get_graph(graph_id)
        if graph is None:
            raise KeyError(f"Graph not found or deleted: {graph_id}")
        from_node = self.get_node(from_node_id)
        to_node = self.get_node(to_node_id)
        if from_node is None:
            raise KeyError(f"Node not found or deleted: {from_node_id}")
        if to_node is None:
            raise KeyError(f"Node not found or deleted: {to_node_id}")
        if str(from_node["graph_id"]) != graph_id or str(to_node["graph_id"]) != graph_id:
            raise ValueError("Edge endpoints must belong to the same graph.")
        next_status = _validate_edge_status(status)
        edge_id = str(uuid4())
        timestamp = _utc_now()
        self._edges[edge_id] = {
            "edge_id": edge_id,
            "graph_id": graph_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "edge_type": edge_type,
            "subtype": (subtype.strip() if subtype is not None and subtype.strip() else None),
            "status": next_status,
            "weight": (float(weight) if weight is not None else None),
            "provenance": _normalize_provenance(provenance),
            "created_by": (
                created_by.strip() if created_by is not None and created_by.strip() else None
            ),
            "created_at": timestamp,
            "updated_at": timestamp,
            "deleted_at": None,
        }
        return edge_id

    def get_edge(self, edge_id: str, *, include_deleted: bool = False) -> Mapping[str, Any] | None:
        edge = self._edges.get(edge_id)
        if edge is None:
            return None
        if not include_deleted and edge["deleted_at"] is not None:
            return None
        return dict(edge)

    def list_edges(
        self,
        *,
        graph_id: str | None = None,
        node_id: str | None = None,
        edge_type: str | None = None,
        status: str | None = None,
        include_deleted: bool = False,
        limit: int = 200,
    ) -> list[Mapping[str, Any]]:
        if limit <= 0:
            return []
        rows: list[dict[str, Any]] = []
        for edge in self._edges.values():
            if not include_deleted and edge["deleted_at"] is not None:
                continue
            if graph_id is not None and str(edge["graph_id"]) != graph_id:
                continue
            if node_id is not None:
                if str(edge["from_node_id"]) != node_id and str(edge["to_node_id"]) != node_id:
                    continue
            if edge_type is not None and str(edge["edge_type"]) != edge_type:
                continue
            if status is not None and str(edge["status"]) != status:
                continue
            rows.append(dict(edge))
        rows.sort(key=lambda row: (str(row["updated_at"]), str(row["edge_id"])), reverse=True)
        return [dict(row) for row in rows[:limit]]

    def update_edge(
        self,
        edge_id: str,
        *,
        status: str | None = None,
        weight: float | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        edge = self._edges.get(edge_id)
        if edge is None or edge["deleted_at"] is not None:
            raise KeyError(f"Edge not found or deleted: {edge_id}")
        if status is not None:
            edge["status"] = _validate_edge_status(status)
        if weight is not None:
            edge["weight"] = float(weight)
        if provenance is not None:
            edge["provenance"] = _normalize_provenance(provenance)
        edge["updated_at"] = _utc_now()

    def mark_node_edges_stale(self, node_id: str) -> int:
        node = self.get_node(node_id)
        if node is None:
            raise KeyError(f"Node not found or deleted: {node_id}")
        changed_rows = 0
        timestamp = _utc_now()
        for edge in self._edges.values():
            if edge["deleted_at"] is not None:
                continue
            if str(edge["from_node_id"]) != node_id and str(edge["to_node_id"]) != node_id:
                continue
            current_status = str(edge["status"])
            if current_status == "rejected":
                continue
            edge_type = str(edge["edge_type"])
            next_status = "possibly_stale" if edge_type == "association" else "stale"
            if current_status == next_status:
                continue
            edge["status"] = next_status
            edge["updated_at"] = timestamp
            changed_rows += 1
        return changed_rows

    def rollback_node_to_revision(self, node_id: str, revision_id: str) -> str:
        node = self._nodes.get(node_id)
        if node is None or node["deleted_at"] is not None:
            raise KeyError(f"Node not found or deleted: {node_id}")
        revisions = self._revisions_by_node.get(node_id, [])
        source_revision = next(
            (row for row in revisions if str(row["revision_id"]) == revision_id),
            None,
        )
        if source_revision is None:
            raise KeyError(f"Revision not found: {revision_id}")

        timestamp = _utc_now()
        node["title"] = str(source_revision["title"])
        node["body"] = str(source_revision["body"])
        node["updated_at"] = timestamp

        next_revision_id = str(uuid4())
        revision_row = {
            "revision_id": next_revision_id,
            "node_id": node_id,
            "revision_no": len(revisions) + 1,
            "title": str(source_revision["title"]),
            "body": str(source_revision["body"]),
            "created_at": timestamp,
            "comment": f"rollback:{revision_id}",
        }
        revisions.append(revision_row)
        node["current_revision_id"] = next_revision_id
        self._capture_revision_manifest(node_id=node_id, revision_id=next_revision_id)
        self.mark_node_edges_stale(node_id)
        return next_revision_id

    def create_tag(self, graph_id: str, name: str) -> str:
        if self.get_graph(graph_id) is None:
            raise KeyError(f"Graph not found or deleted: {graph_id}")
        for tag in self._tags.values():
            if tag["graph_id"] == graph_id and tag["name"] == name and tag["deleted_at"] is None:
                raise ValueError(f"Tag already exists in graph '{graph_id}': {name}")
        tag_id = str(uuid4())
        self._tags[tag_id] = {
            "tag_id": tag_id,
            "graph_id": graph_id,
            "name": name,
            "created_at": _utc_now(),
            "deleted_at": None,
        }
        return tag_id

    def list_tags(self, graph_id: str, *, include_deleted: bool = False) -> list[Mapping[str, Any]]:
        tags: list[Mapping[str, Any]] = [
            dict(tag)
            for tag in self._tags.values()
            if tag["graph_id"] == graph_id and (include_deleted or tag["deleted_at"] is None)
        ]
        tags.sort(key=lambda item: str(item["name"]).lower())
        return tags

    def delete_tag(self, tag_id: str, *, soft_delete: bool = True) -> None:
        tag = self._tags.get(tag_id)
        if tag is None:
            return
        if soft_delete:
            tag["deleted_at"] = _utc_now()
            return
        self._tags.pop(tag_id, None)
        for tags in self._node_tags.values():
            tags.discard(tag_id)

    def add_node_tag(self, node_id: str, tag_id: str) -> None:
        node = self.get_node(node_id)
        if node is None:
            raise KeyError(f"Node not found or deleted: {node_id}")
        tag = self._require_live_tag(tag_id)
        if str(node["graph_id"]) != str(tag["graph_id"]):
            raise ValueError("Node and tag must belong to the same graph.")
        self._node_tags.setdefault(node_id, set()).add(tag_id)

    def remove_node_tag(self, node_id: str, tag_id: str) -> None:
        self._node_tags.get(node_id, set()).discard(tag_id)

    def list_node_tags(self, node_id: str) -> list[Mapping[str, Any]]:
        tag_ids = self._node_tags.get(node_id, set())
        tags: list[Mapping[str, Any]] = []
        for tag_id in sorted(tag_ids):
            tag = self._tags.get(tag_id)
            if tag is None or tag["deleted_at"] is not None:
                continue
            tags.append(dict(tag))
        return tags

    def create_group(self, graph_id: str, name: str) -> str:
        if self.get_graph(graph_id) is None:
            raise KeyError(f"Graph not found or deleted: {graph_id}")
        for group in self._groups.values():
            if (
                group["graph_id"] == graph_id
                and group["name"] == name
                and group["deleted_at"] is None
            ):
                raise ValueError(f"Group already exists in graph '{graph_id}': {name}")
        group_id = str(uuid4())
        self._groups[group_id] = {
            "group_id": group_id,
            "graph_id": graph_id,
            "name": name,
            "created_at": _utc_now(),
            "deleted_at": None,
        }
        return group_id

    def list_groups(
        self, graph_id: str, *, include_deleted: bool = False
    ) -> list[Mapping[str, Any]]:
        groups: list[Mapping[str, Any]] = [
            dict(group)
            for group in self._groups.values()
            if group["graph_id"] == graph_id and (include_deleted or group["deleted_at"] is None)
        ]
        groups.sort(key=lambda item: str(item["name"]).lower())
        return groups

    def delete_group(self, group_id: str, *, soft_delete: bool = True) -> None:
        group = self._groups.get(group_id)
        if group is None:
            return
        if soft_delete:
            group["deleted_at"] = _utc_now()
            return
        self._groups.pop(group_id, None)
        for groups in self._node_groups.values():
            groups.discard(group_id)

    def add_node_to_group(self, node_id: str, group_id: str) -> None:
        node = self.get_node(node_id)
        if node is None:
            raise KeyError(f"Node not found or deleted: {node_id}")
        group = self._require_live_group(group_id)
        if str(node["graph_id"]) != str(group["graph_id"]):
            raise ValueError("Node and group must belong to the same graph.")
        self._node_groups.setdefault(node_id, set()).add(group_id)

    def remove_node_from_group(self, node_id: str, group_id: str) -> None:
        self._node_groups.get(node_id, set()).discard(group_id)

    def list_node_groups(self, node_id: str) -> list[Mapping[str, Any]]:
        group_ids = self._node_groups.get(node_id, set())
        groups: list[Mapping[str, Any]] = []
        for group_id in sorted(group_ids):
            group = self._groups.get(group_id)
            if group is None or group["deleted_at"] is not None:
                continue
            groups.append(dict(group))
        return groups

    def search_fulltext(
        self,
        query: str,
        *,
        graph_id: str | None = None,
        group_id: str | None = None,
        tag_ids: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[Mapping[str, Any]]:
        normalized = query.strip().lower()
        if not normalized or limit <= 0:
            return []
        search_tokens = [token for token in normalized.split() if token]
        tag_filter = set(tag_ids or [])

        scored: list[tuple[int, dict[str, Any]]] = []
        for raw_node in self._nodes.values():
            if raw_node["deleted_at"] is not None:
                continue
            if graph_id is not None and str(raw_node["graph_id"]) != graph_id:
                continue
            node_id = str(raw_node["node_id"])
            if group_id is not None and group_id not in self._node_groups.get(node_id, set()):
                continue
            if tag_filter and not tag_filter.issubset(self._node_tags.get(node_id, set())):
                continue

            content_texts = [
                str(item.get("extracted_text", ""))
                for item in self._content_items.values()
                if item["node_id"] == node_id
                and item["deleted_at"] is None
                and item["extraction_status"] == "done"
            ]
            haystack = " ".join(
                [str(raw_node["title"]), str(raw_node["body"]), " ".join(content_texts)]
            ).lower()
            if not all(token in haystack for token in search_tokens):
                continue

            score = sum(haystack.count(token) for token in search_tokens)
            node_payload = dict(raw_node)
            node_payload["score"] = float(score)
            scored.append((score, node_payload))

        scored.sort(key=lambda item: (item[0], str(item[1]["updated_at"])), reverse=True)
        return [payload for _, payload in scored[:limit]]

    def _require_live_tag(self, tag_id: str) -> Mapping[str, Any]:
        tag = self._tags.get(tag_id)
        if tag is None or tag["deleted_at"] is not None:
            raise KeyError(f"Tag not found or deleted: {tag_id}")
        return tag

    def _require_live_group(self, group_id: str) -> Mapping[str, Any]:
        group = self._groups.get(group_id)
        if group is None or group["deleted_at"] is not None:
            raise KeyError(f"Group not found or deleted: {group_id}")
        return group

    def _get_live_chunk(self, chunk_id: str) -> Mapping[str, Any]:
        for chunks in self._chunks_by_content_item.values():
            for chunk in chunks:
                if str(chunk["chunk_id"]) != chunk_id:
                    continue
                if chunk["deleted_at"] is not None:
                    raise KeyError(f"Chunk not found or deleted: {chunk_id}")
                return chunk
        raise KeyError(f"Chunk not found or deleted: {chunk_id}")

    def _delete_embeddings_for_chunks(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        chunk_id_set = {str(chunk_id) for chunk_id in chunk_ids}
        for key in list(self._chunk_embeddings):
            if key[0] in chunk_id_set:
                self._chunk_embeddings.pop(key, None)

    def _capture_revision_manifest(self, *, node_id: str, revision_id: str) -> None:
        rows = [
            item
            for item in self._content_items.values()
            if str(item["node_id"]) == node_id and item["deleted_at"] is None
        ]
        rows.sort(key=lambda item: (str(item["created_at"]), str(item["content_item_id"])))
        self._revision_content_manifest[revision_id] = [
            str(item["content_item_id"])
            for item in rows
        ]

    def _require_revision(self, *, node_id: str, revision_id: str) -> Mapping[str, Any]:
        revisions = self._revisions_by_node.get(node_id, [])
        for revision in revisions:
            if str(revision["revision_id"]) == revision_id:
                return dict(revision)
        raise KeyError(f"Revision not found: {revision_id}")


class InMemoryVectorIndex:
    def __init__(self, _: Mapping[str, Any] | None = None) -> None:
        self._vectors: dict[str, list[float]] = {}

    def provider_id(self) -> str:
        return "in_memory"

    def version(self) -> str:
        return "0.2.0"

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
        self._jobs: dict[str, dict[str, Any]] = {}
        self._queued_ids: deque[str] = deque()

    def provider_id(self) -> str:
        return "in_process"

    def version(self) -> str:
        return "0.2.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "durable": False,
            "priority_levels": 1,
            "statuses": ["queued", "running", "completed", "failed"],
        }

    def enqueue(self, job_name: str, payload: Mapping[str, Any]) -> str:
        job_id = str(uuid4())
        timestamp = _utc_now()
        self._jobs[job_id] = {
            "job_id": job_id,
            "job_name": job_name,
            "payload": dict(payload),
            "status": "queued",
            "error": None,
            "created_at": timestamp,
            "updated_at": timestamp,
            "started_at": None,
            "finished_at": None,
        }
        self._queued_ids.append(job_id)
        return job_id

    def run_next(self) -> str | None:
        if not self._queued_ids:
            return None
        job_id = self._queued_ids.popleft()
        job = self._jobs[job_id]
        timestamp = _utc_now()
        job["status"] = "running"
        job["started_at"] = timestamp
        job["updated_at"] = timestamp
        return job_id

    def set_status(self, job_id: str, status: str, *, error: str | None = None) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown job_id: {job_id}")
        timestamp = _utc_now()
        job["status"] = status
        job["error"] = error
        job["updated_at"] = timestamp
        if status in {"completed", "failed"}:
            job["finished_at"] = timestamp

    def get_job(self, job_id: str) -> Mapping[str, Any] | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return dict(job)


class PlainTextExtractor:
    def __init__(self, _: Mapping[str, Any] | None = None) -> None:
        self._supported_mime_types = {"text/plain", "text/markdown"}

    def provider_id(self) -> str:
        return "plain_text"

    def version(self) -> str:
        return "0.2.0"

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


class BasicExtractorV1:
    def __init__(self, _: Mapping[str, Any] | None = None) -> None:
        self._text_mimes = {"text/plain", "text/markdown"}
        self._pdf_mimes = {"application/pdf"}
        self._image_mimes = {"image/png", "image/jpeg", "image/webp"}
        self._audio_mimes = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/ogg"}
        self._video_mimes = {"video/mp4", "video/webm", "video/quicktime"}
        self._supported_mime_types = (
            self._text_mimes
            | self._pdf_mimes
            | self._image_mimes
            | self._audio_mimes
            | self._video_mimes
        )

    def provider_id(self) -> str:
        return "basic_v1"

    def version(self) -> str:
        return "0.2.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "mime_types": sorted(
                self._supported_mime_types
            ),
            "outputs": ["text", "thumbnail", "metadata"],
        }

    def supports_mime(self, mime_type: str) -> bool:
        return mime_type in self._supported_mime_types

    def extract(self, content: bytes, *, mime_type: str) -> ExtractedArtifact:
        if mime_type in self._text_mimes:
            text = content.decode("utf-8", errors="replace")
            return ExtractedArtifact(
                content_type="text/plain",
                text=text,
                metadata={"source_mime_type": mime_type},
            )
        if mime_type in self._pdf_mimes:
            text = _extract_pdf_text(content)
            return ExtractedArtifact(
                content_type="text/plain",
                text=text,
                metadata={"source_mime_type": mime_type},
            )
        if mime_type in self._image_mimes:
            digest = hashlib.sha256(content).hexdigest()
            return ExtractedArtifact(
                content_type="image/thumbnail",
                text=None,
                metadata={
                    "source_mime_type": mime_type,
                    "thumbnail_checksum": digest,
                    "source_size_bytes": str(len(content)),
                },
            )
        if mime_type in self._audio_mimes or mime_type in self._video_mimes:
            media_type = "audio" if mime_type in self._audio_mimes else "video"
            return ExtractedArtifact(
                content_type=f"{media_type}/metadata",
                text=None,
                metadata={
                    "source_mime_type": mime_type,
                    "media_type": media_type,
                    "source_size_bytes": str(len(content)),
                },
            )
        raise ValueError(f"Unsupported MIME type: {mime_type}")


class MockEmbeddingProvider:
    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        options = options or {}
        self._dim = int(options.get("dim", 8))
        self._normalize = bool(options.get("normalize", True))

    def provider_id(self) -> str:
        return "mock_text"

    def version(self) -> str:
        return "0.2.0"

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
        return "0.2.0"

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


def _coerce_vector(raw: Any) -> list[float]:
    if isinstance(raw, Sequence) and not isinstance(raw, str | bytes | bytearray):
        return [float(value) for value in raw]
    raise TypeError("Embedding vector must be a sequence of floats.")


def _validate_edge_status(status: str) -> str:
    normalized = status.strip().lower()
    allowed = {"pending", "accepted", "rejected", "stale", "possibly_stale"}
    if normalized not in allowed:
        known = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported edge status '{status}'. Known statuses: {known}")
    return normalized


def _normalize_provenance(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    raw = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
    loaded = json.loads(raw)
    if isinstance(loaded, dict):
        return {str(key): value for key, value in loaded.items()}
    return {}


def _cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if not lhs or not rhs or len(lhs) != len(rhs):
        return 0.0
    dot = sum(a * b for a, b in zip(lhs, rhs, strict=False))
    lhs_norm = math.sqrt(sum(a * a for a in lhs))
    rhs_norm = math.sqrt(sum(b * b for b in rhs))
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 0.0
    return dot / (lhs_norm * rhs_norm)


def _extract_pdf_text(payload: bytes) -> str:
    text_parts: list[str] = []
    for literal in re.findall(rb"\((.*?)\)\s*Tj", payload, flags=re.S):
        text_parts.append(_decode_pdf_literal(literal))
    for array_payload in re.findall(rb"\[(.*?)\]\s*TJ", payload, flags=re.S):
        for literal in re.findall(rb"\((.*?)\)", array_payload, flags=re.S):
            text_parts.append(_decode_pdf_literal(literal))
    if not text_parts:
        decoded = payload.decode("latin-1", errors="ignore")
        text_parts.extend(re.findall(r"\(([^()]*)\)", decoded))
    return " ".join(part.strip() for part in text_parts if part.strip())


def _decode_pdf_literal(raw_literal: bytes) -> str:
    text = raw_literal.decode("latin-1", errors="ignore")
    text = text.replace(r"\n", "\n").replace(r"\r", "\r").replace(r"\t", "\t")
    text = text.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
    return text


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")

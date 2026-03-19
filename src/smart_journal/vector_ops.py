from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from smart_journal.contracts import MetaStore, VectorIndex


@dataclass(frozen=True, slots=True)
class VectorIndexReplayStats:
    pending_ops: int
    applied_ops: int
    upserted_vectors: int
    deleted_vectors: int


@dataclass(frozen=True, slots=True)
class VectorIndexRebuildStats:
    scanned_chunks: int
    upserted_vectors: int
    missing_embeddings: int


class VectorIndexOpsReplayer:
    def __init__(
        self,
        *,
        meta_store: MetaStore,
        vector_index: VectorIndex,
        model_id: str,
    ) -> None:
        self._meta_store = meta_store
        self._vector_index = vector_index
        self._model_id = model_id

    def replay_pending(self, *, limit: int = 1000) -> VectorIndexReplayStats:
        pending_ops = self._meta_store.list_vector_index_ops(
            status="pending",
            model_id=self._model_id,
            limit=limit,
        )
        if not pending_ops:
            return VectorIndexReplayStats(
                pending_ops=0,
                applied_ops=0,
                upserted_vectors=0,
                deleted_vectors=0,
            )

        latest_by_chunk: dict[str, str] = {}
        op_ids: list[str] = []
        for op in pending_ops:
            op_id = str(op["op_id"])
            op_type = str(op["op_type"])
            if op_type not in {"upsert", "delete"}:
                op_ids.append(op_id)
                continue
            latest_by_chunk[str(op["chunk_id"])] = op_type
            op_ids.append(op_id)

        upserts: list[tuple[str, list[float]]] = []
        deletes: list[str] = []
        for chunk_id, op_type in latest_by_chunk.items():
            if op_type == "delete":
                deletes.append(chunk_id)
                continue

            embedding = self._meta_store.get_chunk_embedding(chunk_id, self._model_id)
            if embedding is None:
                deletes.append(chunk_id)
                continue
            upserts.append((chunk_id, _coerce_vector(embedding)))

        if upserts:
            self._vector_index.upsert(upserts)
        if deletes:
            self._vector_index.delete(deletes)
        if upserts or deletes:
            self._vector_index.save()

        self._meta_store.mark_vector_index_ops_applied(op_ids)
        return VectorIndexReplayStats(
            pending_ops=len(pending_ops),
            applied_ops=len(op_ids),
            upserted_vectors=len(upserts),
            deleted_vectors=len(deletes),
        )


def rebuild_vector_index_from_embeddings(
    *,
    meta_store: MetaStore,
    vector_index: VectorIndex,
    model_id: str,
    batch_size: int = 256,
) -> VectorIndexRebuildStats:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    pending_vectors: list[tuple[str, list[float]]] = []
    scanned_chunks = 0
    upserted_vectors = 0
    missing_embeddings = 0

    graphs = meta_store.list_graphs()
    for graph in graphs:
        graph_id = str(graph["graph_id"])
        for node in meta_store.list_nodes(graph_id):
            node_id = str(node["node_id"])
            for content_item in meta_store.list_content_items(node_id):
                content_item_id = str(content_item["content_item_id"])
                for chunk in meta_store.list_chunks(content_item_id):
                    scanned_chunks += 1
                    chunk_id = str(chunk["chunk_id"])
                    embedding = meta_store.get_chunk_embedding(chunk_id, model_id)
                    if embedding is None:
                        missing_embeddings += 1
                        continue
                    try:
                        vector = _coerce_vector(embedding)
                    except TypeError:
                        missing_embeddings += 1
                        continue
                    pending_vectors.append((chunk_id, vector))
                    if len(pending_vectors) >= batch_size:
                        vector_index.upsert(pending_vectors)
                        upserted_vectors += len(pending_vectors)
                        pending_vectors.clear()

    if pending_vectors:
        vector_index.upsert(pending_vectors)
        upserted_vectors += len(pending_vectors)
        pending_vectors.clear()

    if upserted_vectors > 0:
        vector_index.save()

    return VectorIndexRebuildStats(
        scanned_chunks=scanned_chunks,
        upserted_vectors=upserted_vectors,
        missing_embeddings=missing_embeddings,
    )


def _coerce_vector(raw_embedding: Mapping[str, Any]) -> list[float]:
    vector_raw = raw_embedding.get("vector")
    if isinstance(vector_raw, Sequence) and not isinstance(vector_raw, str | bytes | bytearray):
        return [float(value) for value in vector_raw]
    raise TypeError("Embedding row does not contain a valid vector payload.")

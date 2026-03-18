from __future__ import annotations

from dataclasses import dataclass

from smart_journal.contracts import MetaStore, VectorIndex


@dataclass(frozen=True, slots=True)
class SemanticSuggestion:
    edge_id: str
    from_node_id: str
    to_node_id: str
    status: str
    weight: float
    supporting_chunk_hits: int


@dataclass(frozen=True, slots=True)
class _CandidateScore:
    weight: float
    supporting_chunk_hits: int


@dataclass(frozen=True, slots=True)
class _EdgeRef:
    edge_id: str
    status: str
    from_node_id: str
    to_node_id: str


class SemanticLinker:
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

    def suggest_for_node(
        self,
        node_id: str,
        *,
        top_k_per_chunk: int = 10,
        max_suggestions: int = 10,
    ) -> list[SemanticSuggestion]:
        if top_k_per_chunk <= 0 or max_suggestions <= 0:
            return []

        source_node = self._meta_store.get_node(node_id)
        if source_node is None:
            raise KeyError(f"Node not found or deleted: {node_id}")
        graph_id = str(source_node["graph_id"])

        source_embeddings = self._source_chunk_embeddings(node_id)
        if not source_embeddings:
            return []

        candidates: dict[str, _CandidateScore] = {}
        for _, vector in source_embeddings:
            results = self._vector_index.query(vector, top_k=top_k_per_chunk)
            for result in results:
                chunk = self._meta_store.get_chunk(result.external_id)
                if chunk is None:
                    continue
                target_node_id = str(chunk["node_id"])
                if target_node_id == node_id:
                    continue
                target_node = self._meta_store.get_node(target_node_id)
                if target_node is None:
                    continue
                if str(target_node["graph_id"]) != graph_id:
                    continue

                result_score = float(result.score)
                existing = candidates.get(target_node_id)
                if existing is None:
                    candidates[target_node_id] = _CandidateScore(
                        weight=result_score,
                        supporting_chunk_hits=1,
                    )
                    continue
                candidates[target_node_id] = _CandidateScore(
                    weight=max(existing.weight, result_score),
                    supporting_chunk_hits=existing.supporting_chunk_hits + 1,
                )

        if not candidates:
            return []

        existing_edges = self._existing_semantic_edges(graph_id=graph_id, node_id=node_id)
        ranked_candidates = sorted(
            candidates.items(),
            key=lambda item: (item[1].weight, item[1].supporting_chunk_hits, item[0]),
            reverse=True,
        )

        suggestions: list[SemanticSuggestion] = []
        for target_node_id, candidate_score in ranked_candidates:
            if len(suggestions) >= max_suggestions:
                break
            existing_edge = existing_edges.get(target_node_id)
            if existing_edge is not None and existing_edge.status == "rejected":
                continue

            if existing_edge is not None:
                next_status = "accepted" if existing_edge.status == "accepted" else "pending"
                self._meta_store.update_edge(
                    existing_edge.edge_id,
                    status=next_status,
                    weight=candidate_score.weight,
                )
                edge_id = existing_edge.edge_id
                from_node_id = existing_edge.from_node_id
                to_node_id = existing_edge.to_node_id
                status = next_status
            else:
                edge_id = self._meta_store.create_edge(
                    graph_id=graph_id,
                    from_node_id=node_id,
                    to_node_id=target_node_id,
                    edge_type="semantic",
                    status="pending",
                    weight=candidate_score.weight,
                )
                from_node_id = node_id
                to_node_id = target_node_id
                status = "pending"

            suggestions.append(
                SemanticSuggestion(
                    edge_id=edge_id,
                    from_node_id=from_node_id,
                    to_node_id=to_node_id,
                    status=status,
                    weight=candidate_score.weight,
                    supporting_chunk_hits=candidate_score.supporting_chunk_hits,
                )
            )
        return suggestions

    def _source_chunk_embeddings(self, node_id: str) -> list[tuple[str, list[float]]]:
        payload: list[tuple[str, list[float]]] = []
        for content_item in self._meta_store.list_content_items(node_id):
            content_item_id = str(content_item["content_item_id"])
            for chunk in self._meta_store.list_chunks(content_item_id):
                chunk_id = str(chunk["chunk_id"])
                embedding = self._meta_store.get_chunk_embedding(chunk_id, self._model_id)
                if embedding is None:
                    continue
                vector_raw = embedding.get("vector")
                if not isinstance(vector_raw, list):
                    continue
                payload.append((chunk_id, [float(value) for value in vector_raw]))
        return payload

    def _existing_semantic_edges(self, *, graph_id: str, node_id: str) -> dict[str, _EdgeRef]:
        edges = self._meta_store.list_edges(
            graph_id=graph_id,
            node_id=node_id,
            edge_type="semantic",
            include_deleted=False,
            limit=5000,
        )
        payload: dict[str, _EdgeRef] = {}
        for edge in edges:
            from_node_id = str(edge["from_node_id"])
            to_node_id = str(edge["to_node_id"])
            if from_node_id == node_id:
                key = to_node_id
            elif to_node_id == node_id:
                key = from_node_id
            else:
                continue
            if key == node_id:
                continue
            payload[key] = _EdgeRef(
                edge_id=str(edge["edge_id"]),
                status=str(edge["status"]),
                from_node_id=from_node_id,
                to_node_id=to_node_id,
            )
        return payload

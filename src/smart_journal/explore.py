from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from smart_journal.contracts import EmbeddingProvider, LLMProvider, MetaStore, VectorIndex


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    node_id: str
    content_item_id: str
    score: float
    text: str
    source: str


@dataclass(frozen=True, slots=True)
class ExploreInference:
    edge_id: str
    from_node_id: str
    to_node_id: str
    weight: float
    statement: str
    evidence_chunk_ids: tuple[str, ...]
    provenance: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExploreRunResult:
    query: str
    graph_id: str
    explore_session_id: str
    prompt_hash: str
    retrieval: tuple[RetrievedChunk, ...]
    inferences: tuple[ExploreInference, ...]
    synthesis_node_id: str | None
    llm_payload: dict[str, Any]


class ExploreService:
    def __init__(
        self,
        *,
        meta_store: MetaStore,
        vector_index: VectorIndex,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
    ) -> None:
        self._meta_store = meta_store
        self._vector_index = vector_index
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider

    def run(
        self,
        *,
        graph_id: str,
        query: str,
        group_id: str | None = None,
        top_k_chunks: int = 12,
        max_inferences: int = 5,
        create_synthesis: bool = False,
    ) -> ExploreRunResult:
        if top_k_chunks <= 0:
            raise ValueError("top_k_chunks must be greater than 0.")
        if max_inferences <= 0:
            raise ValueError("max_inferences must be greater than 0.")

        graph = self._meta_store.get_graph(graph_id)
        if graph is None:
            raise KeyError(f"Graph not found or deleted: {graph_id}")
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty.")

        explore_session_id = str(uuid4())
        retrieval = self._retrieve(
            graph_id=graph_id,
            query=normalized_query,
            group_id=group_id,
            top_k_chunks=top_k_chunks,
        )
        prompt = self._build_prompt(
            query=normalized_query,
            retrieval=retrieval,
            max_inferences=max_inferences,
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        llm_payload = self._reason_with_llm(prompt=prompt)
        inferred_rows = self._build_inferences(
            query=normalized_query,
            retrieval=retrieval,
            max_inferences=max_inferences,
            llm_payload=llm_payload,
        )
        persisted = self._persist_implication_edges(
            graph_id=graph_id,
            query=normalized_query,
            explore_session_id=explore_session_id,
            prompt_hash=prompt_hash,
            llm_payload=llm_payload,
            rows=inferred_rows,
        )
        synthesis_node_id = (
            self._create_synthesis_node(
                graph_id=graph_id,
                query=normalized_query,
                explore_session_id=explore_session_id,
                inferences=persisted,
            )
            if create_synthesis and persisted
            else None
        )
        return ExploreRunResult(
            query=normalized_query,
            graph_id=graph_id,
            explore_session_id=explore_session_id,
            prompt_hash=prompt_hash,
            retrieval=tuple(retrieval),
            inferences=tuple(persisted),
            synthesis_node_id=synthesis_node_id,
            llm_payload=llm_payload,
        )

    def _retrieve(
        self,
        *,
        graph_id: str,
        query: str,
        group_id: str | None,
        top_k_chunks: int,
    ) -> list[RetrievedChunk]:
        vector = self._embedding_provider.embed_text([query])
        candidates: list[RetrievedChunk] = []
        if vector:
            vector_results = self._vector_index.query(vector[0], top_k=max(10, top_k_chunks * 3))
            for result in vector_results:
                chunk = self._meta_store.get_chunk(result.external_id)
                if chunk is None:
                    continue
                chunk_node_id = str(chunk["node_id"])
                if not self._node_in_scope(
                    node_id=chunk_node_id,
                    graph_id=graph_id,
                    group_id=group_id,
                ):
                    continue
                candidates.append(
                    RetrievedChunk(
                        chunk_id=str(chunk["chunk_id"]),
                        node_id=chunk_node_id,
                        content_item_id=str(chunk["content_item_id"]),
                        score=float(result.score),
                        text=str(chunk["text"]),
                        source="vector",
                    )
                )

        if not candidates:
            fts_rows = self._meta_store.search_fulltext(
                query,
                graph_id=graph_id,
                group_id=group_id,
                limit=top_k_chunks,
            )
            for row in fts_rows:
                node_id = str(row["node_id"])
                first_chunk = self._first_node_chunk(node_id)
                if first_chunk is None:
                    continue
                candidates.append(
                    RetrievedChunk(
                        chunk_id=str(first_chunk["chunk_id"]),
                        node_id=node_id,
                        content_item_id=str(first_chunk["content_item_id"]),
                        score=float(row.get("score", 0.0)),
                        text=str(first_chunk["text"]),
                        source="fts",
                    )
                )

        deduped: list[RetrievedChunk] = []
        seen_chunks: set[str] = set()
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            if candidate.chunk_id in seen_chunks:
                continue
            seen_chunks.add(candidate.chunk_id)
            deduped.append(candidate)
            if len(deduped) >= top_k_chunks:
                break
        return deduped

    def _build_prompt(
        self,
        *,
        query: str,
        retrieval: list[RetrievedChunk],
        max_inferences: int,
    ) -> str:
        lines = [
            "You are ExploreService for Smart Journal.",
            "Return strict JSON with field 'implications'.",
            (
                "Each implication item must include: from_node_id, to_node_id, statement, "
                "evidence_chunk_ids, weight."
            ),
            "Only use the node/chunk ids provided below.",
            f"Query: {query}",
            f"Max implications: {max_inferences}",
            "Evidence chunks:",
        ]
        for item in retrieval[:20]:
            preview = " ".join(item.text.split())[:180]
            lines.append(
                f"- chunk={item.chunk_id} node={item.node_id} score={item.score:.3f} text={preview}"
            )
        return "\n".join(lines)

    def _reason_with_llm(self, *, prompt: str) -> dict[str, Any]:
        schema = {
            "implications": "array",
            "notes": "string",
        }
        try:
            payload = self._llm_provider.generate_structured(prompt, schema)
        except Exception:
            return {}
        if isinstance(payload, dict):
            return {str(key): value for key, value in payload.items()}
        return {}

    def _build_inferences(
        self,
        *,
        query: str,
        retrieval: list[RetrievedChunk],
        max_inferences: int,
        llm_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if len(retrieval) < 2:
            return []

        llm_inferences = self._parse_llm_inferences(
            retrieval=retrieval,
            max_inferences=max_inferences,
            llm_payload=llm_payload,
        )
        if llm_inferences:
            return llm_inferences

        best_by_node: dict[str, RetrievedChunk] = {}
        for row in retrieval:
            current = best_by_node.get(row.node_id)
            if current is None or row.score > current.score:
                best_by_node[row.node_id] = row
        ranked_nodes = sorted(best_by_node.values(), key=lambda item: item.score, reverse=True)
        if len(ranked_nodes) < 2:
            return []

        anchor = ranked_nodes[0]
        inferences: list[dict[str, Any]] = []
        for candidate in ranked_nodes[1:]:
            if len(inferences) >= max_inferences:
                break
            anchor_node = self._meta_store.get_node(anchor.node_id)
            candidate_node = self._meta_store.get_node(candidate.node_id)
            anchor_title = str(anchor_node["title"]) if anchor_node is not None else anchor.node_id
            candidate_title = (
                str(candidate_node["title"]) if candidate_node is not None else candidate.node_id
            )
            inference_weight = max(0.01, min(0.999, float(candidate.score)))
            inferences.append(
                {
                    "from_node_id": anchor.node_id,
                    "to_node_id": candidate.node_id,
                    "weight": inference_weight,
                    "statement": (
                        f"Explore query '{query}' suggests that '{anchor_title}' relates to "
                        f"'{candidate_title}'."
                    ),
                    "evidence_chunk_ids": [anchor.chunk_id, candidate.chunk_id],
                    "reasoning_source": "fallback",
                }
            )
        return inferences

    def _parse_llm_inferences(
        self,
        *,
        retrieval: list[RetrievedChunk],
        max_inferences: int,
        llm_payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        raw_items = llm_payload.get("implications")
        if not isinstance(raw_items, list):
            return []
        valid_nodes = {row.node_id for row in retrieval}
        default_evidence = self._best_chunk_by_node(retrieval)
        parsed: list[dict[str, Any]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            from_node_id = str(raw_item.get("from_node_id", "")).strip()
            to_node_id = str(raw_item.get("to_node_id", "")).strip()
            if not from_node_id or not to_node_id or from_node_id == to_node_id:
                continue
            if from_node_id not in valid_nodes or to_node_id not in valid_nodes:
                continue
            pair = (from_node_id, to_node_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            statement = str(raw_item.get("statement", "")).strip()
            if not statement:
                statement = (
                    f"LLM suggested implication between {from_node_id} and {to_node_id}."
                )
            raw_evidence = raw_item.get("evidence_chunk_ids")
            evidence: list[str] = []
            if isinstance(raw_evidence, list):
                for item in raw_evidence:
                    chunk_id = str(item).strip()
                    if chunk_id:
                        evidence.append(chunk_id)
            if not evidence:
                evidence = [
                    default_evidence.get(from_node_id, ""),
                    default_evidence.get(to_node_id, ""),
                ]
                evidence = [item for item in evidence if item]
            if not evidence:
                continue
            weight = _coerce_weight(raw_item.get("weight"), default=0.5)
            parsed.append(
                {
                    "from_node_id": from_node_id,
                    "to_node_id": to_node_id,
                    "weight": weight,
                    "statement": statement,
                    "evidence_chunk_ids": evidence,
                    "reasoning_source": "llm",
                }
            )
            if len(parsed) >= max_inferences:
                break
        return parsed

    def _persist_implication_edges(
        self,
        *,
        graph_id: str,
        query: str,
        explore_session_id: str,
        prompt_hash: str,
        llm_payload: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> list[ExploreInference]:
        persisted: list[ExploreInference] = []
        for row in rows:
            evidence = tuple(str(item) for item in row["evidence_chunk_ids"])
            provenance = {
                "query": query,
                "explore_session_id": explore_session_id,
                "prompt_hash": prompt_hash,
                "llm_model_id": self._llm_provider.model_id(),
                "reasoning_source": str(row.get("reasoning_source", "fallback")),
                "statement": str(row["statement"]),
                "evidence_chunk_ids": list(evidence),
                "llm_payload": llm_payload,
            }
            edge_id = self._meta_store.create_edge(
                graph_id=graph_id,
                from_node_id=str(row["from_node_id"]),
                to_node_id=str(row["to_node_id"]),
                edge_type="implication",
                subtype=explore_session_id,
                status="pending",
                weight=float(row["weight"]),
                provenance=provenance,
                created_by="llm",
            )
            persisted.append(
                ExploreInference(
                    edge_id=edge_id,
                    from_node_id=str(row["from_node_id"]),
                    to_node_id=str(row["to_node_id"]),
                    weight=float(row["weight"]),
                    statement=str(row["statement"]),
                    evidence_chunk_ids=evidence,
                    provenance=provenance,
                )
            )
        return persisted

    def _create_synthesis_node(
        self,
        *,
        graph_id: str,
        query: str,
        explore_session_id: str,
        inferences: list[ExploreInference],
    ) -> str:
        synthesis_title = f"Synthesis: {query[:80]}".strip()
        lines = [f"Explore query: {query}", "", "Inferences:"]
        source_nodes: set[str] = set()
        inference_edge_ids: list[str] = []
        for index, inference in enumerate(inferences, start=1):
            source_nodes.add(inference.from_node_id)
            source_nodes.add(inference.to_node_id)
            inference_edge_ids.append(inference.edge_id)
            evidence = ", ".join(inference.evidence_chunk_ids)
            lines.append(
                f"{index}. {inference.statement} (evidence chunks: {evidence})"
            )
        synthesis_body = "\n".join(lines)
        synthesis_node_id = self._meta_store.create_node(
            graph_id=graph_id,
            title=synthesis_title,
            body=synthesis_body,
        )
        for source_node_id in sorted(source_nodes):
            self._meta_store.create_edge(
                graph_id=graph_id,
                from_node_id=source_node_id,
                to_node_id=synthesis_node_id,
                edge_type="association",
                subtype=explore_session_id,
                status="accepted",
                weight=1.0,
                provenance={
                    "query": query,
                    "explore_session_id": explore_session_id,
                    "inference_edge_ids": inference_edge_ids,
                },
                created_by="system",
            )
        return synthesis_node_id

    def _node_in_scope(self, *, node_id: str, graph_id: str, group_id: str | None) -> bool:
        node = self._meta_store.get_node(node_id)
        if node is None or str(node["graph_id"]) != graph_id:
            return False
        if group_id is None:
            return True
        groups = self._meta_store.list_node_groups(node_id)
        return any(str(group["group_id"]) == group_id for group in groups)

    def _first_node_chunk(self, node_id: str) -> dict[str, Any] | None:
        for content_item in self._meta_store.list_content_items(node_id):
            content_item_id = str(content_item["content_item_id"])
            chunks = self._meta_store.list_chunks(content_item_id)
            if chunks:
                row = chunks[0]
                return {
                    "chunk_id": str(row["chunk_id"]),
                    "content_item_id": str(row["content_item_id"]),
                    "text": str(row["text"]),
                }
        return None

    def _best_chunk_by_node(self, retrieval: list[RetrievedChunk]) -> dict[str, str]:
        payload: dict[str, tuple[float, str]] = {}
        for row in retrieval:
            current = payload.get(row.node_id)
            if current is None or row.score > current[0]:
                payload[row.node_id] = (row.score, row.chunk_id)
        return {node_id: value[1] for node_id, value in payload.items()}


def _coerce_weight(raw: Any, *, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return max(0.01, min(0.999, value))

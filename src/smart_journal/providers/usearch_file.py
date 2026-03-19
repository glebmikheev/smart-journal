from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from smart_journal.contracts import VectorResult


class USearchFileVectorIndex:
    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        options = options or {}
        self._root = Path(str(options.get("root", "./data/indexes/default/mock-text-embed-v1")))
        self._metric = str(options.get("metric", "cosine")).lower()
        if self._metric not in {"cosine"}:
            raise ValueError(f"Unsupported metric for usearch_file: {self._metric}")
        self._index_path = self._root / "usearch.index"
        self._manifest_path = self._root / "manifest.json"
        self._vectors: dict[str, list[float]] = {}

    def provider_id(self) -> str:
        return "usearch_file"

    def version(self) -> str:
        return "0.6.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "supports_delete": True,
            "supports_filter": False,
            "metric_types": [self._metric],
            "durable": True,
            "index_file": str(self._index_path),
            "manifest_file": str(self._manifest_path),
        }

    def upsert(self, vectors: Sequence[tuple[str, Sequence[float]]]) -> None:
        for external_id, vector in vectors:
            self._vectors[str(external_id)] = [float(value) for value in vector]

    def delete(self, external_ids: Sequence[str]) -> None:
        for external_id in external_ids:
            self._vectors.pop(str(external_id), None)

    def query(self, vector: Sequence[float], top_k: int) -> list[VectorResult]:
        if top_k <= 0:
            return []
        query_vector = [float(value) for value in vector]
        scored = [
            VectorResult(
                external_id=external_id,
                score=_cosine_similarity(query_vector, candidate),
            )
            for external_id, candidate in self._vectors.items()
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def save(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        index_payload = {
            "metric": self._metric,
            "vectors": self._vectors,
        }
        self._index_path.write_text(
            json.dumps(index_payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        manifest_payload = {
            "provider_id": self.provider_id(),
            "version": self.version(),
            "metric": self._metric,
            "vector_count": len(self._vectors),
            "updated_at": _utc_now(),
        }
        self._manifest_path.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load(self) -> None:
        if not self._index_path.exists():
            self._vectors = {}
            return

        payload = json.loads(self._index_path.read_text(encoding="utf-8"))
        vectors_raw = payload.get("vectors")
        if not isinstance(vectors_raw, dict):
            self._vectors = {}
            return

        parsed_vectors: dict[str, list[float]] = {}
        for external_id, vector_raw in vectors_raw.items():
            if (
                not isinstance(external_id, str)
                or not isinstance(vector_raw, Sequence)
                or isinstance(vector_raw, str | bytes | bytearray)
            ):
                continue
            parsed_vectors[external_id] = [float(value) for value in vector_raw]
        self._vectors = parsed_vectors


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

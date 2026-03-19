from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, cast


class MultilingualE5SmallEmbeddingProvider:
    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        options = options or {}
        self._model_name = "intfloat/multilingual-e5-small"
        self._device = str(options.get("device", "cpu"))
        self._normalize = bool(options.get("normalize", True))
        self._batch_size = int(options.get("batch_size", 32))
        self._text_prefix = str(options.get("text_prefix", "passage")).strip()
        self._dim = 384
        self._model: Any | None = None

    def provider_id(self) -> str:
        return "multilingual_e5_small"

    def version(self) -> str:
        return "0.6.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "text": True,
            "image": False,
            "audio": False,
            "video": False,
            "dim": self._dim,
            "batch_limits": self._batch_size,
            "model_name": self._model_name,
            "default_text_prefix": self._text_prefix,
        }

    def model_id(self) -> str:
        return self._model_name

    def dim(self) -> int:
        return self._dim

    def normalize(self) -> bool:
        return self._normalize

    def embed_text(self, chunks: Sequence[str]) -> list[list[float]]:
        if not chunks:
            return []
        model = self._ensure_model()
        inputs = [_apply_text_prefix(str(chunk), self._text_prefix) for chunk in chunks]
        vectors = model.encode(
            inputs,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in vectors.tolist()]

    def embed_image(self, frames: Sequence[bytes]) -> list[list[float]]:
        if frames:
            raise NotImplementedError(
                "Image embeddings are not supported by multilingual_e5_small."
            )
        return []

    def embed_audio(self, segments: Sequence[bytes]) -> list[list[float]]:
        if segments:
            raise NotImplementedError(
                "Audio embeddings are not supported by multilingual_e5_small."
            )
        return []

    def embed_video(self, segments: Sequence[bytes]) -> list[list[float]]:
        if segments:
            raise NotImplementedError(
                "Video embeddings are not supported by multilingual_e5_small."
            )
        return []

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            module = importlib.import_module("sentence_transformers")
            sentence_transformer = cast(Any, module).SentenceTransformer
        except ImportError as error:  # pragma: no cover - depends on runtime environment
            raise RuntimeError(
                "sentence-transformers is required for multilingual_e5_small. "
                "Install it with: python -m pip install sentence-transformers"
            ) from error
        self._model = sentence_transformer(self._model_name, device=self._device)
        return self._model


def _apply_text_prefix(text: str, prefix: str) -> str:
    stripped = text.strip()
    if not prefix:
        return stripped
    return f"{prefix}: {stripped}"

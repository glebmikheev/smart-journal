from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from smart_journal.contracts import BlobInfo, BlobRef


class LocalCASBlobStore:
    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        options = options or {}
        root_value = str(options.get("root", "./data/blobs"))
        self._root = Path(root_value)

    def provider_id(self) -> str:
        return "local_cas"

    def version(self) -> str:
        return "0.2.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "content_addressed": True,
            "schemes": ["localcas"],
            "supports_delete": True,
            "supports_verify": True,
            "durable": True,
        }

    def put(self, data: bytes, *, content_type: str | None = None) -> BlobRef:
        _ = content_type
        digest = hashlib.sha256(data).hexdigest()
        key = f"sha256:{digest}"
        path = self._path_for_digest(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return BlobRef(scheme="localcas", key=key, size=len(data), hash=digest)

    def open(self, blob_ref: BlobRef) -> bytes:
        digest = _digest_from_blob_ref(blob_ref)
        path = self._path_for_digest(digest)
        return path.read_bytes()

    def stat(self, blob_ref: BlobRef) -> BlobInfo:
        digest = _digest_from_blob_ref(blob_ref)
        path = self._path_for_digest(digest)
        stat_result = path.stat()
        return BlobInfo(
            size=int(stat_result.st_size),
            hash=digest,
            modified_unix=float(stat_result.st_mtime),
        )

    def exists(self, blob_ref: BlobRef) -> bool:
        digest = _digest_from_blob_ref(blob_ref)
        path = self._path_for_digest(digest)
        return path.exists()

    def delete(self, blob_ref: BlobRef) -> None:
        digest = _digest_from_blob_ref(blob_ref)
        path = self._path_for_digest(digest)
        if path.exists():
            path.unlink()

    def verify(self, blob_ref: BlobRef) -> bool:
        digest = _digest_from_blob_ref(blob_ref)
        path = self._path_for_digest(digest)
        if not path.exists():
            return False
        data = path.read_bytes()
        computed = hashlib.sha256(data).hexdigest()
        if computed != digest:
            return False
        if blob_ref.hash and computed != blob_ref.hash:
            return False
        return True

    def _path_for_digest(self, digest: str) -> Path:
        return self._root / digest[:2] / digest[2:]


def _digest_from_blob_ref(blob_ref: BlobRef) -> str:
    if not blob_ref.key.startswith("sha256:"):
        raise ValueError(f"Unsupported blob key format: {blob_ref.key}")
    digest = blob_ref.key.split(":", maxsplit=1)[1]
    if len(digest) != 64:
        raise ValueError(f"Unexpected sha256 digest length for key: {blob_ref.key}")
    return digest

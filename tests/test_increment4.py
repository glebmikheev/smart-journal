from __future__ import annotations

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from smart_journal.contracts import BlobRef
from smart_journal.ingestion import build_default_ingestion_pipeline
from smart_journal.providers import (
    BasicExtractorV1,
    InMemoryBlobStore,
    InProcessJobQueue,
    MockEmbeddingProvider,
    SQLiteMetaStore,
)


class MutableInMemoryBlobStore(InMemoryBlobStore):
    def mutate(self, blob_ref: BlobRef, payload: bytes) -> None:
        self._blobs[blob_ref.key] = payload


class CountingEmbeddingProvider(MockEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__({"dim": 8, "normalize": True})
        self.embed_text_calls = 0
        self.embedded_chunks = 0

    def embed_text(self, chunks: Sequence[str]) -> list[list[float]]:
        self.embed_text_calls += 1
        self.embedded_chunks += len(chunks)
        return super().embed_text(chunks)


class IncrementFourAcceptanceTests(unittest.TestCase):
    def test_embeddings_recompute_only_for_changed_chunk_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta_store = SQLiteMetaStore({"path": str(Path(tmp_dir) / "meta.db")})
            try:
                blob_store = MutableInMemoryBlobStore()
                extractor = BasicExtractorV1()
                job_queue = InProcessJobQueue()
                embedding_provider = CountingEmbeddingProvider()

                graph_id = meta_store.create_graph("Increment 4 graph")
                node_id = meta_store.create_node(graph_id=graph_id, title="Doc node")

                original_text = (
                    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
                )
                updated_text = (
                    "alpha beta gamma delta epsilon zeta eta THETA iota kappa lambda mu"
                )

                blob_ref = blob_store.put(
                    original_text.encode("utf-8"),
                    content_type="text/markdown",
                )
                content_item_id = meta_store.attach_content_item(
                    node_id=node_id,
                    blob_ref=blob_ref,
                    filename="doc.md",
                    mime_type="text/markdown",
                )

                pipeline = build_default_ingestion_pipeline(
                    meta_store=meta_store,
                    blob_store=blob_store,
                    extractor=extractor,
                    job_queue=job_queue,
                    embedding_provider=embedding_provider,
                    options={"chunk_size": 22, "chunk_overlap": 0},
                )

                pipeline.ingest_content_item_now(content_item_id)
                first_chunks = meta_store.list_chunks(content_item_id)
                first_checksums = [str(row["checksum"]) for row in first_chunks]
                first_embeddings = meta_store.list_chunk_embeddings(
                    content_item_id,
                    model_id=embedding_provider.model_id(),
                )
                self.assertEqual(len(first_chunks), 4)
                self.assertEqual(len(first_embeddings), len(first_chunks))
                self.assertEqual(embedding_provider.embed_text_calls, 1)
                self.assertEqual(embedding_provider.embedded_chunks, len(first_chunks))

                blob_store.mutate(blob_ref, updated_text.encode("utf-8"))
                pipeline.ingest_content_item_now(content_item_id)

                second_chunks = meta_store.list_chunks(content_item_id)
                second_checksums = [str(row["checksum"]) for row in second_chunks]
                second_embeddings = meta_store.list_chunk_embeddings(
                    content_item_id,
                    model_id=embedding_provider.model_id(),
                )
                self.assertEqual(len(second_embeddings), len(second_chunks))
                self.assertEqual(embedding_provider.embed_text_calls, 2)
                self.assertEqual(
                    embedding_provider.embedded_chunks,
                    len(first_chunks) + 1,
                )

                changed_checksums = sum(
                    1
                    for before, after in zip(first_checksums, second_checksums, strict=True)
                    if before != after
                )
                self.assertEqual(changed_checksums, 1)
            finally:
                meta_store.close()


if __name__ == "__main__":
    unittest.main()

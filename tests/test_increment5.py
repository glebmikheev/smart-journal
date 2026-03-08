from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_journal.config import AppConfig
from smart_journal.factories import ComponentFactory
from smart_journal.ingestion import build_default_ingestion_pipeline
from smart_journal.registry import build_default_registry
from smart_journal.vector_ops import VectorIndexOpsReplayer


class IncrementFiveAcceptanceTests(unittest.TestCase):
    def test_replay_pending_ops_restores_vector_index_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = AppConfig.from_mapping(
                {
                    "blob_store": {
                        "backend": "local_cas",
                        "root": str(tmp_path / "blobs"),
                    },
                    "meta_store": {
                        "backend": "sqlite",
                        "path": str(tmp_path / "meta.db"),
                    },
                    "vector_index": {
                        "backend": "usearch_file",
                        "root": str(tmp_path / "indexes" / "main" / "mock-text-embed-v1"),
                    },
                    "embedding_provider": {
                        "backend": "mock_text",
                        "dim": 8,
                        "normalize": True,
                    },
                }
            )

            first_bundle = ComponentFactory(build_default_registry()).create(config)
            try:
                graph_id = first_bundle.meta_store.create_graph("Increment 5 graph")
                node_id = first_bundle.meta_store.create_node(graph_id=graph_id, title="Doc node")
                blob_ref = first_bundle.blob_store.put(
                    b"alpha beta gamma delta epsilon zeta eta theta iota",
                    content_type="text/markdown",
                )
                content_item_id = first_bundle.meta_store.attach_content_item(
                    node_id=node_id,
                    blob_ref=blob_ref,
                    filename="doc.md",
                    mime_type="text/markdown",
                )

                pipeline = build_default_ingestion_pipeline(
                    meta_store=first_bundle.meta_store,
                    blob_store=first_bundle.blob_store,
                    extractor=first_bundle.extractor,
                    job_queue=first_bundle.job_queue,
                    embedding_provider=first_bundle.embedding_provider,
                    options={"chunk_size": 24, "chunk_overlap": 0},
                )
                pipeline.ingest_content_item_now(content_item_id)

                pending_before_restart = first_bundle.meta_store.list_vector_index_ops(
                    status="pending",
                    model_id=first_bundle.embedding_provider.model_id(),
                )
                self.assertGreaterEqual(len(pending_before_restart), 1)

                query_vector = first_bundle.embedding_provider.embed_text(["delta epsilon"])[0]
                before_results = first_bundle.vector_index.query(query_vector, top_k=5)
                self.assertEqual(before_results, [])
            finally:
                _close_bundle(first_bundle)

            second_bundle = ComponentFactory(build_default_registry()).create(config)
            try:
                second_bundle.vector_index.load()
                replayer = VectorIndexOpsReplayer(
                    meta_store=second_bundle.meta_store,
                    vector_index=second_bundle.vector_index,
                    model_id=second_bundle.embedding_provider.model_id(),
                )
                replay_stats = replayer.replay_pending()
                self.assertGreaterEqual(replay_stats.pending_ops, 1)
                self.assertGreaterEqual(replay_stats.applied_ops, 1)

                query_vector = second_bundle.embedding_provider.embed_text(["delta epsilon"])[0]
                after_results = second_bundle.vector_index.query(query_vector, top_k=5)
                self.assertGreaterEqual(len(after_results), 1)

                chunks = second_bundle.meta_store.list_chunks(content_item_id)
                chunk_ids = {str(chunk["chunk_id"]) for chunk in chunks}
                returned_ids = {result.external_id for result in after_results}
                self.assertTrue(returned_ids.intersection(chunk_ids))

                pending_after_replay = second_bundle.meta_store.list_vector_index_ops(
                    status="pending",
                    model_id=second_bundle.embedding_provider.model_id(),
                )
                self.assertEqual(pending_after_replay, [])
            finally:
                _close_bundle(second_bundle)


def _close_bundle(bundle: object) -> None:
    providers = (
        getattr(bundle, "blob_store", None),
        getattr(bundle, "meta_store", None),
        getattr(bundle, "vector_index", None),
        getattr(bundle, "job_queue", None),
        getattr(bundle, "extractor", None),
        getattr(bundle, "embedding_provider", None),
        getattr(bundle, "llm_provider", None),
    )
    for provider in providers:
        if provider is None:
            continue
        closer = getattr(provider, "close", None)
        if callable(closer):
            closer()


if __name__ == "__main__":
    unittest.main()

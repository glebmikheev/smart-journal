from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_journal.config import AppConfig
from smart_journal.factories import ComponentFactory
from smart_journal.ingestion import build_default_ingestion_pipeline
from smart_journal.registry import build_default_registry
from smart_journal.semantic import SemanticLinker
from smart_journal.vector_ops import VectorIndexOpsReplayer


class IncrementSixAcceptanceTests(unittest.TestCase):
    def test_semantic_link_suggestions_create_pending_edges_and_preserve_rejected(self) -> None:
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
                        "backend": "in_memory",
                    },
                    "embedding_provider": {
                        "backend": "mock_text",
                        "dim": 8,
                        "normalize": True,
                    },
                }
            )
            bundle = ComponentFactory(build_default_registry()).create(config)
            try:
                graph_id = bundle.meta_store.create_graph("Increment 6 graph")
                source_node_id = bundle.meta_store.create_node(graph_id=graph_id, title="Source")
                related_node_id = bundle.meta_store.create_node(graph_id=graph_id, title="Related")
                other_node_id = bundle.meta_store.create_node(graph_id=graph_id, title="Other")

                self._ingest_text(
                    bundle=bundle,
                    node_id=source_node_id,
                    text="alpha beta gamma delta",
                )
                self._ingest_text(
                    bundle=bundle,
                    node_id=related_node_id,
                    text="alpha beta gamma delta",
                )
                self._ingest_text(bundle=bundle, node_id=other_node_id, text="kappa lambda mu nu")

                replayer = VectorIndexOpsReplayer(
                    meta_store=bundle.meta_store,
                    vector_index=bundle.vector_index,
                    model_id=bundle.embedding_provider.model_id(),
                )
                replay_stats = replayer.replay_pending()
                self.assertGreaterEqual(replay_stats.applied_ops, 3)

                linker = SemanticLinker(
                    meta_store=bundle.meta_store,
                    vector_index=bundle.vector_index,
                    model_id=bundle.embedding_provider.model_id(),
                )
                suggestions = linker.suggest_for_node(
                    source_node_id,
                    top_k_per_chunk=5,
                    max_suggestions=5,
                )
                self.assertTrue(suggestions)
                suggested_targets = {item.to_node_id for item in suggestions}
                self.assertIn(related_node_id, suggested_targets)

                semantic_edges = bundle.meta_store.list_edges(
                    graph_id=graph_id,
                    edge_type="semantic",
                    status="pending",
                    limit=20,
                )
                self.assertTrue(semantic_edges)

                edge_to_related = next(
                    (
                        edge
                        for edge in semantic_edges
                        if {
                            str(edge["from_node_id"]),
                            str(edge["to_node_id"]),
                        }
                        == {source_node_id, related_node_id}
                    ),
                    None,
                )
                self.assertIsNotNone(edge_to_related)
                assert edge_to_related is not None

                edge_id = str(edge_to_related["edge_id"])
                bundle.meta_store.update_edge(edge_id, status="rejected")
                rejected = bundle.meta_store.get_edge(edge_id)
                self.assertIsNotNone(rejected)
                assert rejected is not None
                self.assertEqual(str(rejected["status"]), "rejected")

                next_suggestions = linker.suggest_for_node(
                    source_node_id,
                    top_k_per_chunk=5,
                    max_suggestions=5,
                )
                next_targets = {item.to_node_id for item in next_suggestions}
                self.assertNotIn(related_node_id, next_targets)

                still_rejected = bundle.meta_store.get_edge(edge_id)
                self.assertIsNotNone(still_rejected)
                assert still_rejected is not None
                self.assertEqual(str(still_rejected["status"]), "rejected")
            finally:
                _close_bundle(bundle)

    def _ingest_text(self, *, bundle: object, node_id: str, text: str) -> str:
        blob_ref = bundle.blob_store.put(text.encode("utf-8"), content_type="text/markdown")
        content_item_id = bundle.meta_store.attach_content_item(
            node_id=node_id,
            blob_ref=blob_ref,
            filename="doc.md",
            mime_type="text/markdown",
        )
        pipeline = build_default_ingestion_pipeline(
            meta_store=bundle.meta_store,
            blob_store=bundle.blob_store,
            extractor=bundle.extractor,
            job_queue=bundle.job_queue,
            embedding_provider=bundle.embedding_provider,
            options={"chunk_size": 200, "chunk_overlap": 0},
        )
        pipeline.ingest_content_item_now(content_item_id)
        return content_item_id


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

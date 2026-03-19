from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_journal.ingestion import IngestionPipeline, build_default_ingestion_pipeline
from smart_journal.providers import (
    BasicExtractorV1,
    InMemoryBlobStore,
    InMemoryVectorIndex,
    InProcessJobQueue,
    MockEmbeddingProvider,
)
from smart_journal.providers.sqlite_meta import SQLiteMetaStore
from smart_journal.semantic import SemanticLinker
from smart_journal.vector_ops import VectorIndexOpsReplayer


class MutableInMemoryBlobStore(InMemoryBlobStore):
    def mutate(self, blob_key: str, payload: bytes) -> None:
        self._blobs[blob_key] = payload


class IncrementSevenAcceptanceTests(unittest.TestCase):
    def test_rollback_restores_node_snapshot_and_creates_new_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta_store = SQLiteMetaStore({"path": str(Path(tmp_dir) / "meta.db")})
            try:
                graph_id = meta_store.create_graph("Increment 7 graph")
                node_id = meta_store.create_node(
                    graph_id=graph_id,
                    title="Node",
                    body="version one",
                )
                meta_store.update_node(node_id, body="version two")
                revisions_before = meta_store.list_revisions(node_id)
                self.assertEqual(len(revisions_before), 2)
                source_revision_id = str(revisions_before[0]["revision_id"])

                new_revision_id = meta_store.rollback_node_to_revision(node_id, source_revision_id)

                node = meta_store.get_node(node_id)
                self.assertIsNotNone(node)
                assert node is not None
                self.assertEqual(str(node["body"]), "version one")
                self.assertEqual(str(node["current_revision_id"]), new_revision_id)

                revisions_after = meta_store.list_revisions(node_id)
                self.assertEqual(len(revisions_after), 3)
                self.assertEqual(
                    str(revisions_after[-1]["comment"]),
                    f"rollback:{source_revision_id}",
                )
            finally:
                meta_store.close()

    def test_mark_node_edges_stale_applies_status_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta_store = SQLiteMetaStore({"path": str(Path(tmp_dir) / "meta.db")})
            try:
                graph_id = meta_store.create_graph("Increment 7 graph")
                node_a = meta_store.create_node(graph_id=graph_id, title="Node A")
                node_b = meta_store.create_node(graph_id=graph_id, title="Node B")

                semantic_edge_id = meta_store.create_edge(
                    graph_id=graph_id,
                    from_node_id=node_a,
                    to_node_id=node_b,
                    edge_type="semantic",
                    status="accepted",
                    weight=0.91,
                )
                association_edge_id = meta_store.create_edge(
                    graph_id=graph_id,
                    from_node_id=node_a,
                    to_node_id=node_b,
                    edge_type="association",
                    status="accepted",
                    weight=0.81,
                )
                implication_edge_id = meta_store.create_edge(
                    graph_id=graph_id,
                    from_node_id=node_a,
                    to_node_id=node_b,
                    edge_type="implication",
                    status="pending",
                    weight=0.77,
                )
                rejected_edge_id = meta_store.create_edge(
                    graph_id=graph_id,
                    from_node_id=node_a,
                    to_node_id=node_b,
                    edge_type="semantic",
                    status="rejected",
                    weight=0.65,
                )

                changed_rows = meta_store.mark_node_edges_stale(node_a)
                self.assertEqual(changed_rows, 3)

                semantic_edge = meta_store.get_edge(semantic_edge_id)
                association_edge = meta_store.get_edge(association_edge_id)
                implication_edge = meta_store.get_edge(implication_edge_id)
                rejected_edge = meta_store.get_edge(rejected_edge_id)

                self.assertIsNotNone(semantic_edge)
                self.assertIsNotNone(association_edge)
                self.assertIsNotNone(implication_edge)
                self.assertIsNotNone(rejected_edge)
                assert semantic_edge is not None
                assert association_edge is not None
                assert implication_edge is not None
                assert rejected_edge is not None
                self.assertEqual(str(semantic_edge["status"]), "stale")
                self.assertEqual(str(association_edge["status"]), "possibly_stale")
                self.assertEqual(str(implication_edge["status"]), "stale")
                self.assertEqual(str(rejected_edge["status"]), "rejected")
            finally:
                meta_store.close()

    def test_targeted_recompute_refreshes_only_affected_node_semantic_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta_store = SQLiteMetaStore({"path": str(Path(tmp_dir) / "meta.db")})
            blob_store = MutableInMemoryBlobStore()
            extractor = BasicExtractorV1()
            job_queue = InProcessJobQueue()
            embedding_provider = MockEmbeddingProvider({"dim": 8, "normalize": True})
            vector_index = InMemoryVectorIndex()
            try:
                graph_id = meta_store.create_graph("Increment 7 graph")
                source_node = meta_store.create_node(graph_id=graph_id, title="Source")
                original_match = meta_store.create_node(graph_id=graph_id, title="Original match")
                new_match = meta_store.create_node(graph_id=graph_id, title="New match")

                source_item_id, source_blob_key = self._attach_text(
                    meta_store=meta_store,
                    blob_store=blob_store,
                    node_id=source_node,
                    text="alpha beta gamma delta",
                )
                _ = self._attach_text(
                    meta_store=meta_store,
                    blob_store=blob_store,
                    node_id=original_match,
                    text="alpha beta gamma delta",
                )
                _ = self._attach_text(
                    meta_store=meta_store,
                    blob_store=blob_store,
                    node_id=new_match,
                    text="kappa lambda mu",
                )

                pipeline = build_default_ingestion_pipeline(
                    meta_store=meta_store,
                    blob_store=blob_store,
                    extractor=extractor,
                    job_queue=job_queue,
                    embedding_provider=embedding_provider,
                    options={"chunk_size": 200, "chunk_overlap": 0},
                )
                self._ingest_node(meta_store=meta_store, pipeline=pipeline, node_id=source_node)
                self._ingest_node(meta_store=meta_store, pipeline=pipeline, node_id=original_match)
                self._ingest_node(meta_store=meta_store, pipeline=pipeline, node_id=new_match)

                replayer = VectorIndexOpsReplayer(
                    meta_store=meta_store,
                    vector_index=vector_index,
                    model_id=embedding_provider.model_id(),
                )
                replay_stats = replayer.replay_pending()
                self.assertGreaterEqual(replay_stats.applied_ops, 3)

                linker = SemanticLinker(
                    meta_store=meta_store,
                    vector_index=vector_index,
                    model_id=embedding_provider.model_id(),
                )
                first = linker.recompute_for_node(
                    source_node,
                    top_k_per_chunk=10,
                    max_suggestions=1,
                )
                self.assertEqual(first.stale_edge_count, 0)
                self.assertEqual(len(first.suggestions), 1)
                first_target = {
                    first.suggestions[0].from_node_id,
                    first.suggestions[0].to_node_id,
                }
                self.assertEqual(first_target, {source_node, original_match})
                meta_store.update_edge(first.suggestions[0].edge_id, status="accepted")

                blob_store.mutate(source_blob_key, b"kappa lambda mu")
                pipeline.ingest_content_item_now(source_item_id)
                replay_stats_after_update = replayer.replay_pending()
                self.assertGreaterEqual(replay_stats_after_update.applied_ops, 1)

                second = linker.recompute_for_node(
                    source_node,
                    top_k_per_chunk=10,
                    max_suggestions=1,
                )
                self.assertEqual(len(second.suggestions), 1)
                second_target = {
                    second.suggestions[0].from_node_id,
                    second.suggestions[0].to_node_id,
                }
                self.assertEqual(second_target, {source_node, new_match})
                self.assertEqual(second.stale_edge_count, 1)

                stale_edges = meta_store.list_edges(
                    graph_id=graph_id,
                    node_id=source_node,
                    edge_type="semantic",
                    status="stale",
                    limit=20,
                )
                stale_pairs = [
                    {str(edge["from_node_id"]), str(edge["to_node_id"])}
                    for edge in stale_edges
                ]
                self.assertIn({source_node, original_match}, stale_pairs)

                active_edges = meta_store.list_edges(
                    graph_id=graph_id,
                    node_id=source_node,
                    edge_type="semantic",
                    status="pending",
                    limit=20,
                )
                active_pairs = [
                    {str(edge["from_node_id"]), str(edge["to_node_id"])}
                    for edge in active_edges
                ]
                self.assertIn({source_node, new_match}, active_pairs)
            finally:
                meta_store.close()

    def _attach_text(
        self,
        *,
        meta_store: SQLiteMetaStore,
        blob_store: MutableInMemoryBlobStore,
        node_id: str,
        text: str,
    ) -> tuple[str, str]:
        blob_ref = blob_store.put(text.encode("utf-8"), content_type="text/markdown")
        content_item_id = meta_store.attach_content_item(
            node_id=node_id,
            blob_ref=blob_ref,
            filename="doc.md",
            mime_type="text/markdown",
        )
        return content_item_id, str(blob_ref.key)

    def _ingest_node(
        self,
        *,
        meta_store: SQLiteMetaStore,
        pipeline: IngestionPipeline,
        node_id: str,
    ) -> None:
        for item in meta_store.list_content_items(node_id):
            pipeline.ingest_content_item_now(str(item["content_item_id"]))


if __name__ == "__main__":
    unittest.main()

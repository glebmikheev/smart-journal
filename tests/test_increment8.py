from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_journal.explore import ExploreService
from smart_journal.ingestion import IngestionPipeline, build_default_ingestion_pipeline
from smart_journal.providers import (
    BasicExtractorV1,
    InMemoryBlobStore,
    InMemoryVectorIndex,
    InProcessJobQueue,
    MockEmbeddingProvider,
    MockLLMProvider,
)
from smart_journal.providers.sqlite_meta import SQLiteMetaStore
from smart_journal.vector_ops import VectorIndexOpsReplayer


class IncrementEightAcceptanceTests(unittest.TestCase):
    def test_explore_creates_implications_with_evidence_and_synthesis_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta_store = SQLiteMetaStore({"path": str(Path(tmp_dir) / "meta.db")})
            blob_store = InMemoryBlobStore()
            extractor = BasicExtractorV1()
            job_queue = InProcessJobQueue()
            embedding_provider = MockEmbeddingProvider({"dim": 8, "normalize": True})
            llm_provider = MockLLMProvider()
            vector_index = InMemoryVectorIndex()
            try:
                graph_id = meta_store.create_graph("Increment 8 graph")
                node_a = meta_store.create_node(graph_id=graph_id, title="Risk register A")
                node_b = meta_store.create_node(graph_id=graph_id, title="Risk register B")
                node_c = meta_store.create_node(graph_id=graph_id, title="Risk register C")

                self._attach_and_ingest(
                    meta_store=meta_store,
                    blob_store=blob_store,
                    extractor=extractor,
                    job_queue=job_queue,
                    embedding_provider=embedding_provider,
                    node_id=node_a,
                    text=(
                        "alpha milestone dependencies and delivery risks for project phoenix"
                    ),
                )
                self._attach_and_ingest(
                    meta_store=meta_store,
                    blob_store=blob_store,
                    extractor=extractor,
                    job_queue=job_queue,
                    embedding_provider=embedding_provider,
                    node_id=node_b,
                    text=(
                        "milestone alpha planning notes include risk mitigation and status reports"
                    ),
                )
                self._attach_and_ingest(
                    meta_store=meta_store,
                    blob_store=blob_store,
                    extractor=extractor,
                    job_queue=job_queue,
                    embedding_provider=embedding_provider,
                    node_id=node_c,
                    text=(
                        "budget impact and timeline risks for alpha release and integration tasks"
                    ),
                )

                replayer = VectorIndexOpsReplayer(
                    meta_store=meta_store,
                    vector_index=vector_index,
                    model_id=embedding_provider.model_id(),
                )
                replay_stats = replayer.replay_pending()
                self.assertGreaterEqual(replay_stats.applied_ops, 3)

                explore = ExploreService(
                    meta_store=meta_store,
                    vector_index=vector_index,
                    embedding_provider=embedding_provider,
                    llm_provider=llm_provider,
                )
                result = explore.run(
                    graph_id=graph_id,
                    query="alpha milestone risks and dependencies",
                    top_k_chunks=12,
                    max_inferences=5,
                    create_synthesis=True,
                )

                self.assertGreaterEqual(len(result.retrieval), 3)
                self.assertGreaterEqual(len(result.inferences), 2)
                self.assertLessEqual(len(result.inferences), 5)
                self.assertIsNotNone(result.synthesis_node_id)
                assert result.synthesis_node_id is not None
                self.assertTrue(bool(result.explore_session_id))
                self.assertEqual(len(result.prompt_hash), 64)

                synthesis_node = meta_store.get_node(result.synthesis_node_id)
                self.assertIsNotNone(synthesis_node)

                source_nodes: set[str] = set()
                for inference in result.inferences:
                    source_nodes.add(inference.from_node_id)
                    source_nodes.add(inference.to_node_id)
                    self.assertGreaterEqual(len(inference.evidence_chunk_ids), 2)
                    edge = meta_store.get_edge(inference.edge_id)
                    self.assertIsNotNone(edge)
                    assert edge is not None
                    self.assertEqual(str(edge["edge_type"]), "implication")
                    self.assertEqual(str(edge["status"]), "pending")
                    self.assertEqual(str(edge["subtype"]), result.explore_session_id)
                    self.assertEqual(str(edge["created_by"]), "llm")
                    provenance = edge.get("provenance")
                    self.assertIsInstance(provenance, dict)
                    assert isinstance(provenance, dict)
                    self.assertEqual(str(provenance.get("query")), result.query)
                    self.assertEqual(
                        str(provenance.get("explore_session_id")),
                        result.explore_session_id,
                    )
                    for chunk_id in inference.evidence_chunk_ids:
                        self.assertIsNotNone(meta_store.get_chunk(chunk_id))

                synthesis_links = meta_store.list_edges(
                    graph_id=graph_id,
                    node_id=result.synthesis_node_id,
                    edge_type="association",
                    status="accepted",
                    limit=100,
                )
                linked_sources = {
                    str(edge["from_node_id"])
                    for edge in synthesis_links
                    if str(edge["to_node_id"]) == result.synthesis_node_id
                }
                self.assertTrue(source_nodes.issubset(linked_sources))
            finally:
                meta_store.close()

    def _attach_and_ingest(
        self,
        *,
        meta_store: SQLiteMetaStore,
        blob_store: InMemoryBlobStore,
        extractor: BasicExtractorV1,
        job_queue: InProcessJobQueue,
        embedding_provider: MockEmbeddingProvider,
        node_id: str,
        text: str,
    ) -> None:
        blob_ref = blob_store.put(text.encode("utf-8"), content_type="text/markdown")
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
            options={"chunk_size": 220, "chunk_overlap": 0},
        )
        self.assertIsInstance(pipeline, IngestionPipeline)
        pipeline.ingest_content_item_now(content_item_id)


if __name__ == "__main__":
    unittest.main()

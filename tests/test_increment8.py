from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class IncrementEightMultimodalExtractorTests(unittest.TestCase):
    def test_basic_extractor_runs_ocr_and_asr_when_enabled(self) -> None:
        with (
            mock.patch(
                "smart_journal.providers.mock._run_image_ocr",
                return_value=("detected image text", {"ocr_status": "ok"}),
            ) as run_ocr,
            mock.patch(
                "smart_journal.providers.mock._run_audio_asr",
                return_value=("detected transcript", {"asr_status": "ok"}),
            ) as run_asr,
        ):
            extractor = BasicExtractorV1(
                {
                    "enable_image_ocr": True,
                    "enable_audio_asr": True,
                    "ocr_lang": "deu",
                    "asr_model": "tiny",
                    "asr_languages": ["en", "de"],
                    "asr_device": "cpu",
                }
            )
            image_payload = b"\x89PNG\r\n"
            audio_payload = b"RIFF...."
            image_artifact = extractor.extract(image_payload, mime_type="image/png")
            audio_artifact = extractor.extract(audio_payload, mime_type="audio/wav")

        self.assertEqual(image_artifact.content_type, "image/thumbnail")
        self.assertEqual(image_artifact.text, "detected image text")
        self.assertIsNotNone(image_artifact.metadata)
        assert image_artifact.metadata is not None
        self.assertEqual(image_artifact.metadata.get("ocr_status"), "ok")
        run_ocr.assert_called_once()
        ocr_call_args = run_ocr.call_args
        self.assertIsNotNone(ocr_call_args)
        assert ocr_call_args is not None
        self.assertEqual(ocr_call_args.args[0], image_payload)
        self.assertEqual(ocr_call_args.kwargs["backend"], "ppocr_v5")
        self.assertEqual(ocr_call_args.kwargs["lang"], "deu")
        self.assertEqual(ocr_call_args.kwargs["languages"], ["deu"])
        self.assertEqual(ocr_call_args.kwargs["device"], "cpu")
        self.assertEqual(ocr_call_args.kwargs["profile_name"], "mobile_optional")

        self.assertEqual(audio_artifact.content_type, "audio/metadata")
        self.assertEqual(audio_artifact.text, "detected transcript")
        self.assertIsNotNone(audio_artifact.metadata)
        assert audio_artifact.metadata is not None
        self.assertEqual(audio_artifact.metadata.get("asr_status"), "ok")
        run_asr.assert_called_once()
        call_args = run_asr.call_args
        self.assertIsNotNone(call_args)
        assert call_args is not None
        self.assertEqual(call_args.args[0], audio_payload)
        self.assertEqual(call_args.kwargs["mime_type"], "audio/wav")
        self.assertEqual(call_args.kwargs["model"], "tiny")
        self.assertEqual(call_args.kwargs["languages"], ["en", "de"])
        self.assertEqual(call_args.kwargs["device"], "cpu")

    def test_basic_extractor_skips_ocr_and_asr_when_disabled(self) -> None:
        with (
            mock.patch("smart_journal.providers.mock._run_image_ocr") as run_ocr,
            mock.patch("smart_journal.providers.mock._run_audio_asr") as run_asr,
        ):
            extractor = BasicExtractorV1(
                {
                    "enable_image_ocr": False,
                    "enable_audio_asr": False,
                }
            )
            image_artifact = extractor.extract(b"image", mime_type="image/jpeg")
            audio_artifact = extractor.extract(b"audio", mime_type="audio/mpeg")
            video_artifact = extractor.extract(b"video", mime_type="video/mp4")

        run_ocr.assert_not_called()
        run_asr.assert_not_called()

        self.assertIsNone(image_artifact.text)
        self.assertIsNotNone(image_artifact.metadata)
        assert image_artifact.metadata is not None
        self.assertEqual(image_artifact.metadata.get("ocr_status"), "disabled")

        self.assertIsNone(audio_artifact.text)
        self.assertIsNotNone(audio_artifact.metadata)
        assert audio_artifact.metadata is not None
        self.assertEqual(audio_artifact.metadata.get("asr_status"), "disabled")

        self.assertEqual(video_artifact.content_type, "video/metadata")
        self.assertIsNone(video_artifact.text)

    def test_basic_extractor_asr_capabilities_expose_model_and_language_controls(self) -> None:
        extractor = BasicExtractorV1(
            {
                "asr_model": "small",
                "asr_languages": "en,ru",
            }
        )
        capabilities = extractor.capabilities()

        self.assertEqual(capabilities.get("audio_asr_model"), "small")
        self.assertEqual(capabilities.get("audio_asr_default_model"), "small")
        self.assertEqual(capabilities.get("audio_asr_language_mode"), "hinted")
        self.assertEqual(
            capabilities.get("audio_asr_configured_languages"),
            ["en", "ru"],
        )
        supported_models = capabilities.get("audio_asr_supported_models")
        self.assertIsInstance(supported_models, list)
        assert isinstance(supported_models, list)
        self.assertIn("small", supported_models)

    def test_basic_extractor_defaults_to_small_model_with_auto_language(self) -> None:
        extractor = BasicExtractorV1()
        capabilities = extractor.capabilities()
        self.assertEqual(capabilities.get("audio_asr_model"), "small")
        self.assertEqual(capabilities.get("audio_asr_language_mode"), "auto")
        self.assertEqual(capabilities.get("audio_asr_configured_languages"), [])

    def test_basic_extractor_keeps_legacy_single_asr_language_option(self) -> None:
        extractor = BasicExtractorV1({"asr_language": "fr"})
        capabilities = extractor.capabilities()
        self.assertEqual(capabilities.get("audio_asr_configured_languages"), ["fr"])

    def test_basic_extractor_defaults_to_mobile_optional_ocr_profile(self) -> None:
        extractor = BasicExtractorV1()
        capabilities = extractor.capabilities()

        self.assertEqual(capabilities.get("ocr_backend"), "ppocr_v5")
        self.assertEqual(capabilities.get("ocr_active_profile"), "mobile_optional")
        profile_names = capabilities.get("ocr_profiles")
        self.assertIsInstance(profile_names, list)
        assert isinstance(profile_names, list)
        self.assertIn("mobile_optional", profile_names)
        self.assertIn("server_optional", profile_names)

    def test_basic_extractor_can_switch_active_ocr_profile(self) -> None:
        extractor = BasicExtractorV1()
        switched = extractor.set_active_ocr_profile("server")
        self.assertEqual(str(switched.get("profile_name")), "server")
        capabilities = extractor.capabilities()
        self.assertEqual(capabilities.get("ocr_active_profile"), "server")

    def test_basic_extractor_rejects_unknown_ocr_profile(self) -> None:
        extractor = BasicExtractorV1()
        with self.assertRaises(ValueError):
            extractor.set_active_ocr_profile("no_such_profile")

    def test_basic_extractor_rejects_non_ppocr_backend(self) -> None:
        with self.assertRaises(ValueError):
            BasicExtractorV1({"ocr_backend": "pytesseract"})


if __name__ == "__main__":
    unittest.main()

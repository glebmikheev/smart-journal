from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_journal.config import AppConfig
from smart_journal.factories import ComponentFactory
from smart_journal.ingestion import build_default_ingestion_pipeline, split_text_into_chunks
from smart_journal.registry import build_default_registry


class IncrementTwoAcceptanceTests(unittest.TestCase):
    def test_pdf_and_markdown_ingestion_creates_extracted_text_and_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bundle = ComponentFactory(build_default_registry()).create(
                AppConfig.from_mapping(
                    {
                        "blob_store": {
                            "backend": "local_cas",
                            "root": str(tmp_path / "blobs"),
                        },
                        "meta_store": {
                            "backend": "sqlite",
                            "path": str(tmp_path / "meta.db"),
                        },
                        "extractor": {
                            "backend": "basic_v1",
                        },
                    }
                )
            )
            try:
                graph_id = bundle.meta_store.create_graph("Increment 2 graph")
                node_id = bundle.meta_store.create_node(graph_id=graph_id, title="Doc node")

                markdown_bytes = b"# heading\nalpha beta gamma delta"
                pdf_bytes = _build_simple_pdf_bytes("pdf omega tokens")

                markdown_blob_ref = bundle.blob_store.put(
                    markdown_bytes,
                    content_type="text/markdown",
                )
                pdf_blob_ref = bundle.blob_store.put(pdf_bytes, content_type="application/pdf")

                markdown_item_id = bundle.meta_store.attach_content_item(
                    node_id=node_id,
                    blob_ref=markdown_blob_ref,
                    filename="doc.md",
                    mime_type="text/markdown",
                )
                pdf_item_id = bundle.meta_store.attach_content_item(
                    node_id=node_id,
                    blob_ref=pdf_blob_ref,
                    filename="doc.pdf",
                    mime_type="application/pdf",
                )

                pipeline = build_default_ingestion_pipeline(
                    meta_store=bundle.meta_store,
                    blob_store=bundle.blob_store,
                    extractor=bundle.extractor,
                    job_queue=bundle.job_queue,
                    options={"chunk_size": 16, "chunk_overlap": 4},
                )
                md_job_id = pipeline.enqueue_content_item(markdown_item_id)
                pdf_job_id = pipeline.enqueue_content_item(pdf_item_id)
                pipeline.process_next()
                pipeline.process_next()

                md_job = bundle.job_queue.get_job(md_job_id)
                pdf_job = bundle.job_queue.get_job(pdf_job_id)
                self.assertIsNotNone(md_job)
                self.assertIsNotNone(pdf_job)
                assert md_job is not None
                assert pdf_job is not None
                self.assertEqual(md_job["status"], "completed")
                self.assertEqual(pdf_job["status"], "completed")

                md_item = bundle.meta_store.get_content_item(markdown_item_id)
                pdf_item = bundle.meta_store.get_content_item(pdf_item_id)
                self.assertIsNotNone(md_item)
                self.assertIsNotNone(pdf_item)
                assert md_item is not None
                assert pdf_item is not None
                self.assertEqual(md_item["extraction_status"], "done")
                self.assertEqual(pdf_item["extraction_status"], "done")
                self.assertIn("alpha beta", str(md_item["extracted_text"]))
                self.assertIn("pdf omega tokens", str(pdf_item["extracted_text"]))

                md_chunks = bundle.meta_store.list_chunks(markdown_item_id)
                pdf_chunks = bundle.meta_store.list_chunks(pdf_item_id)
                self.assertGreaterEqual(len(md_chunks), 1)
                self.assertGreaterEqual(len(pdf_chunks), 1)
                first_md_chunk = bundle.meta_store.get_chunk(str(md_chunks[0]["chunk_id"]))
                self.assertIsNotNone(first_md_chunk)
                assert first_md_chunk is not None
                self.assertEqual(str(first_md_chunk["content_item_id"]), markdown_item_id)
                for chunk in md_chunks + pdf_chunks:
                    self.assertEqual(len(str(chunk["checksum"])), 64)
            finally:
                close_meta_store = getattr(bundle.meta_store, "close", None)
                if callable(close_meta_store):
                    close_meta_store()

    def test_split_text_into_chunks_uses_checksum(self) -> None:
        chunks = split_text_into_chunks(
            "one two three four five six seven eight",
            chunk_size=14,
            chunk_overlap=3,
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].chunk_index, 0)
        for chunk in chunks:
            self.assertEqual(len(chunk.checksum), 64)


def _build_simple_pdf_bytes(text: str) -> bytes:
    body = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    return "\n".join(
        [
            "%PDF-1.4",
            "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
            "2 0 obj << /Type /Pages /Count 1 /Kids [3 0 R] >> endobj",
            (
                "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                "/Contents 4 0 R >> endobj"
            ),
            f"4 0 obj << /Length {len(body)} >> stream",
            body,
            "endstream endobj",
            "xref",
            "0 5",
            "0000000000 65535 f ",
            "trailer << /Root 1 0 R /Size 5 >>",
            "startxref",
            "0",
            "%%EOF",
        ]
    ).encode("latin-1")


if __name__ == "__main__":
    unittest.main()

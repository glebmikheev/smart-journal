from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_journal.config import AppConfig
from smart_journal.factories import ComponentFactory
from smart_journal.registry import build_default_registry


class IncrementThreeAcceptanceTests(unittest.TestCase):
    def test_search_supports_scope_graph_group_and_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle = self._build_bundle(Path(tmp_dir))
            try:
                graph_id = bundle.meta_store.create_graph("Main graph")
                second_graph_id = bundle.meta_store.create_graph("Second graph")

                research_tag_id = bundle.meta_store.create_tag(graph_id, "research")
                work_group_id = bundle.meta_store.create_group(graph_id, "work")

                node_pdf = bundle.meta_store.create_node(
                    graph_id=graph_id,
                    title="Paper notes",
                    body="Summary",
                )
                node_md = bundle.meta_store.create_node(
                    graph_id=graph_id,
                    title="Sprint plan",
                    body="Roadmap draft",
                )
                node_other_graph = bundle.meta_store.create_node(
                    graph_id=second_graph_id,
                    title="Another paper",
                    body="Unrelated graph",
                )

                pdf_blob_ref = bundle.blob_store.put(b"pdf-bytes", content_type="application/pdf")
                md_blob_ref = bundle.blob_store.put(b"md-bytes", content_type="text/markdown")
                other_blob_ref = bundle.blob_store.put(
                    b"other-pdf-bytes",
                    content_type="application/pdf",
                )

                pdf_item_id = bundle.meta_store.attach_content_item(
                    node_id=node_pdf,
                    blob_ref=pdf_blob_ref,
                    filename="paper.pdf",
                    mime_type="application/pdf",
                )
                md_item_id = bundle.meta_store.attach_content_item(
                    node_id=node_md,
                    blob_ref=md_blob_ref,
                    filename="plan.md",
                    mime_type="text/markdown",
                )
                other_item_id = bundle.meta_store.attach_content_item(
                    node_id=node_other_graph,
                    blob_ref=other_blob_ref,
                    filename="other.pdf",
                    mime_type="application/pdf",
                )

                bundle.meta_store.set_content_item_extraction(
                    pdf_item_id,
                    status="done",
                    extracted_text="Transformer embeddings benchmark results",
                )
                bundle.meta_store.set_content_item_extraction(
                    md_item_id,
                    status="done",
                    extracted_text="Weekly tasks and roadmap updates",
                )
                bundle.meta_store.set_content_item_extraction(
                    other_item_id,
                    status="done",
                    extracted_text="Transformer embeddings in another graph",
                )

                bundle.meta_store.add_node_tag(node_pdf, research_tag_id)
                bundle.meta_store.add_node_to_group(node_pdf, work_group_id)

                all_results = bundle.meta_store.search_fulltext(
                    "transformer embeddings",
                    graph_id=graph_id,
                )
                self.assertEqual(len(all_results), 1)
                self.assertEqual(all_results[0]["node_id"], node_pdf)

                group_results = bundle.meta_store.search_fulltext(
                    "transformer embeddings",
                    graph_id=graph_id,
                    group_id=work_group_id,
                )
                self.assertEqual(len(group_results), 1)
                self.assertEqual(group_results[0]["node_id"], node_pdf)

                tagged_results = bundle.meta_store.search_fulltext(
                    "transformer embeddings",
                    graph_id=graph_id,
                    tag_ids=[research_tag_id],
                )
                self.assertEqual(len(tagged_results), 1)
                self.assertEqual(tagged_results[0]["node_id"], node_pdf)
            finally:
                close_meta_store = getattr(bundle.meta_store, "close", None)
                if callable(close_meta_store):
                    close_meta_store()

    def test_tag_group_crud_and_fts_sync_after_text_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle = self._build_bundle(Path(tmp_dir))
            try:
                graph_id = bundle.meta_store.create_graph("Knowledge graph")
                node_id = bundle.meta_store.create_node(
                    graph_id=graph_id,
                    title="Doc node",
                    body="",
                )

                tag_id = bundle.meta_store.create_tag(graph_id, "docs")
                group_id = bundle.meta_store.create_group(graph_id, "library")
                bundle.meta_store.add_node_tag(node_id, tag_id)
                bundle.meta_store.add_node_to_group(node_id, group_id)

                tags = bundle.meta_store.list_node_tags(node_id)
                groups = bundle.meta_store.list_node_groups(node_id)
                self.assertEqual([tag["tag_id"] for tag in tags], [tag_id])
                self.assertEqual([group["group_id"] for group in groups], [group_id])

                blob_ref = bundle.blob_store.put(b"content", content_type="text/markdown")
                content_item_id = bundle.meta_store.attach_content_item(
                    node_id=node_id,
                    blob_ref=blob_ref,
                    filename="doc.md",
                    mime_type="text/markdown",
                )
                bundle.meta_store.set_content_item_extraction(
                    content_item_id,
                    status="done",
                    extracted_text="alpha phrase",
                )

                alpha_results = bundle.meta_store.search_fulltext("alpha", graph_id=graph_id)
                self.assertEqual([row["node_id"] for row in alpha_results], [node_id])

                bundle.meta_store.set_content_item_extraction(
                    content_item_id,
                    status="done",
                    extracted_text="beta phrase",
                )
                alpha_after_update = bundle.meta_store.search_fulltext("alpha", graph_id=graph_id)
                beta_after_update = bundle.meta_store.search_fulltext("beta", graph_id=graph_id)
                self.assertEqual(alpha_after_update, [])
                self.assertEqual([row["node_id"] for row in beta_after_update], [node_id])
                prefix_results = bundle.meta_store.search_fulltext("be", graph_id=graph_id)
                self.assertEqual([row["node_id"] for row in prefix_results], [node_id])

                bundle.meta_store.remove_node_tag(node_id, tag_id)
                bundle.meta_store.remove_node_from_group(node_id, group_id)
                self.assertEqual(bundle.meta_store.list_node_tags(node_id), [])
                self.assertEqual(bundle.meta_store.list_node_groups(node_id), [])
            finally:
                close_meta_store = getattr(bundle.meta_store, "close", None)
                if callable(close_meta_store):
                    close_meta_store()

    def _build_bundle(self, tmp_path: Path) -> object:
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
            }
        )
        return ComponentFactory(build_default_registry()).create(config)


if __name__ == "__main__":
    unittest.main()

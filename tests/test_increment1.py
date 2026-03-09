from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from smart_journal.config import AppConfig
from smart_journal.factories import ComponentFactory
from smart_journal.registry import build_default_registry


class IncrementOneAcceptanceTests(unittest.TestCase):
    def test_sqlite_schema_v1_contains_required_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "meta.db"
            bundle = ComponentFactory(build_default_registry()).create(
                AppConfig.from_mapping(
                    {
                        "meta_store": {
                            "backend": "sqlite",
                            "path": str(db_path),
                        }
                    }
                )
            )
            try:
                conn = sqlite3.connect(db_path)
                try:
                    rows = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                finally:
                    conn.close()
                table_names = {str(row[0]) for row in rows}

                self.assertTrue(
                    {"nodes", "revisions", "content_items", "tags", "groups", "edges"}.issubset(
                        table_names
                    )
                )
            finally:
                close_meta_store = getattr(bundle.meta_store, "close", None)
                if callable(close_meta_store):
                    close_meta_store()

    def test_acceptance_flow_create_attach_read_and_soft_delete(self) -> None:
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
                }
            )
            bundle = ComponentFactory(build_default_registry()).create(config)
            try:
                graph_id = bundle.meta_store.create_graph("Acceptance Graph")
                graphs = bundle.meta_store.list_graphs()
                self.assertEqual(len(graphs), 1)
                self.assertEqual(str(graphs[0]["graph_id"]), graph_id)
                node_id = bundle.meta_store.create_node(
                    graph_id=graph_id,
                    title="Node with files",
                    body="Initial body",
                )
                bundle.meta_store.update_node(node_id, body="Updated body")
                revisions = bundle.meta_store.list_revisions(node_id)
                self.assertEqual(len(revisions), 2)

                blob_ref_a = bundle.blob_store.put(b"first file", content_type="text/plain")
                blob_ref_b = bundle.blob_store.put(b"second file", content_type="text/plain")

                _ = bundle.meta_store.attach_content_item(
                    node_id=node_id,
                    blob_ref=blob_ref_a,
                    filename="first.txt",
                    mime_type="text/plain",
                )
                _ = bundle.meta_store.attach_content_item(
                    node_id=node_id,
                    blob_ref=blob_ref_b,
                    filename="second.txt",
                    mime_type="text/plain",
                )

                items = bundle.meta_store.list_content_items(node_id)
                self.assertEqual(len(items), 2)
                self.assertEqual(bundle.blob_store.open(blob_ref_a), b"first file")
                self.assertEqual(bundle.blob_store.open(blob_ref_b), b"second file")

                bundle.meta_store.delete_node(node_id, soft_delete=True)
                self.assertIsNone(bundle.meta_store.get_node(node_id))
                deleted_node = bundle.meta_store.get_node(node_id, include_deleted=True)
                self.assertIsNotNone(deleted_node)
                assert deleted_node is not None
                self.assertIsNotNone(deleted_node["deleted_at"])
            finally:
                close_meta_store = getattr(bundle.meta_store, "close", None)
                if callable(close_meta_store):
                    close_meta_store()

    def test_sqlite_meta_store_connection_is_usable_from_another_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "meta.db"
            bundle = ComponentFactory(build_default_registry()).create(
                AppConfig.from_mapping(
                    {
                        "meta_store": {
                            "backend": "sqlite",
                            "path": str(db_path),
                        }
                    }
                )
            )
            try:
                bundle.meta_store.create_graph("Threaded graph")
                errors: list[BaseException] = []

                def _worker() -> None:
                    try:
                        _ = bundle.meta_store.list_graphs()
                    except BaseException as error:  # noqa: BLE001
                        errors.append(error)

                thread = threading.Thread(target=_worker)
                thread.start()
                thread.join(timeout=5)
                self.assertFalse(errors, f"Unexpected thread error: {errors}")
            finally:
                close_meta_store = getattr(bundle.meta_store, "close", None)
                if callable(close_meta_store):
                    close_meta_store()


if __name__ == "__main__":
    unittest.main()

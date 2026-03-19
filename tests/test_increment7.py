from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_journal.providers.sqlite_meta import SQLiteMetaStore


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


if __name__ == "__main__":
    unittest.main()

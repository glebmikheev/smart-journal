from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from smart_journal.web.app import create_app


class IncrementEightApiIntegrationTests(unittest.TestCase):
    def test_explore_and_edge_status_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self._client(Path(tmp_dir)) as client:
                graph_id = self._create_graph(client, "Increment 8 API graph")
                node_a = self._create_node(
                    client,
                    graph_id=graph_id,
                    title="Node A",
                    body="alpha risk and milestone planning",
                )
                node_b = self._create_node(
                    client,
                    graph_id=graph_id,
                    title="Node B",
                    body="milestone dependencies and mitigation notes",
                )
                self._attach_text(
                    client,
                    node_id=node_a,
                    filename="a.md",
                    text="alpha milestone risk details and context",
                )
                self._attach_text(
                    client,
                    node_id=node_b,
                    filename="b.md",
                    text="milestone dependency context and risk mitigation",
                )

                explore_response = client.post(
                    "/api/explore/run",
                    json={
                        "query": "alpha milestone risk dependencies",
                        "graph_id": graph_id,
                        "top_k_chunks": 10,
                        "max_inferences": 4,
                        "create_synthesis": True,
                        "replay_vector_ops": True,
                    },
                )
                self.assertEqual(explore_response.status_code, 200)
                explore_payload = explore_response.json()
                self.assertTrue(explore_payload["explore_session_id"])
                self.assertEqual(len(explore_payload["prompt_hash"]), 64)
                self.assertGreaterEqual(len(explore_payload["retrieval"]), 2)
                self.assertGreaterEqual(len(explore_payload["inferences"]), 1)

                inference = explore_payload["inferences"][0]
                edge_id = str(inference["edge_id"])
                self.assertIsInstance(inference.get("provenance"), dict)
                self.assertEqual(
                    str(inference["provenance"].get("explore_session_id")),
                    str(explore_payload["explore_session_id"]),
                )
                self.assertEqual(
                    str(inference["provenance"].get("query")),
                    str(explore_payload["query"]),
                )
                self.assertIsNotNone(explore_payload.get("synthesis_node_id"))

                accept_response = client.post(f"/api/edges/{edge_id}/accept")
                self.assertEqual(accept_response.status_code, 200)
                accepted_edge = accept_response.json()
                self.assertEqual(str(accepted_edge["status"]), "accepted")

                reject_response = client.post(f"/api/edges/{edge_id}/reject")
                self.assertEqual(reject_response.status_code, 200)
                rejected_edge = reject_response.json()
                self.assertEqual(str(rejected_edge["status"]), "rejected")

    def test_details_and_topology_include_edges_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self._client(Path(tmp_dir)) as client:
                graph_id = self._create_graph(client, "Topology graph")
                node_a = self._create_node(
                    client,
                    graph_id=graph_id,
                    title="A",
                    body="alpha planning",
                )
                node_b = self._create_node(
                    client,
                    graph_id=graph_id,
                    title="B",
                    body="alpha dependencies",
                )
                self._attach_text(
                    client,
                    node_id=node_a,
                    filename="a.md",
                    text="alpha planning and delivery risks",
                )
                self._attach_text(
                    client,
                    node_id=node_b,
                    filename="b.md",
                    text="delivery dependencies and alpha risk signals",
                )

                explore_response = client.post(
                    "/api/explore/run",
                    json={
                        "query": "alpha dependencies",
                        "graph_id": graph_id,
                        "top_k_chunks": 8,
                        "max_inferences": 3,
                        "create_synthesis": False,
                        "replay_vector_ops": True,
                    },
                )
                self.assertEqual(explore_response.status_code, 200)
                explore_payload = explore_response.json()
                self.assertGreaterEqual(len(explore_payload["inferences"]), 1)
                from_node_id = str(explore_payload["inferences"][0]["from_node_id"])

                graph_details_response = client.get(f"/api/graphs/{graph_id}/details")
                self.assertEqual(graph_details_response.status_code, 200)
                graph_details = graph_details_response.json()
                self.assertEqual(graph_details["edges"]["supported"], True)
                self.assertGreaterEqual(int(graph_details["edges"]["count"]), 1)
                self.assertIn("implication", graph_details["edges"]["by_type"])
                self.assertGreaterEqual(int(graph_details["edges"]["by_status"]["pending"]), 1)

                topology_response = client.get(f"/api/graphs/{graph_id}/topology")
                self.assertEqual(topology_response.status_code, 200)
                topology = topology_response.json()
                self.assertEqual(topology["edges"]["supported"], True)
                self.assertGreaterEqual(int(topology["edges"]["count"]), 1)
                self.assertEqual(
                    len(topology["edges"]["items"]),
                    int(topology["edges"]["count"]),
                )

                node_details_response = client.get(f"/api/nodes/{from_node_id}/details")
                self.assertEqual(node_details_response.status_code, 200)
                node_details = node_details_response.json()
                relationships = node_details["relationships"]
                self.assertEqual(relationships["supported"], True)
                self.assertGreaterEqual(int(relationships["count"]), 1)
                relationship = relationships["items"][0]
                self.assertIn(relationship["direction"], {"incoming", "outgoing", "self"})
                self.assertTrue(str(relationship["other_node_id"]))
                self.assertIsInstance(relationship.get("provenance"), dict)

    def test_revision_diff_endpoint_reports_manifest_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self._client(Path(tmp_dir)) as client:
                graph_id = self._create_graph(client, "Revision graph")
                node_id = self._create_node(
                    client,
                    graph_id=graph_id,
                    title="Revision node",
                    body="v1",
                )

                self._attach_text(
                    client,
                    node_id=node_id,
                    filename="first.txt",
                    text="first attachment",
                )
                patch_one = client.patch(f"/api/nodes/{node_id}", json={"body": "v2"})
                self.assertEqual(patch_one.status_code, 200)

                revisions_after_first_patch = client.get(f"/api/nodes/{node_id}/revisions")
                self.assertEqual(revisions_after_first_patch.status_code, 200)
                revision_ids = [
                    str(row["revision_id"])
                    for row in revisions_after_first_patch.json()
                ]
                self.assertGreaterEqual(len(revision_ids), 2)
                from_revision_id = revision_ids[-1]

                second_attachment = self._attach_text(
                    client,
                    node_id=node_id,
                    filename="second.txt",
                    text="second attachment",
                )
                second_attachment_id = str(second_attachment["content_item"]["content_item_id"])
                patch_two = client.patch(f"/api/nodes/{node_id}", json={"body": "v3"})
                self.assertEqual(patch_two.status_code, 200)

                revisions_after_second_patch = client.get(f"/api/nodes/{node_id}/revisions")
                self.assertEqual(revisions_after_second_patch.status_code, 200)
                revision_ids = [
                    str(row["revision_id"])
                    for row in revisions_after_second_patch.json()
                ]
                self.assertGreaterEqual(len(revision_ids), 3)
                to_revision_id = revision_ids[-1]

                diff_response = client.get(
                    f"/api/nodes/{node_id}/revisions/diff",
                    params={
                        "from_revision_id": from_revision_id,
                        "to_revision_id": to_revision_id,
                    },
                )
                self.assertEqual(diff_response.status_code, 200)
                diff_payload = diff_response.json()
                self.assertEqual(str(diff_payload["node_id"]), node_id)
                self.assertEqual(diff_payload["added_content_item_ids"], [second_attachment_id])
                self.assertEqual(diff_payload["removed_content_item_ids"], [])
                self.assertTrue(bool(diff_payload["body_changed"]))

    def _client(self, tmp_dir: Path) -> TestClient:
        config_path = tmp_dir / "smart-journal.test.toml"
        blobs_root = (tmp_dir / "blobs").as_posix()
        meta_path = (tmp_dir / "meta.db").as_posix()
        index_root = (tmp_dir / "indexes").as_posix()
        config_path.write_text(
            "\n".join(
                [
                    "[blob_store]",
                    'backend = "local_cas"',
                    f'root = "{blobs_root}"',
                    "",
                    "[meta_store]",
                    'backend = "sqlite"',
                    f'path = "{meta_path}"',
                    "",
                    "[vector_index]",
                    'backend = "in_memory"',
                    f'index_path = "{index_root}"',
                    "",
                    "[job_queue]",
                    'backend = "in_process"',
                    "",
                    "[extractor]",
                    'backend = "basic_v1"',
                    "",
                    "[embedding_provider]",
                    'backend = "mock_text"',
                    "dim = 8",
                    "normalize = true",
                    "",
                    "[llm_provider]",
                    'backend = "mock_chat"',
                ]
            ),
            encoding="utf-8",
        )
        return TestClient(create_app(config_path=config_path))

    def _create_graph(self, client: TestClient, title: str) -> str:
        response = client.post("/api/graphs", json={"title": title})
        self.assertEqual(response.status_code, 201)
        payload = response.json()
        return str(payload["graph_id"])

    def _create_node(
        self,
        client: TestClient,
        *,
        graph_id: str,
        title: str,
        body: str = "",
    ) -> str:
        response = client.post(
            f"/api/graphs/{graph_id}/nodes",
            json={
                "title": title,
                "body": body,
            },
        )
        self.assertEqual(response.status_code, 201)
        payload = response.json()
        return str(payload["node_id"])

    def _attach_text(
        self,
        client: TestClient,
        *,
        node_id: str,
        filename: str,
        text: str,
    ) -> dict[str, object]:
        response = client.post(
            f"/api/nodes/{node_id}/content-items",
            files={"file": (filename, text.encode("utf-8"), "text/markdown")},
        )
        self.assertEqual(response.status_code, 201)
        return response.json()


if __name__ == "__main__":
    unittest.main()

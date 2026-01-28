from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from smart_journal.cli import run_cli


class CliTests(unittest.TestCase):
    def test_providers_command_outputs_capabilities_json(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = run_cli(["providers", "--json"])
        self.assertEqual(exit_code, 0)

        payload = json.loads(output.getvalue())
        self.assertIn("embedding_provider", payload)
        provider = payload["embedding_provider"][0]
        self.assertEqual(provider["provider_id"], "mock_text")
        self.assertTrue(provider["capabilities"]["text"])

    def test_run_command_bootstraps_empty_app_shell(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = run_cli(["run", "--json"])
        self.assertEqual(exit_code, 0)

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["blob_store"]["provider_id"], "in_memory")
        self.assertEqual(payload["llm_provider"]["provider_id"], "mock_chat")


if __name__ == "__main__":
    unittest.main()


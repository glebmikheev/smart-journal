from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from smart_journal.cli import run_cli


class CliTests(unittest.TestCase):
    def test_providers_command_outputs_capabilities_json(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = run_cli(["providers", "--json"])
        self.assertEqual(exit_code, 0)

        payload = json.loads(output.getvalue())
        self.assertIn("embedding_provider", payload)
        providers = payload["embedding_provider"]
        provider_ids = {provider["provider_id"] for provider in providers}
        self.assertIn("mock_text", provider_ids)
        self.assertIn("multilingual_e5_small", provider_ids)

    def test_run_command_bootstraps_empty_app_shell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "smart-journal.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[blob_store]",
                        'backend = "local_cas"',
                        f'root = "{(Path(tmp_dir) / "blobs").as_posix()}"',
                        "",
                        "[meta_store]",
                        'backend = "sqlite"',
                        f'path = "{(Path(tmp_dir) / "meta.db").as_posix()}"',
                        "",
                        "[llm_provider]",
                        'backend = "mock_chat"',
                    ]
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = run_cli(["--config", str(config_path), "run", "--json"])
            self.assertEqual(exit_code, 0)

            payload = json.loads(output.getvalue())
            self.assertEqual(payload["blob_store"]["provider_id"], "local_cas")
            self.assertEqual(payload["meta_store"]["provider_id"], "sqlite")
            self.assertEqual(payload["llm_provider"]["provider_id"], "mock_chat")


if __name__ == "__main__":
    unittest.main()

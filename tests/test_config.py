from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_journal.config import AppConfig, load_config


class ConfigTests(unittest.TestCase):
    def test_from_mapping_overrides_backends_and_options(self) -> None:
        config = AppConfig.from_mapping(
            {
                "blob_store": {"backend": "in_memory", "root": "./blobs"},
                "embedding_provider": {"backend": "mock_text", "dim": 16, "normalize": False},
            }
        )
        self.assertEqual(config.blob_store.backend, "in_memory")
        self.assertEqual(config.blob_store.options["root"], "./blobs")
        self.assertEqual(config.embedding_provider.options["dim"], 16)
        self.assertFalse(config.embedding_provider.options["normalize"])

    def test_load_config_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "smart-journal.toml"
            path.write_text(
                "\n".join(
                    [
                        "[blob_store]",
                        'backend = "in_memory"',
                        "",
                        "[llm_provider]",
                        'backend = "mock_chat"',
                    ]
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.blob_store.backend, "in_memory")
            self.assertEqual(config.llm_provider.backend, "mock_chat")


if __name__ == "__main__":
    unittest.main()


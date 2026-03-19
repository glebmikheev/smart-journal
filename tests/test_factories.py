from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from smart_journal.config import AppConfig
from smart_journal.factories import ComponentFactory
from smart_journal.registry import build_default_registry


class FactoryTests(unittest.TestCase):
    def test_factory_creates_all_components(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            registry = build_default_registry()
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
            bundle = ComponentFactory(registry).create(config)

            try:
                self.assertEqual(bundle.blob_store.provider_id(), "local_cas")
                self.assertEqual(bundle.meta_store.provider_id(), "sqlite")
                self.assertEqual(bundle.vector_index.provider_id(), "in_memory")
                self.assertEqual(bundle.job_queue.provider_id(), "in_process")
                self.assertEqual(bundle.extractor.provider_id(), "basic_v1")
                self.assertEqual(bundle.embedding_provider.provider_id(), "multilingual_e5_small")
                self.assertEqual(bundle.llm_provider.provider_id(), "mock_chat")
            finally:
                close_meta_store = getattr(bundle.meta_store, "close", None)
                if callable(close_meta_store):
                    close_meta_store()


if __name__ == "__main__":
    unittest.main()

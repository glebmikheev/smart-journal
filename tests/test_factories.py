from __future__ import annotations

import unittest

from smart_journal.config import AppConfig
from smart_journal.factories import ComponentFactory
from smart_journal.registry import build_default_registry


class FactoryTests(unittest.TestCase):
    def test_factory_creates_all_components(self) -> None:
        registry = build_default_registry()
        bundle = ComponentFactory(registry).create(AppConfig())

        self.assertEqual(bundle.blob_store.provider_id(), "local_cas")
        self.assertEqual(bundle.meta_store.provider_id(), "sqlite")
        self.assertEqual(bundle.vector_index.provider_id(), "in_memory")
        self.assertEqual(bundle.job_queue.provider_id(), "in_process")
        self.assertEqual(bundle.extractor.provider_id(), "basic_v1")
        self.assertEqual(bundle.embedding_provider.provider_id(), "mock_text")
        self.assertEqual(bundle.llm_provider.provider_id(), "mock_chat")


if __name__ == "__main__":
    unittest.main()

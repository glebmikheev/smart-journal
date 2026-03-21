from __future__ import annotations

import unittest

from smart_journal.registry import build_default_registry


class RegistryTests(unittest.TestCase):
    def test_registry_lists_all_required_categories(self) -> None:
        registry = build_default_registry()
        categories = registry.categories()
        self.assertIn("blob_store", categories)
        self.assertIn("meta_store", categories)
        self.assertIn("vector_index", categories)
        self.assertIn("job_queue", categories)
        self.assertIn("extractor", categories)
        self.assertIn("embedding_provider", categories)
        self.assertIn("llm_provider", categories)

    def test_registry_can_create_configured_provider(self) -> None:
        registry = build_default_registry()
        provider = registry.create(category="embedding_provider", provider_id="mock_text")
        self.assertEqual(provider.provider_id(), "mock_text")
        self.assertTrue(bool(provider.capabilities()["text"]))
        e5_provider = registry.create(
            category="embedding_provider",
            provider_id="multilingual_e5_small",
        )
        self.assertEqual(e5_provider.provider_id(), "multilingual_e5_small")

    def test_registry_exposes_increment_one_backends(self) -> None:
        registry = build_default_registry()
        blob_ids = [descriptor.provider_id for descriptor in registry.available("blob_store")]
        meta_ids = [descriptor.provider_id for descriptor in registry.available("meta_store")]
        vector_ids = [descriptor.provider_id for descriptor in registry.available("vector_index")]
        extractor_ids = [descriptor.provider_id for descriptor in registry.available("extractor")]
        llm_ids = [descriptor.provider_id for descriptor in registry.available("llm_provider")]
        self.assertIn("local_cas", blob_ids)
        self.assertIn("sqlite", meta_ids)
        self.assertIn("usearch_file", vector_ids)
        self.assertIn("basic_v1", extractor_ids)
        self.assertIn("mock_chat", llm_ids)
        self.assertIn("ollama_chat", llm_ids)


if __name__ == "__main__":
    unittest.main()

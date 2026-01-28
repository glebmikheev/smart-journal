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


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from smart_journal.contracts import Capabilities, ProviderInfo
from smart_journal.providers import (
    InMemoryBlobStore,
    InMemoryMetaStore,
    InMemoryVectorIndex,
    InProcessJobQueue,
    LocalCASBlobStore,
    MockEmbeddingProvider,
    MockLLMProvider,
    PlainTextExtractor,
    SQLiteMetaStore,
)

ProviderConstructor = Callable[[Mapping[str, Any]], ProviderInfo]


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    category: str
    provider_id: str
    version: str
    capabilities: Capabilities


class ProviderRegistry:
    def __init__(self) -> None:
        self._constructors: dict[str, dict[str, ProviderConstructor]] = {}

    def register(
        self,
        *,
        category: str,
        provider_id: str,
        constructor: ProviderConstructor,
    ) -> None:
        self._constructors.setdefault(category, {})[provider_id] = constructor

    def categories(self) -> list[str]:
        return sorted(self._constructors)

    def available(self, category: str) -> list[ProviderDescriptor]:
        constructors = self._constructors.get(category, {})
        descriptions: list[ProviderDescriptor] = []
        for provider_id, constructor in sorted(constructors.items()):
            provider = constructor({})
            try:
                descriptions.append(
                    ProviderDescriptor(
                        category=category,
                        provider_id=provider_id,
                        version=provider.version(),
                        capabilities=dict(provider.capabilities()),
                    )
                )
            finally:
                closer = getattr(provider, "close", None)
                if callable(closer):
                    closer()
        return descriptions

    def available_all(self) -> dict[str, list[ProviderDescriptor]]:
        return {category: self.available(category) for category in self.categories()}

    def create(
        self,
        *,
        category: str,
        provider_id: str,
        options: Mapping[str, Any] | None = None,
    ) -> ProviderInfo:
        constructors = self._constructors.get(category)
        if constructors is None:
            known = ", ".join(self.categories()) or "<none>"
            raise KeyError(f"Unknown provider category '{category}'. Known categories: {known}")

        constructor = constructors.get(provider_id)
        if constructor is None:
            known = ", ".join(sorted(constructors)) or "<none>"
            raise KeyError(
                "Unknown provider "
                f"'{provider_id}' for category '{category}'. Known providers: {known}"
            )

        return constructor(options or {})


def build_default_registry() -> ProviderRegistry:
    registry = ProviderRegistry()

    registry.register(
        category="blob_store",
        provider_id="in_memory",
        constructor=lambda options: InMemoryBlobStore(options),
    )
    registry.register(
        category="blob_store",
        provider_id="local_cas",
        constructor=lambda options: LocalCASBlobStore(options),
    )
    registry.register(
        category="meta_store",
        provider_id="in_memory",
        constructor=lambda options: InMemoryMetaStore(options),
    )
    registry.register(
        category="meta_store",
        provider_id="sqlite",
        constructor=lambda options: SQLiteMetaStore(options),
    )
    registry.register(
        category="vector_index",
        provider_id="in_memory",
        constructor=lambda options: InMemoryVectorIndex(options),
    )
    registry.register(
        category="job_queue",
        provider_id="in_process",
        constructor=lambda options: InProcessJobQueue(options),
    )
    registry.register(
        category="extractor",
        provider_id="plain_text",
        constructor=lambda options: PlainTextExtractor(options),
    )
    registry.register(
        category="embedding_provider",
        provider_id="mock_text",
        constructor=lambda options: MockEmbeddingProvider(options),
    )
    registry.register(
        category="llm_provider",
        provider_id="mock_chat",
        constructor=lambda options: MockLLMProvider(options),
    )

    return registry

from __future__ import annotations

from dataclasses import dataclass

from smart_journal.config import AppConfig
from smart_journal.contracts import (
    BlobStore,
    EmbeddingProvider,
    Extractor,
    JobQueue,
    LLMProvider,
    MetaStore,
    VectorIndex,
)
from smart_journal.registry import ProviderRegistry


@dataclass(slots=True)
class ComponentBundle:
    blob_store: BlobStore
    meta_store: MetaStore
    vector_index: VectorIndex
    job_queue: JobQueue
    extractor: Extractor
    embedding_provider: EmbeddingProvider
    llm_provider: LLMProvider


class ComponentFactory:
    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def create(self, config: AppConfig) -> ComponentBundle:
        blob_store_raw = self._registry.create(
            category="blob_store",
            provider_id=config.blob_store.backend,
            options=config.blob_store.options,
        )
        meta_store_raw = self._registry.create(
            category="meta_store",
            provider_id=config.meta_store.backend,
            options=config.meta_store.options,
        )
        vector_index_raw = self._registry.create(
            category="vector_index",
            provider_id=config.vector_index.backend,
            options=config.vector_index.options,
        )
        job_queue_raw = self._registry.create(
            category="job_queue",
            provider_id=config.job_queue.backend,
            options=config.job_queue.options,
        )
        extractor_raw = self._registry.create(
            category="extractor",
            provider_id=config.extractor.backend,
            options=config.extractor.options,
        )
        embedding_raw = self._registry.create(
            category="embedding_provider",
            provider_id=config.embedding_provider.backend,
            options=config.embedding_provider.options,
        )
        llm_raw = self._registry.create(
            category="llm_provider",
            provider_id=config.llm_provider.backend,
            options=config.llm_provider.options,
        )

        if not isinstance(blob_store_raw, BlobStore):
            raise TypeError("Configured blob_store provider does not implement BlobStore.")
        if not isinstance(meta_store_raw, MetaStore):
            raise TypeError("Configured meta_store provider does not implement MetaStore.")
        if not isinstance(vector_index_raw, VectorIndex):
            raise TypeError("Configured vector_index provider does not implement VectorIndex.")
        if not isinstance(job_queue_raw, JobQueue):
            raise TypeError("Configured job_queue provider does not implement JobQueue.")
        if not isinstance(extractor_raw, Extractor):
            raise TypeError("Configured extractor provider does not implement Extractor.")
        if not isinstance(embedding_raw, EmbeddingProvider):
            raise TypeError(
                "Configured embedding_provider does not implement EmbeddingProvider."
            )
        if not isinstance(llm_raw, LLMProvider):
            raise TypeError("Configured llm_provider does not implement LLMProvider.")

        return ComponentBundle(
            blob_store=blob_store_raw,
            meta_store=meta_store_raw,
            vector_index=vector_index_raw,
            job_queue=job_queue_raw,
            extractor=extractor_raw,
            embedding_provider=embedding_raw,
            llm_provider=llm_raw,
        )

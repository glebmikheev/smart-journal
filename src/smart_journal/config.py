from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_FILE = "smart-journal.toml"


@dataclass(slots=True)
class ComponentConfig:
    backend: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AppConfig:
    blob_store: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(
            backend="local_cas",
            options={"root": "./data/blobs"},
        )
    )
    meta_store: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(
            backend="sqlite",
            options={"path": "./data/meta.db"},
        )
    )
    vector_index: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(backend="in_memory")
    )
    job_queue: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(backend="in_process")
    )
    extractor: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(backend="basic_v1")
    )
    embedding_provider: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(backend="mock_text")
    )
    llm_provider: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(backend="mock_chat")
    )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> AppConfig:
        return cls(
            blob_store=_component_from_section(
                raw.get("blob_store"),
                default_backend="local_cas",
                default_options={"root": "./data/blobs"},
            ),
            meta_store=_component_from_section(
                raw.get("meta_store"),
                default_backend="sqlite",
                default_options={"path": "./data/meta.db"},
            ),
            vector_index=_component_from_section(
                raw.get("vector_index"), default_backend="in_memory"
            ),
            job_queue=_component_from_section(raw.get("job_queue"), default_backend="in_process"),
            extractor=_component_from_section(raw.get("extractor"), default_backend="basic_v1"),
            embedding_provider=_component_from_section(
                raw.get("embedding_provider"),
                default_backend="mock_text",
            ),
            llm_provider=_component_from_section(
                raw.get("llm_provider"), default_backend="mock_chat"
            ),
        )


def load_config(config_path: Path | None = None) -> AppConfig:
    if config_path is None:
        default_path = Path(DEFAULT_CONFIG_FILE)
        if not default_path.exists():
            return AppConfig()
        config_path = default_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return AppConfig.from_mapping(raw_data)


def _component_from_section(
    raw_section: Any,
    *,
    default_backend: str,
    default_options: Mapping[str, Any] | None = None,
) -> ComponentConfig:
    default_options_dict = dict(default_options or {})
    if not isinstance(raw_section, Mapping):
        return ComponentConfig(backend=default_backend, options=default_options_dict)

    backend = str(raw_section.get("backend", default_backend))
    options = default_options_dict
    options.update(
        {
            str(key): value
            for key, value in raw_section.items()
            if key != "backend"
        }
    )
    return ComponentConfig(backend=backend, options=options)

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from smart_journal.config import load_config
from smart_journal.contracts import ProviderInfo
from smart_journal.factories import ComponentBundle, ComponentFactory
from smart_journal.registry import (
    ProviderDescriptor,
    ProviderRegistry,
    build_default_registry,
)
from smart_journal.vector_ops import VectorIndexOpsReplayer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-journal",
        description="Smart Journal increment 5 CLI.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to TOML config file. If omitted, smart-journal.toml is used when present.",
    )

    subparsers = parser.add_subparsers(dest="command")
    providers_parser = subparsers.add_parser(
        "providers",
        help="List available providers by category.",
    )
    providers_parser.add_argument("--json", action="store_true", dest="as_json")

    run_parser = subparsers.add_parser(
        "run",
        help="Create app shell and print selected providers.",
    )
    run_parser.add_argument("--json", action="store_true", dest="as_json")

    return parser


def run_cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "providers"

    registry = build_default_registry()

    if command == "providers":
        available_payload = _available_providers_payload(registry)
        if args.as_json:
            print(json.dumps(available_payload, indent=2, ensure_ascii=False))
        else:
            _print_available(available_payload)
        return 0

    if command == "run":
        config = load_config(args.config)
        bundle = ComponentFactory(registry).create(config)
        try:
            _bootstrap_vector_index(bundle)
            selected_payload = _selected_providers_payload(bundle)
            if args.as_json:
                print(json.dumps(selected_payload, indent=2, ensure_ascii=False))
            else:
                print("Smart Journal is running.")
                _print_selected(selected_payload)
            return 0
        finally:
            _close_bundle_resources(bundle)

    parser.print_help()
    return 1


def main() -> None:
    raise SystemExit(run_cli())


def _provider_payload(provider: ProviderInfo) -> dict[str, Any]:
    return {
        "provider_id": provider.provider_id(),
        "version": provider.version(),
        "capabilities": dict(provider.capabilities()),
    }


def _descriptor_payload(descriptor: ProviderDescriptor) -> dict[str, Any]:
    return {
        "provider_id": descriptor.provider_id,
        "version": descriptor.version,
        "capabilities": dict(descriptor.capabilities),
    }


def _available_providers_payload(registry: ProviderRegistry) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for category, descriptors in registry.available_all().items():
        payload[category] = [_descriptor_payload(descriptor) for descriptor in descriptors]
    return payload


def _selected_providers_payload(bundle: ComponentBundle) -> dict[str, dict[str, Any]]:
    return {
        "blob_store": _provider_payload(bundle.blob_store),
        "meta_store": _provider_payload(bundle.meta_store),
        "vector_index": _provider_payload(bundle.vector_index),
        "job_queue": _provider_payload(bundle.job_queue),
        "extractor": _provider_payload(bundle.extractor),
        "embedding_provider": _provider_payload(bundle.embedding_provider),
        "llm_provider": _provider_payload(bundle.llm_provider),
    }


def _print_available(payload: dict[str, list[dict[str, Any]]]) -> None:
    print("Available providers:")
    for category, providers in payload.items():
        print(f"- {category}")
        for provider in providers:
            caps = ", ".join(sorted(str(key) for key in provider["capabilities"]))
            print(f"  - {provider['provider_id']} (v{provider['version']}) | capabilities: {caps}")


def _print_selected(payload: dict[str, dict[str, Any]]) -> None:
    print("Selected providers:")
    for category, provider in payload.items():
        capabilities = json.dumps(provider["capabilities"], ensure_ascii=False, sort_keys=True)
        print(f"- {category}: {provider['provider_id']} (v{provider['version']})")
        print(f"  capabilities: {capabilities}")


def _close_bundle_resources(bundle: ComponentBundle) -> None:
    for provider in (
        bundle.blob_store,
        bundle.meta_store,
        bundle.vector_index,
        bundle.job_queue,
        bundle.extractor,
        bundle.embedding_provider,
        bundle.llm_provider,
    ):
        closer = getattr(provider, "close", None)
        if callable(closer):
            closer()


def _bootstrap_vector_index(bundle: ComponentBundle) -> None:
    bundle.vector_index.load()
    replayer = VectorIndexOpsReplayer(
        meta_store=bundle.meta_store,
        vector_index=bundle.vector_index,
        model_id=bundle.embedding_provider.model_id(),
    )
    replayer.replay_pending()


if __name__ == "__main__":
    sys.exit(run_cli())

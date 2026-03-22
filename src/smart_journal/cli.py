from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib import error, request

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
        description="Smart Journal increment 8 CLI.",
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
    serve_parser = subparsers.add_parser(
        "serve",
        help="Run FastAPI server for web UI.",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Check optional runtime dependencies (ocr/asr/llm).",
    )
    preflight_parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help=(
            "Profile to check: core, ocr, asr, llm, all. "
            "Can be repeated or passed as comma-separated values."
        ),
    )
    preflight_parser.add_argument("--json", action="store_true", dest="as_json")
    preflight_parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when any selected check fails.",
    )

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

    if command == "serve":
        if args.config is not None:
            os.environ["SMART_JOURNAL_CONFIG"] = str(args.config)
        else:
            os.environ.pop("SMART_JOURNAL_CONFIG", None)
        try:
            import uvicorn

            from smart_journal.web import create_app

            _ = create_app()
        except (ImportError, RuntimeError) as error:
            print(
                "UI dependencies are not installed. "
                "Install with: python -m pip install -e .[ui]"
            )
            print(f"Details: {error}")
            return 2

        uvicorn.run(
            "smart_journal.web.app:create_app",
            factory=True,
            host=str(args.host),
            port=int(args.port),
            reload=bool(args.reload),
        )
        return 0

    if command == "preflight":
        selected_profiles = _resolve_preflight_profiles(list(args.profile))
        if selected_profiles is None:
            print(
                "Unknown preflight profile. Supported values: core, ocr, asr, llm, all."
            )
            return 2
        payload = _run_preflight(selected_profiles)
        if args.as_json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            _print_preflight(payload)
        if args.strict and not bool(payload.get("ok", False)):
            return 2
        return 0

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


def _resolve_preflight_profiles(raw_profiles: list[str]) -> list[str] | None:
    allowed = {"core", "ocr", "asr", "llm", "all"}
    selected: set[str] = set()
    if not raw_profiles:
        return ["core"]

    for raw_item in raw_profiles:
        for token in str(raw_item).split(","):
            normalized = token.strip().lower()
            if not normalized:
                continue
            if normalized not in allowed:
                return None
            if normalized == "all":
                selected.update({"core", "ocr", "asr", "llm"})
            else:
                selected.add(normalized)

    if not selected:
        return ["core"]
    return sorted(selected)


def _run_preflight(profiles: list[str]) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    for profile in profiles:
        if profile == "core":
            _append_preflight_check(
                checks,
                profile="core",
                name="python>=3.12",
                status=("ok" if sys.version_info >= (3, 12) else "fail"),
                detail=f"Detected Python {sys.version.split()[0]}",
                hint="Use Python 3.12+.",
            )
            continue

        if profile == "ocr":
            _append_preflight_check(
                checks,
                profile="ocr",
                name="module:paddle",
                status=_module_check_status("paddle"),
                detail="Required for PP-OCR runtime.",
                hint="Install with: python -m pip install -e .",
            )
            _append_preflight_check(
                checks,
                profile="ocr",
                name="module:paddleocr",
                status=_module_check_status("paddleocr"),
                detail="Required for PP-OCRv5 API.",
                hint="Install with: python -m pip install -e .",
            )
            _append_cache_write_check(
                checks,
                profile="ocr",
                name="cache:PADDLE_HOME",
                path=Path(
                    os.getenv("PADDLE_HOME")
                    or (Path.home() / ".cache" / "paddle")
                ),
                hint="Set PADDLE_HOME to a writable path, e.g. .runtime/paddle.",
            )
            _append_cache_write_check(
                checks,
                profile="ocr",
                name="cache:PADDLE_PDX_CACHE_HOME",
                path=Path(
                    os.getenv("PADDLE_PDX_CACHE_HOME")
                    or (Path.home() / ".paddlex")
                ),
                hint="Set PADDLE_PDX_CACHE_HOME to a writable path, e.g. .runtime/paddlex.",
            )
            continue

        if profile == "asr":
            _append_preflight_check(
                checks,
                profile="asr",
                name="module:whisper",
                status=_module_check_status("whisper"),
                detail="Required for ASR transcription backend.",
                hint="Install with: python -m pip install -e .",
            )
            ffmpeg_path = shutil.which("ffmpeg")
            _append_preflight_check(
                checks,
                profile="asr",
                name="binary:ffmpeg",
                status=("ok" if ffmpeg_path else "fail"),
                detail=(ffmpeg_path or "ffmpeg is not found in PATH."),
                hint="Install ffmpeg and ensure it is available in PATH.",
            )
            _append_cache_write_check(
                checks,
                profile="asr",
                name="cache:XDG_CACHE_HOME",
                path=Path(os.getenv("XDG_CACHE_HOME") or (Path.home() / ".cache")),
                hint="Set XDG_CACHE_HOME to a writable path for Whisper model cache.",
            )
            continue

        if profile == "llm":
            openai_module_ok = _module_check_status("openai") == "ok"
            _append_preflight_check(
                checks,
                profile="llm",
                name="module:openai",
                status=("ok" if openai_module_ok else "warn"),
                detail=(
                    "OpenAI SDK module is available."
                    if openai_module_ok
                    else "OpenAI SDK module is not installed."
                ),
                hint="Install with: python -m pip install -e .",
            )
            has_api_key = bool(str(os.getenv("OPENAI_API_KEY", "")).strip())
            _append_preflight_check(
                checks,
                profile="llm",
                name="env:OPENAI_API_KEY",
                status=("ok" if has_api_key else "warn"),
                detail=(
                    "OPENAI_API_KEY is configured."
                    if has_api_key
                    else "OPENAI_API_KEY is not set."
                ),
                hint="Set OPENAI_API_KEY before running OpenAI smoke/live calls.",
            )

            ollama_binary = shutil.which("ollama")
            _append_preflight_check(
                checks,
                profile="llm",
                name="binary:ollama",
                status=("ok" if ollama_binary else "warn"),
                detail=(
                    ollama_binary
                    if ollama_binary
                    else "ollama CLI is not found in PATH."
                ),
                hint="Install Ollama or ensure 'ollama' is available in PATH.",
            )
            ollama_base_url = (
                str(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).strip()
                or "http://127.0.0.1:11434"
            )
            ollama_online, ollama_detail = _probe_ollama(ollama_base_url)
            _append_preflight_check(
                checks,
                profile="llm",
                name="endpoint:ollama",
                status=("ok" if ollama_online else "warn"),
                detail=ollama_detail,
                hint=(
                    "Start Ollama service (for example: ollama serve) "
                    "or set OLLAMA_BASE_URL."
                ),
            )
            continue

    summary = {"ok": 0, "warn": 0, "fail": 0}
    for check in checks:
        status = str(check["status"])
        if status in summary:
            summary[status] += 1
    return {
        "ok": summary["fail"] == 0,
        "profiles": profiles,
        "summary": summary,
        "checks": checks,
    }


def _module_check_status(module_name: str) -> str:
    return "ok" if importlib.util.find_spec(module_name) is not None else "fail"


def _probe_ollama(base_url: str) -> tuple[bool, str]:
    normalized = base_url.rstrip("/")
    url = f"{normalized}/api/tags"
    try:
        with request.urlopen(url, timeout=1.5) as response:
            status_code = int(getattr(response, "status", 0))
            if status_code < 400:
                return True, f"Reachable: {url} (HTTP {status_code})"
            return False, f"Unhealthy: {url} (HTTP {status_code})"
    except error.URLError as url_error:
        return False, f"Unreachable: {url} ({url_error})"
    except Exception as unknown_error:  # noqa: BLE001
        return False, f"Probe failed: {url} ({unknown_error})"


def _append_cache_write_check(
    checks: list[dict[str, str]],
    *,
    profile: str,
    name: str,
    path: Path,
    hint: str,
) -> None:
    writable, message = _directory_is_writable(path)
    _append_preflight_check(
        checks,
        profile=profile,
        name=name,
        status=("ok" if writable else "fail"),
        detail=(message or f"Writable: {path}"),
        hint=hint,
    )


def _directory_is_writable(path: Path) -> tuple[bool, str]:
    target = path.expanduser()
    if target.exists():
        writable = os.access(target, os.W_OK)
        if writable:
            return True, f"Writable: {target}"
        return False, f"{target} exists but is not writable."

    probe = target.parent
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent

    if not probe.exists():
        return False, f"Cannot resolve writable parent for {target}."

    writable_parent = os.access(probe, os.W_OK)
    if writable_parent:
        return True, f"Writable parent: {probe} (target: {target})"
    return False, f"Parent directory is not writable: {probe} (target: {target})"


def _append_preflight_check(
    checks: list[dict[str, str]],
    *,
    profile: str,
    name: str,
    status: str,
    detail: str,
    hint: str,
) -> None:
    checks.append(
        {
            "profile": profile,
            "name": name,
            "status": status,
            "detail": detail,
            "hint": hint,
        }
    )


def _print_preflight(payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    print("Preflight checks:")
    print(
        f"- ok={summary.get('ok', 0)} warn={summary.get('warn', 0)} fail={summary.get('fail', 0)}"
    )
    for check in payload.get("checks", []):
        print(
            f"- [{check.get('status')}] ({check.get('profile')}) {check.get('name')}: "
            f"{check.get('detail')}"
        )


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

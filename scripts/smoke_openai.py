from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from smart_journal.config import AppConfig, ComponentConfig
from smart_journal.explore import ExploreService
from smart_journal.factories import ComponentFactory
from smart_journal.ingestion import build_default_ingestion_pipeline
from smart_journal.registry import build_default_registry
from smart_journal.vector_ops import VectorIndexOpsReplayer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenAI-backed smoke checks for chat + structured + Explore pipeline.",
    )
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model id.")
    parser.add_argument(
        "--api-key",
        default="",
        help="OpenAI API key. Falls back to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Optional API base URL (for compatible endpoints).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=90.0,
        help="Request timeout for OpenAI calls.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for OpenAI chat calls.",
    )
    parser.add_argument(
        "--query",
        default="What risks connect node A and node B?",
        help="Explore query text for the smoke scenario.",
    )
    return parser


def _build_config(
    *,
    root: Path,
    model: str,
    api_key: str,
    base_url: str,
    timeout_seconds: float,
    temperature: float,
) -> AppConfig:
    llm_options: dict[str, Any] = {
        "model": model,
        "timeout_seconds": timeout_seconds,
        "temperature": temperature,
    }
    if api_key:
        llm_options["api_key"] = api_key
    if base_url:
        llm_options["base_url"] = base_url

    return AppConfig(
        blob_store=ComponentConfig("local_cas", {"root": str(root / "blobs")}),
        meta_store=ComponentConfig("sqlite", {"path": str(root / "meta.db")}),
        vector_index=ComponentConfig("in_memory", {}),
        job_queue=ComponentConfig("in_process", {}),
        extractor=ComponentConfig("basic_v1", {}),
        embedding_provider=ComponentConfig("mock_text", {"dim": 8, "normalize": True}),
        llm_provider=ComponentConfig("openai_chat", llm_options),
    )


def _attach_text(
    *,
    bundle: Any,
    node_id: str,
    filename: str,
    text: str,
) -> str:
    blob_ref = bundle.blob_store.put(text.encode("utf-8"), content_type="text/markdown")
    return bundle.meta_store.attach_content_item(
        node_id=node_id,
        blob_ref=blob_ref,
        filename=filename,
        mime_type="text/markdown",
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    api_key = str(args.api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key and not args.base_url:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        "OpenAI API key is required. Pass --api-key or set OPENAI_API_KEY. "
                        "If you use a compatible proxy, also pass --base-url."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        config = _build_config(
            root=root,
            model=str(args.model),
            api_key=api_key,
            base_url=str(args.base_url).strip(),
            timeout_seconds=float(args.timeout_seconds),
            temperature=float(args.temperature),
        )
        registry = build_default_registry()
        bundle = ComponentFactory(registry).create(config)
        bundle.vector_index.load()

        try:
            graph_id = bundle.meta_store.create_graph("OpenAI smoke graph")
            node_a = bundle.meta_store.create_node(
                graph_id,
                "Node A",
                "alpha release risks and delays",
            )
            node_b = bundle.meta_store.create_node(
                graph_id,
                "Node B",
                "dependency risk mitigation and planning",
            )
            item_a = _attach_text(
                bundle=bundle,
                node_id=node_a,
                filename="a.md",
                text="alpha release schedule and risk dependencies",
            )
            item_b = _attach_text(
                bundle=bundle,
                node_id=node_b,
                filename="b.md",
                text="dependency mapping and mitigation for release risks",
            )

            ingestion = build_default_ingestion_pipeline(
                meta_store=bundle.meta_store,
                blob_store=bundle.blob_store,
                extractor=bundle.extractor,
                job_queue=bundle.job_queue,
                embedding_provider=bundle.embedding_provider,
            )
            ingestion.ingest_content_item_now(item_a)
            ingestion.ingest_content_item_now(item_b)

            replayer = VectorIndexOpsReplayer(
                meta_store=bundle.meta_store,
                vector_index=bundle.vector_index,
                model_id=bundle.embedding_provider.model_id(),
            )
            replay_stats = replayer.replay_pending(limit=1_000)

            chat_text = bundle.llm_provider.chat(
                [{"role": "user", "content": "Reply with exactly one word: alive"}]
            )
            structured_payload = bundle.llm_provider.generate_structured(
                prompt=(
                    "Return JSON object with two fields: "
                    "implications (array) and notes (string)."
                ),
                schema={"implications": "array", "notes": "string"},
            )

            explore = ExploreService(
                meta_store=bundle.meta_store,
                vector_index=bundle.vector_index,
                embedding_provider=bundle.embedding_provider,
                llm_provider=bundle.llm_provider,
            )
            result = explore.run(
                graph_id=graph_id,
                query=str(args.query),
                top_k_chunks=8,
                max_inferences=3,
                create_synthesis=False,
            )

            output = {
                "ok": True,
                "llm_provider": bundle.llm_provider.provider_id(),
                "llm_model_id": bundle.llm_provider.model_id(),
                "chat_preview": chat_text[:120],
                "structured_payload": structured_payload,
                "vector_replay_applied_ops": replay_stats.applied_ops,
                "retrieval_count": len(result.retrieval),
                "inference_count": len(result.inferences),
                "llm_payload_keys": sorted(list(result.llm_payload.keys())),
                "sample_reasoning_source": (
                    str(result.inferences[0].provenance.get("reasoning_source"))
                    if result.inferences
                    else None
                ),
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 0
        except Exception as error:  # noqa: BLE001
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": str(error),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1
        finally:
            for provider in [
                bundle.blob_store,
                bundle.meta_store,
                bundle.vector_index,
                bundle.job_queue,
                bundle.extractor,
                bundle.embedding_provider,
                bundle.llm_provider,
            ]:
                closer = getattr(provider, "close", None)
                if callable(closer):
                    closer()


if __name__ == "__main__":
    raise SystemExit(main())

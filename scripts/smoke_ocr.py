from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from smart_journal.providers import BasicExtractorV1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run OCR smoke check through BasicExtractorV1 with PP-OCRv5 profiles "
            "and print JSON report."
        ),
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help=(
            "Path to image file. Can be passed multiple times. "
            "If omitted, the script tries to generate a synthetic sample via Pillow."
        ),
    )
    parser.add_argument(
        "--ocr-backend",
        default="ppocr_v5",
        help="OCR backend (default: ppocr_v5).",
    )
    parser.add_argument(
        "--ocr-profile",
        default="mobile_optional",
        help="OCR profile (default: mobile_optional).",
    )
    parser.add_argument(
        "--ocr-device",
        default="cpu",
        help="OCR device hint, e.g. cpu or gpu:0 (default: cpu).",
    )
    parser.add_argument(
        "--ocr-languages",
        default="",
        help="Comma-separated OCR language hints, e.g. en,ru. Empty means default mode.",
    )
    parser.add_argument(
        "--ocr-strict-language",
        action="store_true",
        help="Fail OCR when an unsupported language hint is provided.",
    )
    parser.add_argument(
        "--sample-text",
        default="SMART JOURNAL OCR 2026",
        help="Text used for generated synthetic sample when no --image is passed.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    image_paths = [Path(raw).expanduser().resolve() for raw in args.image]

    if not image_paths:
        generated_path, generation_error = _create_synthetic_image(str(args.sample_text))
        if generation_error is not None:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": (
                            "No --image provided and synthetic sample generation failed. "
                            f"{generation_error}"
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        assert generated_path is not None
        image_paths = [generated_path]

    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "Some image paths do not exist.",
                    "missing": missing,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    ocr_languages = _parse_languages(str(args.ocr_languages))
    extractor = BasicExtractorV1(
        {
            "enable_image_ocr": True,
            "enable_audio_asr": False,
            "ocr_backend": str(args.ocr_backend),
            "ocr_profile": str(args.ocr_profile),
            "ocr_device": str(args.ocr_device),
            "ocr_languages": ocr_languages,
            "ocr_strict_language": bool(args.ocr_strict_language),
        }
    )

    sample_reports: list[dict[str, Any]] = []
    temp_files_to_cleanup: list[Path] = []
    for path in image_paths:
        if path.name.startswith("smoke_ocr_sample_"):
            temp_files_to_cleanup.append(path)
        sample_reports.append(_run_one_image(extractor=extractor, path=path))

    try:
        statuses = [str(sample.get("ocr_status", "")) for sample in sample_reports]
        ok_count = sum(1 for status in statuses if status == "ok")
        summary = {
            "total": len(sample_reports),
            "ok": ok_count,
            "error": sum(1 for status in statuses if status == "error"),
            "unavailable": sum(1 for status in statuses if status == "unavailable"),
            "disabled": sum(1 for status in statuses if status == "disabled"),
        }
        output = {
            "ok": ok_count > 0,
            "extractor": {
                "provider_id": extractor.provider_id(),
                "version": extractor.version(),
                "capabilities": dict(extractor.capabilities()),
            },
            "active_ocr_profile": _safe_mapping(extractor.get_active_ocr_profile()),
            "samples": sample_reports,
            "summary": summary,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0 if ok_count > 0 else 1
    finally:
        for path in temp_files_to_cleanup:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


def _run_one_image(*, extractor: BasicExtractorV1, path: Path) -> dict[str, Any]:
    mime_type = _guess_mime_type(path)
    payload = path.read_bytes()
    artifact = extractor.extract(payload, mime_type=mime_type)
    metadata = dict(artifact.metadata or {})
    text = (artifact.text or "").strip()
    return {
        "path": str(path),
        "mime_type": mime_type,
        "content_type": artifact.content_type,
        "ocr_status": str(metadata.get("ocr_status", "")),
        "ocr_backend": str(metadata.get("ocr_backend", "")),
        "text_length": len(text),
        "text_preview": text[:240],
        "metadata": metadata,
    }


def _guess_mime_type(path: Path) -> str:
    suffix = path.suffix.strip().lower()
    if suffix in {".png"}:
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix in {".webp"}:
        return "image/webp"
    return "image/png"


def _parse_languages(raw: str) -> list[str]:
    normalized = [part.strip() for part in raw.split(",")]
    return [part for part in normalized if part]


def _create_synthetic_image(text: str) -> tuple[Path | None, str | None]:
    try:
        image_module = __import__("PIL.Image", fromlist=["Image"])
        image_draw_module = __import__("PIL.ImageDraw", fromlist=["ImageDraw"])
        image_font_module = __import__("PIL.ImageFont", fromlist=["ImageFont"])
    except Exception as error:  # noqa: BLE001
        return None, f"Pillow is not available: {error}"

    try:
        image = image_module.new("RGB", (1280, 320), color=(255, 255, 255))
        draw = image_draw_module.Draw(image)
        try:
            font = image_font_module.truetype("arial.ttf", 56)
        except Exception:
            font = image_font_module.load_default()

        draw.text((48, 72), text, fill=(0, 0, 0), font=font)
        draw.text((48, 160), "profile=mobile_optional", fill=(0, 0, 0), font=font)

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".png",
            delete=False,
            prefix="smoke_ocr_sample_",
        ) as temp_file:
            path = Path(temp_file.name)
        image.save(path, format="PNG")
        return path, None
    except Exception as error:  # noqa: BLE001
        return None, f"Failed to generate synthetic sample: {error}"


def _safe_mapping(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {str(key): value for key, value in dict(payload).items()}
    return {str(key): value for key, value in payload.items()}


if __name__ == "__main__":
    raise SystemExit(main())

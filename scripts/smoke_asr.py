from __future__ import annotations

import argparse
import json
import math
import os
import struct
import tempfile
import wave
from pathlib import Path
from typing import Any

from smart_journal.providers import BasicExtractorV1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ASR smoke check through BasicExtractorV1 (Whisper backend) "
            "and print JSON report."
        ),
    )
    parser.add_argument(
        "--audio",
        action="append",
        default=[],
        help=(
            "Path to audio file (.wav/.mp3/.ogg). Can be passed multiple times. "
            "If omitted, the script generates a synthetic WAV sample."
        ),
    )
    parser.add_argument(
        "--asr-model",
        default="small",
        help="Whisper model id (default: small).",
    )
    parser.add_argument(
        "--asr-languages",
        default="",
        help="Comma-separated language hints, e.g. en,ru. Empty means auto-detect.",
    )
    parser.add_argument(
        "--asr-device",
        default="",
        help="ASR device hint (e.g. cpu, cuda). Empty means model default.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=2.0,
        help="Duration of generated sample in seconds when --audio is omitted.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Sample rate for generated WAV sample (default: 16000).",
    )
    parser.add_argument(
        "--tone-hz",
        type=float,
        default=220.0,
        help="Sine tone frequency for generated sample (default: 220Hz).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    runtime_env = _ensure_asr_runtime_env()
    audio_paths = [Path(raw).expanduser().resolve() for raw in args.audio]

    if not audio_paths:
        generated_path, generation_error = _create_synthetic_audio(
            duration_seconds=float(args.duration_seconds),
            sample_rate=int(args.sample_rate),
            tone_hz=float(args.tone_hz),
        )
        if generation_error is not None:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": (
                            "No --audio provided and synthetic sample generation failed. "
                            f"{generation_error}"
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        assert generated_path is not None
        audio_paths = [generated_path]

    missing = [str(path) for path in audio_paths if not path.exists()]
    if missing:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "Some audio paths do not exist.",
                    "missing": missing,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    asr_languages = _parse_languages(str(args.asr_languages))
    options: dict[str, Any] = {
        "enable_image_ocr": False,
        "enable_audio_asr": True,
        "asr_model": str(args.asr_model),
        "asr_languages": asr_languages,
    }
    asr_device = str(args.asr_device).strip()
    if asr_device:
        options["asr_device"] = asr_device
    extractor = BasicExtractorV1(options)

    sample_reports: list[dict[str, Any]] = []
    temp_files_to_cleanup: list[Path] = []
    for path in audio_paths:
        if path.name.startswith("smoke_asr_sample_"):
            temp_files_to_cleanup.append(path)
        sample_reports.append(_run_one_audio(extractor=extractor, path=path))

    try:
        statuses = [str(sample.get("asr_status", "")) for sample in sample_reports]
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
            "runtime_env": runtime_env,
            "extractor": {
                "provider_id": extractor.provider_id(),
                "version": extractor.version(),
                "capabilities": dict(extractor.capabilities()),
            },
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


def _run_one_audio(*, extractor: BasicExtractorV1, path: Path) -> dict[str, Any]:
    mime_type = _guess_mime_type(path)
    payload = path.read_bytes()
    artifact = extractor.extract(payload, mime_type=mime_type)
    metadata = dict(artifact.metadata or {})
    text = (artifact.text or "").strip()
    return {
        "path": str(path),
        "mime_type": mime_type,
        "content_type": artifact.content_type,
        "asr_status": str(metadata.get("asr_status", "")),
        "asr_backend": str(metadata.get("asr_backend", "")),
        "text_length": len(text),
        "text_preview": text[:240],
        "metadata": metadata,
    }


def _guess_mime_type(path: Path) -> str:
    suffix = path.suffix.strip().lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".ogg":
        return "audio/ogg"
    return "audio/wav"


def _parse_languages(raw: str) -> list[str]:
    normalized = [part.strip() for part in raw.split(",")]
    return [part for part in normalized if part]


def _ensure_asr_runtime_env() -> dict[str, str]:
    runtime_root = Path.cwd() / ".runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    applied: dict[str, str] = {}
    candidates = {
        "XDG_CACHE_HOME": runtime_root / "cache",
        "HF_HOME": runtime_root / "hf",
        "TORCH_HOME": runtime_root / "torch",
    }
    for key, path in candidates.items():
        if os.getenv(key):
            continue
        path.mkdir(parents=True, exist_ok=True)
        value = str(path.resolve())
        os.environ[key] = value
        applied[key] = value
    return applied


def _create_synthetic_audio(
    *,
    duration_seconds: float,
    sample_rate: int,
    tone_hz: float,
) -> tuple[Path | None, str | None]:
    try:
        duration = max(0.3, float(duration_seconds))
        rate = max(8_000, int(sample_rate))
        frequency = max(40.0, float(tone_hz))
        total_samples = int(duration * rate)
        amplitude = 0.2
        frames = bytearray()
        for sample_index in range(total_samples):
            raw = amplitude * math.sin((2.0 * math.pi * frequency * sample_index) / rate)
            value = int(max(-1.0, min(1.0, raw)) * 32767)
            frames.extend(struct.pack("<h", value))
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".wav",
            delete=False,
            prefix="smoke_asr_sample_",
        ) as temp_file:
            path = Path(temp_file.name)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(rate)
            wav_file.writeframes(bytes(frames))
        return path, None
    except Exception as error:  # noqa: BLE001
        return None, f"Failed to generate synthetic audio: {error}"


if __name__ == "__main__":
    raise SystemExit(main())

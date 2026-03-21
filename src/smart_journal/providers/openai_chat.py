from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any


class OpenAIChatProvider:
    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        options = options or {}
        model = str(options.get("model", "gpt-4.1-mini")).strip()
        self._model = model or "gpt-4.1-mini"
        self._context_window = int(options.get("context_window", 128_000))
        self._supports_vision = bool(options.get("supports_vision", True))
        self._timeout_seconds = float(options.get("timeout_seconds", 60.0))
        self._temperature = float(options.get("temperature", 0.0))
        self._api_key = str(options.get("api_key", os.getenv("OPENAI_API_KEY", ""))).strip()
        self._base_url = str(options.get("base_url", os.getenv("OPENAI_BASE_URL", ""))).strip()
        self._organization = str(
            options.get("organization", os.getenv("OPENAI_ORG_ID", ""))
        ).strip()
        self._project = str(options.get("project", os.getenv("OPENAI_PROJECT_ID", ""))).strip()
        self._client: Any | None = options.get("client")

    def provider_id(self) -> str:
        return "openai_chat"

    def version(self) -> str:
        return "0.1.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "supports_vision": self._supports_vision,
            "structured_output": True,
            "tools": True,
            "cloud": True,
        }

    def model_id(self) -> str:
        return self._model

    def context_window(self) -> int:
        return self._context_window

    def supports_vision(self) -> bool:
        return self._supports_vision

    def generate_structured(self, prompt: str, schema: Mapping[str, Any]) -> Mapping[str, Any]:
        schema_json = json.dumps(dict(schema), ensure_ascii=False, sort_keys=True)
        messages = [
            {
                "role": "system",
                "content": (
                    "Return only valid JSON object that matches the requested schema. "
                    "No markdown, no commentary."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Schema:\n"
                    f"{schema_json}\n\n"
                    "Task:\n"
                    f"{prompt}"
                ),
            },
        ]
        response = self._chat_completion(messages=messages, json_mode=True)
        content = _extract_completion_content(response)
        return _parse_json_object(content)

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        if not messages:
            return ""
        response = self._chat_completion(messages=messages, json_mode=False)
        return _extract_completion_content(response)

    def _chat_completion(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        json_mode: bool,
    ) -> Any:
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": _normalize_messages(messages),
            "temperature": self._temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**payload)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        openai_sdk = _import_openai_sdk()
        kwargs: dict[str, Any] = {
            "timeout": self._timeout_seconds,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if self._organization:
            kwargs["organization"] = self._organization
        if self._project:
            kwargs["project"] = self._project
        self._client = openai_sdk.OpenAI(**kwargs)
        return self._client


def _import_openai_sdk() -> Any:
    try:
        return importlib.import_module("openai")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "openai package is not installed. Install it with: python -m pip install openai"
        ) from error


def _normalize_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        content = str(message.get("content", ""))
        normalized.append(
            {
                "role": role,
                "content": content,
            }
        )
    return normalized


def _extract_completion_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, Sequence) or len(choices) == 0:
        raise RuntimeError("OpenAI response does not include choices.")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError("OpenAI response does not include message payload.")
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, str | bytes | bytearray):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                text_value = part.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
                continue
            text_value = getattr(part, "text", None)
            if isinstance(text_value, str):
                parts.append(text_value)
        if parts:
            return "\n".join(parts).strip()
    raise RuntimeError("OpenAI response content is empty.")


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    normalized = raw_text.strip()
    if not normalized:
        return {}
    if normalized.startswith("```"):
        normalized = _strip_markdown_code_fence(normalized)
    try:
        loaded = json.loads(normalized)
    except json.JSONDecodeError:
        loaded = _parse_json_substring(normalized)
    if not isinstance(loaded, dict):
        raise ValueError("Structured LLM response must be a JSON object.")
    return {str(key): value for key, value in loaded.items()}


def _strip_markdown_code_fence(raw_text: str) -> str:
    lines = raw_text.strip().splitlines()
    if not lines:
        return raw_text
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_json_substring(raw_text: str) -> dict[str, Any]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Cannot parse JSON object from LLM response.")
    candidate = raw_text[start : end + 1]
    loaded = json.loads(candidate)
    if not isinstance(loaded, dict):
        raise ValueError("Extracted JSON response is not an object.")
    return {str(key): value for key, value in loaded.items()}

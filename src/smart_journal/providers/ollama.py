from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib import error, request


class OllamaLLMProvider:
    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        options = options or {}
        base_url = str(options.get("base_url", "http://127.0.0.1:11434")).strip().rstrip("/")
        self._base_url = base_url or "http://127.0.0.1:11434"
        self._model = str(options.get("model", "llama3.1:8b-instruct")).strip()
        self._context_window = int(options.get("context_window", 8192))
        self._supports_vision = bool(options.get("supports_vision", False))
        self._timeout_seconds = float(options.get("timeout_seconds", 60.0))
        self._temperature = float(options.get("temperature", 0.0))

    def provider_id(self) -> str:
        return "ollama_chat"

    def version(self) -> str:
        return "0.1.0"

    def capabilities(self) -> Mapping[str, bool | int | float | str | list[str]]:
        return {
            "supports_vision": self._supports_vision,
            "structured_output": True,
            "tools": False,
            "local_runtime": "ollama",
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
        response_payload = self._request_chat(messages=messages, format_json=True)
        response_text = _extract_message_content(response_payload)
        return _parse_json_object(response_text)

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        if not messages:
            return ""
        response_payload = self._request_chat(messages=messages, format_json=False)
        return _extract_message_content(response_payload)

    def _request_chat(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        format_json: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": _normalize_messages(messages),
            "stream": False,
            "options": {
                "temperature": self._temperature,
            },
        }
        if format_json:
            payload["format"] = "json"
        endpoint = f"{self._base_url}/api/chat"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self._timeout_seconds) as response:
                raw = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("Ollama request timed out.") from exc

        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama response is not valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Ollama response payload must be a JSON object.")
        return {str(key): value for key, value in parsed.items()}


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


def _extract_message_content(payload: Mapping[str, Any]) -> str:
    message = payload.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str):
            return content
    response = payload.get("response")
    if isinstance(response, str):
        return response
    raise RuntimeError("Ollama response does not include assistant content.")


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

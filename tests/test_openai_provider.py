from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from smart_journal.providers.openai_chat import OpenAIChatProvider


class OpenAIProviderTests(unittest.TestCase):
    def test_chat_uses_openai_client(self) -> None:
        completions = _FakeCompletions([_response_with_content("pong")])
        client = _fake_client(completions)
        provider = OpenAIChatProvider(
            {
                "client": client,
                "model": "gpt-test",
                "temperature": 0.25,
            }
        )

        result = provider.chat([{"role": "user", "content": "ping"}])

        self.assertEqual(result, "pong")
        self.assertEqual(len(completions.calls), 1)
        call = completions.calls[0]
        self.assertEqual(str(call["model"]), "gpt-test")
        self.assertEqual(call["messages"], [{"role": "user", "content": "ping"}])
        self.assertNotIn("response_format", call)

    def test_generate_structured_requests_json_mode_and_parses_payload(self) -> None:
        completions = _FakeCompletions(
            [_response_with_content("```json\n{\"implications\": [], \"notes\": \"ok\"}\n```")]
        )
        client = _fake_client(completions)
        provider = OpenAIChatProvider({"client": client, "model": "gpt-test"})

        result = provider.generate_structured(
            prompt="build implications",
            schema={"implications": "array", "notes": "string"},
        )

        self.assertEqual(result, {"implications": [], "notes": "ok"})
        self.assertEqual(len(completions.calls), 1)
        call = completions.calls[0]
        self.assertEqual(str(call["response_format"]["type"]), "json_object")
        self.assertGreaterEqual(len(call["messages"]), 2)
        self.assertEqual(str(call["messages"][0]["role"]), "system")

    def test_runtime_error_when_openai_sdk_is_missing(self) -> None:
        provider = OpenAIChatProvider({"model": "gpt-test"})
        with patch(
            "smart_journal.providers.openai_chat.importlib.import_module",
            side_effect=ModuleNotFoundError("openai"),
        ):
            with self.assertRaisesRegex(RuntimeError, "openai package is not installed"):
                provider.chat([{"role": "user", "content": "hello"}])


class _FakeCompletions:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        if self._responses:
            return self._responses.pop(0)
        return _response_with_content("")


def _fake_client(completions: _FakeCompletions) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _response_with_content(content: object) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from smart_journal.providers.ollama import OllamaLLMProvider


class OllamaProviderTests(unittest.TestCase):
    def test_chat_sends_messages_and_returns_content(self) -> None:
        with _OllamaStubServer(
            responses=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "ready",
                    }
                }
            ]
        ) as server:
            provider = OllamaLLMProvider(
                {
                    "base_url": server.base_url,
                    "model": "test-model",
                }
            )
            result = provider.chat([{"role": "user", "content": "ping"}])

            self.assertEqual(result, "ready")
            self.assertEqual(len(server.requests), 1)
            request_payload = server.requests[0]
            self.assertEqual(str(request_payload["model"]), "test-model")
            self.assertEqual(request_payload["messages"], [{"role": "user", "content": "ping"}])
            self.assertNotIn("format", request_payload)

    def test_generate_structured_uses_json_mode_and_parses_markdown_json(self) -> None:
        with _OllamaStubServer(
            responses=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "```json\n{\"implications\": [], \"notes\": \"ok\"}\n```",
                    }
                }
            ]
        ) as server:
            provider = OllamaLLMProvider(
                {
                    "base_url": server.base_url,
                    "model": "structured-model",
                }
            )
            result = provider.generate_structured(
                prompt="build implications",
                schema={"implications": "array", "notes": "string"},
            )

            self.assertEqual(
                result,
                {
                    "implications": [],
                    "notes": "ok",
                },
            )
            self.assertEqual(len(server.requests), 1)
            request_payload = server.requests[0]
            self.assertEqual(str(request_payload["model"]), "structured-model")
            self.assertEqual(str(request_payload["format"]), "json")
            self.assertGreaterEqual(len(request_payload["messages"]), 2)


class _OllamaStubServer:
    def __init__(self, *, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.base_url = ""
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _OllamaStubServer:
        responses = self._responses
        captured_requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                _ = format
                _ = args

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/api/chat":
                    self.send_response(404)
                    self.end_headers()
                    return
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length)
                captured_requests.append(json.loads(raw_body.decode("utf-8")))
                payload = responses.pop(0) if responses else {"message": {"content": ""}}
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.base_url = f"http://127.0.0.1:{self._server.server_port}"
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        _ = exc_type
        _ = exc
        _ = tb
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()

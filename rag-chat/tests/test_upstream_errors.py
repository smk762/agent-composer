"""upstream_error: a 4xx relays the service's own message, a 5xx a fixed one (info-leak guard)."""

import unittest

import httpx

from app.upstream_errors import upstream_error


class UpstreamErrorTests(unittest.TestCase):
    def test_client_errors_relay_the_message_capped(self) -> None:
        e = upstream_error(httpx.Response(422, text="unknown voice 'x'"), "TTS server")
        self.assertEqual((e.status_code, e.detail), (422, "unknown voice 'x'"))
        self.assertEqual(len(upstream_error(httpx.Response(409, text="b" * 5000), "x").detail), 500)

    def test_server_errors_keep_the_status_but_not_the_body(self) -> None:
        body = "Traceback ... File /opt/app/server.py, http://10.0.0.7:11434 connection refused"
        with self.assertLogs("rag-chat.upstream", "WARNING") as logs:
            e = upstream_error(httpx.Response(503, text=body), "Ollama")
        self.assertEqual((e.status_code, e.detail), (503, "Ollama failed (503)"))
        self.assertIn("10.0.0.7", logs.output[0])  # the operator still gets it


if __name__ == "__main__":
    unittest.main()

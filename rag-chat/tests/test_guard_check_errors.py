"""/api/guard/check maps Ollama transport failures to 503/504, not an unhandled 500. No network."""

import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import guard


class GuardTransportErrorTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(guard.router)
        self.client = TestClient(app)

    def _check_with(self, exc: Exception):
        async def failing_generate(*a, **kw):
            raise exc

        with patch.object(guard, "_generate", failing_generate):
            return self.client.post("/api/guard/check", json={"input": "hello", "thinking": "fast"})

    def test_timeout_is_a_504(self) -> None:
        r = self._check_with(httpx.ReadTimeout("slow"))
        self.assertEqual(r.status_code, 504)
        self.assertIn("timed out", r.json()["detail"])

    def test_dropped_connection_is_a_503(self) -> None:
        r = self._check_with(httpx.RemoteProtocolError("restarted"))
        self.assertEqual(r.status_code, 503)

    def test_refused_connection_is_still_a_503(self) -> None:
        r = self._check_with(httpx.ConnectError("down"))
        self.assertEqual(r.status_code, 503)


if __name__ == "__main__":
    unittest.main()

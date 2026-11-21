"""/api/guard/check reports guard-side failures as upstream errors, never as the caller's 4xx. No network."""

import json
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import guard


def _status_error(code: int, body: dict) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://ollama/api/generate")
    resp = httpx.Response(code, request=req, content=json.dumps(body).encode())
    return httpx.HTTPStatusError("upstream", request=req, response=resp)


class GuardUpstreamStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(guard.router)
        self.client = TestClient(app)

    def _check_with(self, exc: Exception, **body):
        async def failing_generate(*a, **kw):
            raise exc

        with patch.object(guard, "_generate", failing_generate):
            return self.client.post("/api/guard/check", json={"input": "hello", "thinking": "fast", **body})

    def test_ollama_404_is_a_502_not_a_404(self) -> None:
        r = self._check_with(_status_error(404, {"error": "model 'x' not found"}))
        self.assertEqual(r.status_code, 502)
        self.assertIn("model 'x' not found", r.json()["detail"])

    def test_ollama_401_is_a_502_not_an_auth_failure(self) -> None:
        self.assertEqual(self._check_with(_status_error(401, {"error": "nope"})).status_code, 502)

    def test_ollama_500_is_a_503(self) -> None:
        self.assertEqual(self._check_with(_status_error(500, {"error": "boom"})).status_code, 503)

    def test_undecodable_ollama_body_is_a_502_not_a_400(self) -> None:
        r = self._check_with(json.JSONDecodeError("Expecting value", "<html>", 0))
        self.assertEqual(r.status_code, 502)
        self.assertNotIn("Expecting value", r.json()["detail"])

    def test_bad_input_is_still_a_400(self) -> None:
        r = self.client.post("/api/guard/check", json={"input": "hi", "thinking": "bogus"})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()

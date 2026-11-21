"""Voice clone import/list/download: upstream timeouts and dropped connections are gateway
errors (504/503), and a non-JSON 2xx is a 502, never an unhandled 500. No network."""

import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import voice

BASE = "http://tts:8031"


def _client_raising(exc=None, response=None):
    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *e):
            return False

        async def _go(self, url, **kw):
            if exc is not None:
                raise exc
            return response

        get = post = _go

    return _Client


class CloneTransportErrorTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(voice.router)
        self.client = TestClient(app, raise_server_exceptions=False)
        for p in (
            patch.object(voice, "TTS_ENGINES", {"xtts": BASE}),
            patch.object(voice, "_tts_base", lambda engine: BASE),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _import(self, client_cls):
        with patch.object(voice.httpx, "AsyncClient", client_cls):
            return self.client.post(
                "/api/voice/clones/import", files={"archive": ("v.zip", b"PK", "application/zip")}
            )

    def test_import_timeout_is_a_504(self) -> None:
        self.assertEqual(self._import(_client_raising(httpx.ReadTimeout("slow"))).status_code, 504)

    def test_import_dropped_connection_is_a_503(self) -> None:
        self.assertEqual(self._import(_client_raising(httpx.RemoteProtocolError("closed"))).status_code, 503)

    def test_import_non_json_success_is_a_502(self) -> None:
        resp = httpx.Response(200, text="<html>ok</html>", request=httpx.Request("POST", BASE))
        self.assertEqual(self._import(_client_raising(response=resp)).status_code, 502)

    def test_list_timeout_is_a_504(self) -> None:
        with patch.object(voice.httpx, "AsyncClient", _client_raising(httpx.ReadTimeout("slow"))):
            self.assertEqual(self.client.get("/api/voice/clones").status_code, 504)

    def test_download_timeout_is_a_504(self) -> None:
        with patch.object(voice.httpx, "AsyncClient", _client_raising(httpx.ReadTimeout("slow"))):
            self.assertEqual(self.client.get("/api/voice/clones/abc/download").status_code, 504)


if __name__ == "__main__":
    unittest.main()

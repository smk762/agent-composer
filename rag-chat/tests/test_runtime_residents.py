"""GPU-resident gateway (/api/runtime/residents) and the OLLAMA_REQUEST_KEEP_ALIVE=server switch."""

import importlib
import os
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import auth
from app.db import get_db
from app.routers import runtime


class _Upstream:
    """Fake httpx.AsyncClient recording requests and answering from a table."""

    calls: list = []
    answers: dict = {}

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def request(self, method, url, headers=None):
        type(self).calls.append((method, url, headers or {}))
        answer = type(self).answers.get(url)
        if answer is None:
            raise httpx.ConnectError("down")
        status, body = answer
        return httpx.Response(status, json=body)


class ResidentGatewayTest(unittest.TestCase):
    def setUp(self) -> None:
        async def no_db():
            yield None

        app = FastAPI()
        app.include_router(runtime.router)
        app.dependency_overrides[get_db] = no_db
        self.client = TestClient(app)
        _Upstream.calls = []
        _Upstream.answers = {}
        for p in (
            patch.object(auth, "DEV_AUTH_BYPASS", True),
            patch.object(runtime.httpx, "AsyncClient", _Upstream),
            patch.object(runtime, "MODERNBERT_URL", "http://mb:7998"),
            patch.object(runtime, "WHISPER_URL", "http://whisper:8030"),
            patch.object(runtime, "INFINITY_URL", "http://inf:7997"),
            patch.object(runtime, "TTS_ENGINES", {"xtts": "http://xtts:8031"}),
            patch.object(runtime, "ARGUS_ADMIN_TOKEN", ""),
        ):
            p.start()
            self.addCleanup(p.stop)

    def test_requires_auth(self) -> None:
        with patch.object(auth, "DEV_AUTH_BYPASS", False):
            self.assertEqual(self.client.get("/api/runtime/residents").status_code, 401)
            self.assertEqual(self.client.post("/api/runtime/residents/xtts/admin/unload").status_code, 401)
        self.assertEqual(_Upstream.calls, [])

    def test_unknown_resident_and_action(self) -> None:
        self.assertEqual(self.client.get("/api/runtime/residents/nope/health").status_code, 404)
        self.assertEqual(self.client.post("/api/runtime/residents/xtts/admin/explode").status_code, 422)
        self.assertEqual(_Upstream.calls, [])

    def test_admin_relays_status_and_forwards_token(self) -> None:
        _Upstream.answers["http://xtts:8031/admin/unload"] = (409, {"status": "busy", "active": 1})
        with patch.object(runtime, "ARGUS_ADMIN_TOKEN", "s3cret"):
            r = self.client.post("/api/runtime/residents/xtts/admin/unload")
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["status"], "busy")
        method, url, headers = _Upstream.calls[-1]
        self.assertEqual((method, url), ("POST", "http://xtts:8031/admin/unload"))
        self.assertEqual(headers, {"Authorization": "Bearer s3cret"})

    def test_health_is_relayed_without_token(self) -> None:
        _Upstream.answers["http://whisper:8030/health"] = (200, {"gpu": {"loaded": False}})
        with patch.object(runtime, "ARGUS_ADMIN_TOKEN", "s3cret"):
            r = self.client.get("/api/runtime/residents/whisper/health")
        self.assertEqual(r.json(), {"gpu": {"loaded": False}})
        self.assertEqual(_Upstream.calls[-1][2], {})

    def test_list_marks_unreachable(self) -> None:
        _Upstream.answers["http://mb:7998/health"] = (
            200,
            {"model_state": "gpu", "busy": True, "idle_s": None, "gpu": {"loaded": True, "model_state": "gpu"}},
        )
        body = self.client.get("/api/runtime/residents").json()["residents"]
        by_name = {r["name"]: r for r in body}
        self.assertEqual(set(by_name), {"modernbert", "whisper", "infinity", "xtts"})
        self.assertTrue(by_name["modernbert"]["reachable"])
        self.assertEqual(by_name["modernbert"]["gpu"]["model_state"], "gpu")
        self.assertIs(by_name["modernbert"]["busy"], True)
        self.assertFalse(by_name["whisper"]["reachable"])
        self.assertEqual(by_name["whisper"]["status_code"], 503)

    def test_unreachable_admin_is_503(self) -> None:
        self.assertEqual(self.client.post("/api/runtime/residents/modernbert/admin/wake").status_code, 503)


class KeepAliveSwitchTest(unittest.TestCase):
    def _config(self, value, request=None):
        from app import config

        env = {"OLLAMA_KEEP_ALIVE": value} if value is not None else {}
        if request is not None:
            env["OLLAMA_REQUEST_KEEP_ALIVE"] = request
        with patch.dict(os.environ, env, clear=False):
            if value is None:
                os.environ.pop("OLLAMA_KEEP_ALIVE", None)
            if request is None:
                os.environ.pop("OLLAMA_REQUEST_KEEP_ALIVE", None)
            return importlib.reload(config)

    def tearDown(self) -> None:
        from app import config

        importlib.reload(config)

    def test_default_sends_15m(self) -> None:
        self.assertEqual(self._config(None).ollama_keep_alive(), {"keep_alive": "15m"})

    def test_server_and_empty_send_nothing(self) -> None:
        for value in ("server", "", "none", "OFF"):
            with self.subTest(value=value):
                cfg = self._config(value)
                self.assertEqual(cfg.ollama_keep_alive(), {})
                self.assertEqual(cfg.OLLAMA_KEEP_ALIVE, "")

    def test_explicit_value_passes_through(self) -> None:
        self.assertEqual(self._config("-1").ollama_keep_alive(), {"keep_alive": "-1"})

    def test_request_variable_wins_over_the_daemon_default(self) -> None:
        """OLLAMA_KEEP_ALIVE stays the daemon's duration; the clients send none (#53)."""
        self.assertEqual(self._config("15m", request="server").ollama_keep_alive(), {})
        self.assertEqual(self._config("15m", request="30m").ollama_keep_alive(), {"keep_alive": "30m"})
        self.assertEqual(self._config(None, request="-1").ollama_keep_alive(), {"keep_alive": "-1"})

    def test_empty_request_variable_falls_back(self) -> None:
        # As compose's ${OLLAMA_REQUEST_KEEP_ALIVE:-...} does.
        self.assertEqual(self._config("20m", request=" ").ollama_keep_alive(), {"keep_alive": "20m"})


if __name__ == "__main__":
    unittest.main()


class ExtraResidentsTest(unittest.TestCase):
    """RUNTIME_EXTRA_RESIDENTS adds gateway-only residents (the Qwen tts sidecar), not TTS engines."""

    def test_parse(self) -> None:
        from app.config import _parse_residents

        self.assertEqual(
            _parse_residents(" tts=http://tts:8031 , ,bad, x=ftp://y, Foo=https://foo:1 "),
            {"tts": "http://tts:8031", "foo": "https://foo:1"},
        )
        self.assertEqual(_parse_residents(""), {})

    def test_extras_join_the_gateway_and_engines_win_on_a_clash(self) -> None:
        extras = {"tts": "http://tts:8031/", "xtts": "http://elsewhere:1"}
        with patch.object(runtime, "RUNTIME_EXTRA_RESIDENTS", extras), \
                patch.object(runtime, "TTS_ENGINES", {"xtts": "http://xtts:8031"}):
            found = runtime._residents()
        self.assertEqual(found["tts"], "http://tts:8031")
        self.assertEqual(found["xtts"], "http://xtts:8031")

    def test_env_reaches_the_gateway_but_not_the_engines(self) -> None:
        from app import config

        env = {"RUNTIME_EXTRA_RESIDENTS": "tts=http://tts:8031", "TTS_URL": "http://xtts:8031",
               "XTTS_TTS_URL": "http://xtts:8031"}
        self.addCleanup(importlib.reload, config)
        with patch.dict(os.environ, env):
            reloaded = importlib.reload(config)
        self.assertEqual(reloaded.RUNTIME_EXTRA_RESIDENTS, {"tts": "http://tts:8031"})
        self.assertNotIn("tts", reloaded.TTS_ENGINES)

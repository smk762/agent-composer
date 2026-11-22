"""GPU-resident gateway (/api/runtime/residents) and the OLLAMA_REQUEST_KEEP_ALIVE=server switch."""

import importlib
import os
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI, HTTPException
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
            patch.object(runtime, "INFINITY_URL", "http://inf:7997"),
            patch.object(runtime, "ARGUS_ADMIN_TOKEN", ""),
        ):
            p.start()
            self.addCleanup(p.stop)

    def test_requires_auth(self) -> None:
        with patch.object(auth, "DEV_AUTH_BYPASS", False):
            self.assertEqual(self.client.get("/api/runtime/residents").status_code, 401)
            self.assertEqual(self.client.post("/api/runtime/residents/infinity/admin/unload").status_code, 401)
        self.assertEqual(_Upstream.calls, [])

    def test_unknown_resident_and_action(self) -> None:
        self.assertEqual(self.client.get("/api/runtime/residents/nope/health").status_code, 404)
        self.assertEqual(self.client.post("/api/runtime/residents/infinity/admin/explode").status_code, 422)
        self.assertEqual(_Upstream.calls, [])

    def test_admin_relays_status_and_forwards_token(self) -> None:
        _Upstream.answers["http://inf:7997/admin/unload"] = (409, {"status": "busy", "active": 1})
        with patch.object(runtime, "ARGUS_ADMIN_TOKEN", "s3cret"):
            r = self.client.post("/api/runtime/residents/infinity/admin/unload")
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["status"], "busy")
        method, url, headers = _Upstream.calls[-1]
        self.assertEqual((method, url), ("POST", "http://inf:7997/admin/unload"))
        self.assertEqual(headers, {"Authorization": "Bearer s3cret"})

    def test_health_is_relayed_without_token(self) -> None:
        _Upstream.answers["http://inf:7997/health"] = (200, {"gpu": {"loaded": False}})
        with patch.object(runtime, "ARGUS_ADMIN_TOKEN", "s3cret"):
            r = self.client.get("/api/runtime/residents/infinity/health")
        self.assertEqual(r.json(), {"gpu": {"loaded": False}})
        self.assertEqual(_Upstream.calls[-1][2], {})

    def test_list_marks_unreachable(self) -> None:
        _Upstream.answers["http://mb:7998/health"] = (
            200,
            {"model_state": "gpu", "busy": True, "idle_s": None, "gpu": {"loaded": True, "model_state": "gpu"}},
        )
        body = self.client.get("/api/runtime/residents").json()["residents"]
        by_name = {r["name"]: r for r in body}
        self.assertEqual(set(by_name), {"modernbert", "infinity"})
        self.assertTrue(by_name["modernbert"]["reachable"])
        self.assertEqual(by_name["modernbert"]["gpu"]["model_state"], "gpu")
        self.assertIs(by_name["modernbert"]["busy"], True)
        self.assertFalse(by_name["infinity"]["reachable"])
        self.assertEqual(by_name["infinity"]["status_code"], 503)

    def test_a_failing_sidecar_is_reachable_but_not_healthy(self) -> None:
        _Upstream.answers["http://mb:7998/health"] = (503, {"detail": "model load failed"})
        _Upstream.answers["http://inf:7997/health"] = (200, {"model_state": "gpu"})
        by_name = {r["name"]: r for r in self.client.get("/api/runtime/residents").json()["residents"]}
        mb = by_name["modernbert"]
        self.assertEqual((mb["reachable"], mb["healthy"], mb["status_code"]), (True, False, 503))
        self.assertEqual(mb["detail"], "model load failed")
        self.assertTrue(by_name["infinity"]["healthy"])

    def test_a_non_string_detail_does_not_break_the_listing(self) -> None:
        _Upstream.answers["http://mb:7998/health"] = (503, {"detail": {"code": "oom", "free_mb": 12}})
        _Upstream.answers["http://inf:7997/health"] = (500, {"error": 7})
        r = self.client.get("/api/runtime/residents")
        self.assertEqual(r.status_code, 200)
        by_name = {x["name"]: x for x in r.json()["residents"]}
        self.assertEqual(by_name["modernbert"]["detail"], '{"code": "oom", "free_mb": 12}')
        self.assertEqual(by_name["infinity"]["detail"], "7")

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
    """RUNTIME_EXTRA_RESIDENTS adds gateway-only residents (any name=url), not TTS engines."""

    def test_parse(self) -> None:
        from app.config import _parse_residents

        self.assertEqual(
            _parse_residents(" tts=http://tts:8031 , ,bad, x=ftp://y, Foo=https://foo:1 "),
            {"tts": "http://tts:8031", "foo": "https://foo:1"},
        )
        self.assertEqual(_parse_residents(""), {})

    def test_extras_join_the_gateway_and_backends_win_on_a_clash(self) -> None:
        extras = {"tts": "http://tts:8031/", "modernbert": "http://elsewhere:1"}
        with patch.object(runtime, "RUNTIME_EXTRA_RESIDENTS", extras), \
                patch.object(runtime, "MODERNBERT_URL", "http://mb:7998"):
            found = runtime._residents()
        self.assertEqual(found["tts"], "http://tts:8031")
        self.assertEqual(found["modernbert"], "http://mb:7998")

    def test_voice_engines_are_echos_residents_not_the_gateways(self) -> None:
        # #81: iris reaches argus-echo's engines directly; rag-chat only calls them by URL.
        found = runtime._residents()
        self.assertFalse({"xtts", "whisper", "miso", "tts"} & set(found), found)

    def test_an_unset_backend_does_not_drop_an_extra_resident(self) -> None:
        with patch.object(runtime, "RUNTIME_EXTRA_RESIDENTS", {"modernbert": "http://mb-elsewhere:7998"}), \
                patch.object(runtime, "MODERNBERT_URL", ""):
            found = runtime._residents()
        self.assertEqual(found["modernbert"], "http://mb-elsewhere:7998")

    def test_env_reaches_the_gateway_but_not_the_engines(self) -> None:
        from app import config

        env = {"RUNTIME_EXTRA_RESIDENTS": "tts=http://tts:8031", "TTS_URL": "http://xtts:8031",
               "XTTS_TTS_URL": "http://xtts:8031"}
        self.addCleanup(importlib.reload, config)
        with patch.dict(os.environ, env):
            reloaded = importlib.reload(config)
        self.assertEqual(reloaded.RUNTIME_EXTRA_RESIDENTS, {"tts": "http://tts:8031"})
        self.assertNotIn("tts", reloaded.TTS_ENGINES)


class OllamaEvictTest(unittest.TestCase):
    """POST /api/runtime/ollama/evict only unloads models Ollama reports as loaded."""

    def setUp(self) -> None:
        async def no_db():
            yield None

        app = FastAPI()
        app.include_router(runtime.router)
        app.dependency_overrides[get_db] = no_db
        self.client = TestClient(app)
        self.generated: list = []

        async def fake_get(path, *, timeout=30):
            assert path == "/api/ps"
            return {"models": [{"name": "qwen2.5:7b"}, {"name": "llama3:latest"}]}

        async def fake_post(path, payload, *, timeout=None):
            self.generated.append(payload)
            return {}

        for p in (
            patch.object(auth, "DEV_AUTH_BYPASS", True),
            patch.object(runtime, "_ollama_get", fake_get),
            patch.object(runtime, "_ollama_post", fake_post),
        ):
            p.start()
            self.addCleanup(p.stop)

    def test_a_model_that_is_not_loaded_is_never_loaded_to_evict_it(self) -> None:
        r = self.client.post("/api/runtime/ollama/evict", json={"model": "mistral:large"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"state": "already_idle", "evicted": []})
        self.assertEqual(self.generated, [])

    def test_a_loaded_model_is_evicted_by_its_loaded_name(self) -> None:
        r = self.client.post("/api/runtime/ollama/evict", json={"model": "llama3"})
        self.assertEqual(r.json()["evicted"], ["llama3:latest"])
        self.assertEqual(self.generated, [{"model": "llama3:latest", "prompt": "", "stream": False, "keep_alive": 0}])

    def test_no_model_evicts_everything_loaded(self) -> None:
        r = self.client.post("/api/runtime/ollama/evict", json={})
        self.assertEqual(r.json()["evicted"], ["qwen2.5:7b", "llama3:latest"])

    def test_a_failure_partway_reports_what_was_already_evicted(self) -> None:
        async def failing_second(path, payload, *, timeout=None):
            if payload["model"] == "llama3:latest":
                raise HTTPException(503, detail="Ollama unreachable")
            self.generated.append(payload)
            return {}

        with patch.object(runtime, "_ollama_post", failing_second):
            r = self.client.post("/api/runtime/ollama/evict", json={})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json(), {"state": "partial", "evicted": ["qwen2.5:7b"], "failed": "llama3:latest",
                                    "detail": "Ollama unreachable", "loaded": ["llama3:latest"]})

    def test_evicting_one_model_reports_the_others_still_loaded(self) -> None:
        r = self.client.post("/api/runtime/ollama/evict", json={"model": "llama3"})
        self.assertEqual(r.json()["state"], "loaded")
        self.assertEqual(r.json()["loaded"], ["qwen2.5:7b"])

    def test_evicting_everything_reports_idle(self) -> None:
        r = self.client.post("/api/runtime/ollama/evict", json={})
        self.assertEqual(r.json()["state"], "idle")
        self.assertEqual(r.json()["loaded"], [])


class OllamaWakeAndErrorsTest(unittest.TestCase):
    """Wake falls back to CHAT_MODEL for a blank name; transport failures map to 503/504."""

    def setUp(self) -> None:
        async def no_db():
            yield None

        app = FastAPI()
        app.include_router(runtime.router)
        app.dependency_overrides[get_db] = no_db
        self.client = TestClient(app)
        p = patch.object(auth, "DEV_AUTH_BYPASS", True)
        p.start()
        self.addCleanup(p.stop)

    def test_blank_model_wakes_the_chat_model(self) -> None:
        sent: list = []

        async def fake_post(path, payload, *, timeout=None):
            sent.append(payload)
            return {}

        with patch.object(runtime, "_ollama_post", fake_post), patch.object(runtime, "CHAT_MODEL", "qwen2.5:7b"):
            r = self.client.post("/api/runtime/ollama/wake", json={"model": "   "})
        self.assertEqual(r.json()["model"], "qwen2.5:7b")
        self.assertEqual(sent[0]["model"], "qwen2.5:7b")

    def _client_raising(self, exc):
        class _Failing:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *e):
                return False

            async def get(self, *a, **kw):
                raise exc

            post = get

        return patch.object(runtime.httpx, "AsyncClient", _Failing)

    def test_ollama_timeout_is_a_504(self) -> None:
        with self._client_raising(httpx.ReadTimeout("slow")):
            r = self.client.get("/api/runtime/ollama/status")
        self.assertEqual(r.status_code, 504)

    def test_other_transport_errors_are_a_503(self) -> None:
        with self._client_raising(httpx.RemoteProtocolError("dropped")):
            r = self.client.post("/api/runtime/ollama/wake", json={})
        self.assertEqual(r.status_code, 503)

    def test_infinity_timeout_is_a_504(self) -> None:
        with self._client_raising(httpx.ConnectTimeout("slow")):
            r = self.client.post("/api/runtime/infinity/evict")
        self.assertEqual(r.status_code, 504)

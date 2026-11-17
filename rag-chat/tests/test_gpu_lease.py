"""argus-iris leases around Ollama calls (app/gpu_lease.py, #48).

rag-chat and rag-ingest carry identical copies of gpu_lease.py and of this test, except
GuardRouteLeaseTest, which is rag-chat's alone.
"""

import contextlib
import json
import unittest
from unittest.mock import patch

import httpx
from argus_cortex.gpu import GpuLease, GpuLeaseRefused, GpuLeaseTimeout, NullCoordinator
from argus_cortex.gpu_broker import BrokerCoordinator, BrokerError
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import gpu_lease
from app.gpu_lease import OllamaLeaseUnavailable, PriorityMiddleware, ollama_lease
from app.inference_retry import run_with_oom_wait

LOCAL = "http://ollama:11434"


class _FakeCoordinator:
    """Records each lease request; grants *handle* or raises *error* on entry."""

    name = "broker"

    def __init__(self, handle=None, error=None):
        self.handle = handle or GpuLease(via="broker", lease_id="L1", node="dragon", base_url="http://dragon:11434")
        self.error = error
        self.requests = []
        self.exits = []

    @contextlib.asynccontextmanager
    async def alease(self, **kw):
        self.requests.append(kw)
        if self.error:
            raise self.error
        try:
            yield self.handle
        except BaseException as exc:
            self.exits.append(type(exc).__name__)
            raise
        else:
            self.exits.append(None)


class OllamaLeaseTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        for name, value in {
            "OLLAMA_URL": LOCAL,
            "HOME_NODE": "dragon",
            "NODE_URLS": {},
            "PINNED_MODELS": {"singguard:q4", "nomic-embed-text"},
        }.items():
            p = patch.object(gpu_lease, name, value)
            p.start()
            self.addCleanup(p.stop)

    def _use(self, coordinator):
        p = patch.object(gpu_lease, "_coordinator", coordinator)
        p.start()
        self.addCleanup(p.stop)
        return coordinator

    async def test_no_broker_means_local_ollama_and_no_lease(self) -> None:
        self._use(NullCoordinator())
        async with ollama_lease("qwen3:8b", kind="chat", priority="interactive") as url:
            self.assertEqual(url, LOCAL)

    async def test_grant_on_the_home_node_keeps_the_local_url(self) -> None:
        coord = self._use(_FakeCoordinator())
        async with ollama_lease("qwen3:8b", kind="chat", priority="interactive") as url:
            self.assertEqual(url, LOCAL)
        req = coord.requests[0]
        self.assertEqual((req["target"], req["kind"], req["priority"]), ("ollama:qwen3:8b", "chat", "interactive"))
        self.assertIsNone(req["constraints"])
        self.assertFalse(req["preemptible"])
        # Whole footprint plus its weights, so iris reuses a loaded model (#50).
        self.assertEqual((req["min_vram_mb"], req["model_mb"]),
                         (gpu_lease.vram_mb("qwen3:8b"), gpu_lease.model_mb("qwen3:8b")))
        self.assertIsNotNone(req["model_mb"])
        self.assertEqual(coord.exits, [None])

    async def test_spill_uses_the_granted_node(self) -> None:
        spill = GpuLease(via="broker", lease_id="L2", node="jarrah", base_url="http://jarrah:11434/")
        self._use(_FakeCoordinator(spill))
        async with ollama_lease("qwen3:8b", kind="chat", priority="interactive") as url:
            self.assertEqual(url, "http://jarrah:11434")
        with patch.object(gpu_lease, "NODE_URLS", {"jarrah": "http://192.168.1.109:11434"}):
            async with ollama_lease("qwen3:8b", kind="chat", priority="interactive") as url:
                self.assertEqual(url, "http://192.168.1.109:11434")

    async def test_fallback_handle_means_local(self) -> None:
        self._use(_FakeCoordinator(GpuLease(via="fallback")))
        async with ollama_lease("qwen3:8b", kind="chat", priority="interactive") as url:
            self.assertEqual(url, LOCAL)

    async def test_pinned_models_carry_the_home_node(self) -> None:
        coord = self._use(_FakeCoordinator())
        async with ollama_lease("singguard:q4", kind="guard", priority="interactive"):
            pass
        self.assertEqual(coord.requests[0]["constraints"].node, "dragon")

    async def test_header_priority_overrides_the_call_site(self) -> None:
        coord = self._use(_FakeCoordinator())
        token = gpu_lease._priority.set("batch")
        try:
            async with ollama_lease("qwen3:8b", kind="chat", priority="interactive"):
                pass
        finally:
            gpu_lease._priority.reset(token)
        self.assertEqual(coord.requests[0]["priority"], "batch")

    async def test_refusal_and_timeout_are_lease_errors_not_ooms(self) -> None:
        for error, retry_after in ((GpuLeaseRefused("can never fit"), None), (GpuLeaseTimeout("queued"), 30)):
            with self.subTest(error=type(error).__name__):
                self._use(_FakeCoordinator(error=error))
                with self.assertRaises(OllamaLeaseUnavailable) as caught:
                    async with ollama_lease("qwen3:8b", kind="chat", priority="interactive"):
                        self.fail("body must not run")
                self.assertEqual(caught.exception.retry_after, retry_after)
                self.assertNotIsInstance(caught.exception, RuntimeError)

    async def test_oom_loop_does_not_retry_a_lease_error(self) -> None:
        calls = []

        async def refused():
            calls.append(1)
            raise OllamaLeaseUnavailable("CUDA out of memory, says the broker")

        with self.assertRaises(OllamaLeaseUnavailable):
            await run_with_oom_wait(refused, max_wait_s=60, interval_s=1, on_oom=lambda: None)
        self.assertEqual(calls, [1])

    async def test_broker_4xx_falls_back_to_local_ollama(self) -> None:
        self._use(_FakeCoordinator(error=BrokerError("broker refused the request (401)")))
        with self.assertLogs("gpu-lease", "ERROR"):
            async with ollama_lease("qwen3:8b", kind="chat", priority="interactive") as url:
                self.assertEqual(url, LOCAL)

    async def test_a_failure_resolving_the_url_releases_the_lease(self) -> None:
        coord = self._use(_FakeCoordinator())

        def broken(handle):
            raise KeyError(handle.node)

        with patch.object(gpu_lease, "url_for", broken):
            with self.assertRaises(KeyError):
                async with ollama_lease("qwen3:8b", kind="chat", priority="interactive"):
                    self.fail("body must not run")
        self.assertEqual(coord.exits, ["KeyError"])

    async def test_body_errors_propagate_and_release_the_lease(self) -> None:
        coord = self._use(_FakeCoordinator())
        with self.assertRaises(httpx.ConnectError):
            async with ollama_lease("qwen3:8b", kind="chat", priority="interactive"):
                raise httpx.ConnectError("down")
        self.assertEqual(coord.exits, ["ConnectError"])

    async def test_against_cortex_broker_client(self) -> None:
        """The real BrokerCoordinator, with a mock broker that grants on jarrah."""
        seen = []

        def broker(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content or b"{}")
            seen.append((request.method, request.url.path, body))
            if request.url.path == "/leases":
                return httpx.Response(200, json={
                    "lease_version": "1.0", "lease_id": "abc", "state": "granted",
                    "node": "jarrah", "base_url": "http://jarrah:11434", "ttl_s": 60,
                })
            if request.url.path == "/leases/abc/release":
                return httpx.Response(200, json={"lease_version": "1.0", "lease_id": "abc", "state": "released"})
            return httpx.Response(404)

        self._use(BrokerCoordinator(
            "http://morpheus:8120", tenant="agent-composer", token="t",
            async_transport=httpx.MockTransport(broker),
        ))
        async with ollama_lease("qwen3:8b", kind="repair", priority="batch") as url:
            self.assertEqual(url, "http://jarrah:11434")
        method, path, req = seen[0]
        self.assertEqual((method, path), ("POST", "/leases"))
        self.assertEqual((req["target"], req["kind"], req["priority"], req["tenant"]),
                         ("ollama:qwen3:8b", "repair", "batch", "agent-composer"))
        self.assertEqual((req["vram_mb"], req["model_mb"]),
                         (gpu_lease.vram_mb("qwen3:8b"), gpu_lease.model_mb("qwen3:8b")))
        self.assertEqual(seen[-1][1], "/leases/abc/release")
        self.assertEqual(seen[-1][2]["outcome"], "ok")


class VramAndPinsTest(unittest.TestCase):
    def test_footprint_lookup(self) -> None:
        with patch.object(gpu_lease, "VRAM_MB", {"nomic-embed-text": 600, "qwen3": 5200, "qwen3:8b": 6000}), \
                patch.object(gpu_lease, "DEFAULT_VRAM_MB", 4000):
            self.assertEqual(gpu_lease.vram_mb("qwen3:8b"), 6000)
            self.assertEqual(gpu_lease.vram_mb("qwen3:4b"), 5200)
            self.assertEqual(gpu_lease.vram_mb("nomic-embed-text"), 600)
            self.assertEqual(gpu_lease.vram_mb("llama3.2:3b"), 4000)

    def test_model_lookup(self) -> None:
        with patch.object(gpu_lease, "VRAM_MB", {"sg": 6000, "small": 300}), \
                patch.object(gpu_lease, "MODEL_MB", {"sg": 3900, "qwen3:8b": 5300, "small": 500}), \
                patch.object(gpu_lease, "DEFAULT_VRAM_MB", 6000):
            self.assertEqual(gpu_lease.model_mb("sg:q4"), 3900)
            self.assertEqual(gpu_lease.model_mb("qwen3:8b"), 5300)
            self.assertIsNone(gpu_lease.model_mb("qwen3:4b"))
            self.assertEqual(gpu_lease.model_mb("small"), 300)  # never more than the footprint

    def test_defaults_declare_weights_for_the_known_models(self) -> None:
        for model, weights in {"hf.co/inclusionAI/SingGuard-4b-GGUF:Q4_K_M": 3900,
                               "nomic-embed-text": 320, "qwen3:8b": 5300}.items():
            self.assertEqual(gpu_lease._lookup(gpu_lease._DEFAULT_MODEL_MB, model), weights)

    def test_env_overrides_the_tables(self) -> None:
        with patch.dict("os.environ", {"OLLAMA_LEASE_MODEL_MB": "qwen3:8b=5000,bad=x"}):
            table = gpu_lease._mb_table("OLLAMA_LEASE_MODEL_MB", gpu_lease._DEFAULT_MODEL_MB)
        self.assertEqual(table["qwen3:8b"], 5000)
        self.assertEqual(table["nomic-embed-text"], 320)
        self.assertNotIn("bad", table)

    def test_pinned_defaults_follow_guard_and_embed_models(self) -> None:
        env = {"GUARD_MODEL": "sg:q8", "EMBED_MODEL": "emb"}
        with patch.dict("os.environ", env, clear=False):
            import os

            os.environ.pop("OLLAMA_LEASE_PINNED_MODELS", None)
            self.assertEqual(gpu_lease._pinned_models(), {"sg:q8", "emb"})
            with patch.dict("os.environ", {"OLLAMA_LEASE_PINNED_MODELS": ""}):
                self.assertEqual(gpu_lease._pinned_models(), {"sg:q8", "emb"})
            with patch.dict("os.environ", {"OLLAMA_LEASE_PINNED_MODELS": "none"}):
                self.assertEqual(gpu_lease._pinned_models(), set())
            with patch.dict("os.environ", {"OLLAMA_LEASE_PINNED_MODELS": "a, b"}):
                self.assertEqual(gpu_lease._pinned_models(), {"a", "b"})


class PriorityMiddlewareTest(unittest.TestCase):
    def test_header_sets_the_priority_for_the_request(self) -> None:
        app = FastAPI()
        app.add_middleware(PriorityMiddleware)

        @app.get("/p")
        async def p():
            return {"priority": gpu_lease._priority.get()}

        client = TestClient(app)
        self.assertEqual(client.get("/p", headers={"X-Argus-Priority": "Batch"}).json(), {"priority": "batch"})
        self.assertEqual(client.get("/p", headers={"X-Argus-Priority": "urgent"}).json(), {"priority": None})
        self.assertEqual(client.get("/p").json(), {"priority": None})


if __name__ == "__main__":
    unittest.main()


class GuardRouteLeaseTest(unittest.TestCase):
    """A refused lease on /api/guard/check is a 503 with Retry-After, via main's handler."""

    def test_guard_check_answers_503(self) -> None:
        from app.main import app

        with patch.object(gpu_lease, "_coordinator", _FakeCoordinator(error=GpuLeaseTimeout("queued 3 ahead"))):
            r = TestClient(app).post("/api/guard/check", json={"input": "hello", "thinking": "fast"})
        self.assertEqual(r.status_code, 503, r.text)
        self.assertEqual(r.headers.get("retry-after"), "30")
        self.assertIn("no GPU lease for guard", r.json()["detail"])

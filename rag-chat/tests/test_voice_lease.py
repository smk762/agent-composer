"""Voice proxy: XTTS synth runs inside an argus-iris lease on the xtts target (#79).

A refused or timed-out lease moves a default-path request to the next engine; on the
last engine it becomes a CPU run on the local XTTS (XTTS_LEASE_CPU_FALLBACK) or a 503
carrying the broker's reason, never a bare 500. No network: httpx is mocked and
service_lease is replaced by a recorder.
"""

import contextlib
import json
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import auth
from app.db import get_db
from app.gpu_lease import GpuLeaseUnavailable
from app.routers import voice

XTTS = "http://xtts:8031"
MISO = "http://miso-host:8034"
MORPHEUS = "http://192.168.1.39:8141"
_RealAsyncClient = httpx.AsyncClient


class _Leases:
    """Stands in for service_lease: grants *url*, or raises *error*; records each lease."""

    def __init__(self, url=XTTS, error=None):
        self.url, self.error = url, error
        self.calls: list[dict] = []
        self.held = 0
        self.released = 0

    def __call__(self, target, **kw):
        self.calls.append({"target": target, **kw})

        @contextlib.asynccontextmanager
        async def cm():
            if self.error:
                raise self.error
            self.held += 1
            try:
                yield self.url
            finally:
                self.held -= 1
                self.released += 1

        return cm()


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        async def no_db():
            yield None

        app = FastAPI()
        app.include_router(voice.router)
        app.dependency_overrides[get_db] = no_db
        self.client = TestClient(app)
        self.requests: list[httpx.Request] = []
        for p in (
            patch.object(auth, "DEV_AUTH_BYPASS", True),
            patch.object(voice, "TTS_ENGINES", {"xtts": XTTS, "miso": MISO}),
            patch.object(voice, "XTTS_TTS_URL", XTTS),
            patch.object(voice, "VOICE_S3_REHOST", False),
            patch.object(voice, "XTTS_LEASE_CPU_FALLBACK", True),
            patch.object(voice, "tts_engine_chain", lambda e: [("xtts", XTTS)]),
        ):
            p.start()
            self.addCleanup(p.stop)
        self._route(lambda req: httpx.Response(200, json={"audio_url": "", "duration": 1.0}))

    def _route(self, handler):
        seen = self.requests

        def record(req):
            seen.append(req)
            return handler(req)

        class _Client(_RealAsyncClient):
            def __init__(self, *a, **kw):
                kw["transport"] = httpx.MockTransport(record)
                super().__init__(*a, **kw)

        p = patch.object(voice.httpx, "AsyncClient", _Client)
        p.start()
        self.addCleanup(p.stop)

    def _leases(self, **kw) -> _Leases:
        leases = _Leases(**kw)
        p = patch.object(voice, "service_lease", leases)
        p.start()
        self.addCleanup(p.stop)
        return leases

    def _chain(self, chain):
        p = patch.object(voice, "tts_engine_chain", lambda e: chain)
        p.start()
        self.addCleanup(p.stop)

    def _body(self, i=-1) -> dict:
        return json.loads(self.requests[i].content)


REFUSED = GpuLeaseUnavailable("GPU broker refused tts-synth on xtts: never fits on dragon", retry_after=None)
QUEUED = GpuLeaseUnavailable("no GPU lease for tts-synth on xtts: timed out after 60 s")


class SynthLeaseTests(_Base):
    def test_synth_holds_a_lease_and_calls_the_granted_instance(self) -> None:
        leases = self._leases(url=MORPHEUS)
        r = self.client.post("/api/voice/synthesise", json={"text": "hi", "pause_ms": 0})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(str(self.requests[0].url), f"{MORPHEUS}/synthesise")
        call = leases.calls[0]
        self.assertEqual((call["target"], call["kind"], call["priority"], call["home_url"]),
                         ("xtts", "tts-synth", "interactive", XTTS))
        self.assertFalse(call["pin_home"])
        self.assertNotIn("device", self._body())
        self.assertEqual((leases.held, leases.released), (0, 1))

    def test_a_voice_clone_stays_on_the_home_node(self) -> None:
        leases = self._leases()
        self.client.post("/api/voice/synthesise",
                         json={"text": "hi", "pause_ms": 0, "voice_clone_id": "voice_clones/a/prompt.pt"})
        self.assertTrue(leases.calls[0]["pin_home"])

    def test_other_engines_are_not_leased(self) -> None:
        leases = self._leases()
        self._chain([("miso", MISO)])
        r = self.client.post("/api/voice/synthesise", json={"text": "hi", "pause_ms": 0})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(leases.calls, [])

    def test_refusal_on_the_last_engine_runs_on_cpu(self) -> None:
        self._leases(error=REFUSED)
        r = self.client.post("/api/voice/synthesise", json={"text": "hi", "pause_ms": 0})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(str(self.requests[0].url), f"{XTTS}/synthesise")
        self.assertEqual(self._body()["device"], "cpu")

    def test_refusal_without_cpu_fallback_is_a_503_with_the_reason(self) -> None:
        self._leases(error=QUEUED)
        with patch.object(voice, "XTTS_LEASE_CPU_FALLBACK", False):
            r = self.client.post("/api/voice/synthesise", json={"text": "hi", "pause_ms": 0})
        self.assertEqual(r.status_code, 503)
        self.assertIn("timed out after 60 s", r.json()["detail"])
        self.assertEqual(r.headers["retry-after"], "30")
        self.assertEqual(self.requests, [])

    def test_refusal_moves_a_default_request_to_the_next_engine(self) -> None:
        self._leases(error=REFUSED)
        self._chain([("xtts", XTTS), ("miso", MISO)])
        r = self.client.post("/api/voice/synthesise", json={"text": "hi", "pause_ms": 0})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual([str(q.url) for q in self.requests], [f"{MISO}/synthesise"])
        self.assertNotIn("device", self._body())

    def test_refusal_then_unreachable_next_engine_reports_the_refusal(self) -> None:
        self._leases(error=REFUSED)
        self._chain([("xtts", XTTS), ("miso", MISO)])

        def handler(req):
            raise httpx.ConnectError("down", request=req)

        self._route(handler)
        r = self.client.post("/api/voice/synthesise", json={"text": "hi", "pause_ms": 0})
        self.assertEqual(r.status_code, 503)
        self.assertIn("never fits on dragon", r.json()["detail"])
        self.assertNotIn("retry-after", r.headers)

    def test_unreachable_granted_instance_releases_its_lease(self) -> None:
        leases = self._leases(url=MORPHEUS)
        self._chain([("xtts", XTTS), ("miso", MISO)])

        def handler(req):
            if req.url.host == "192.168.1.39":
                raise httpx.ConnectError("down", request=req)
            return httpx.Response(200, json={"audio_url": "", "duration": 1.0})

        self._route(handler)
        r = self.client.post("/api/voice/synthesise", json={"text": "hi", "pause_ms": 0})
        self.assertEqual(r.status_code, 200)
        self.assertEqual((leases.held, leases.released), (0, 1))


class StreamLeaseTests(_Base):
    def _events(self):
        r = self.client.post("/api/voice/synthesise/stream", json={"text": "One. Two."})
        return [line[6:] for line in r.text.splitlines() if line.startswith("data: ")]

    def test_one_lease_spans_the_stream(self) -> None:
        leases = self._leases()
        events = self._events()
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(len(self.requests), 2)
        self.assertEqual((len(leases.calls), leases.released), (1, 1))

    def test_refusal_without_cpu_fallback_is_an_error_event_with_the_reason(self) -> None:
        self._leases(error=REFUSED)
        with patch.object(voice, "XTTS_LEASE_CPU_FALLBACK", False):
            events = self._events()
        self.assertEqual(json.loads(events[0]),
                         {"error": REFUSED.detail, "status": 503, "chunk_index": 0})
        self.assertEqual(self.requests, [])

    def test_refusal_runs_every_sentence_on_cpu(self) -> None:
        self._leases(error=REFUSED)
        self._events()
        self.assertEqual([json.loads(q.content)["device"] for q in self.requests], ["cpu", "cpu"])


class PcmLeaseTests(_Base):
    def test_pcm_lease_is_realtime_and_held_until_the_stream_closes(self) -> None:
        leases = self._leases()
        held_during = []

        def handler(req):
            held_during.append(leases.held)
            return httpx.Response(200, content=b"\x00" * 8, headers={"x-sample-rate": "24000"})

        self._route(handler)
        r = self.client.post("/api/voice/synthesise/pcm", json={"text": "hi"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(held_during, [1])
        self.assertEqual(leases.calls[0]["priority"], "realtime")
        self.assertEqual((leases.held, leases.released), (0, 1))

    def test_pcm_refusal_without_cpu_fallback_is_a_503(self) -> None:
        self._leases(error=REFUSED)
        with patch.object(voice, "XTTS_LEASE_CPU_FALLBACK", False):
            r = self.client.post("/api/voice/synthesise/pcm", json={"text": "hi"})
        self.assertEqual(r.status_code, 503)
        self.assertIn("never fits", r.json()["detail"])

    def test_pcm_upstream_error_releases_the_lease(self) -> None:
        leases = self._leases()
        self._route(lambda req: httpx.Response(503, text="GPU OOM"))
        r = self.client.post("/api/voice/synthesise/pcm", json={"text": "hi"})
        self.assertEqual(r.status_code, 503)
        self.assertEqual((leases.held, leases.released), (0, 1))


if __name__ == "__main__":
    unittest.main()

"""Voice proxy: upstream failures map to clean 5xx errors, no internal detail leaks, PCM cleanup, 8-bit silence.

No network: httpx.AsyncClient is swapped for one over an httpx.MockTransport.
"""

import io
import json
import unittest
import wave
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import auth
from app.db import get_db
from app.routers import voice

BASE = "http://tts-internal-host:8031"
ALT = "http://fallback-host:8031"
WHISPER = "http://whisper-internal:8031"
_RealAsyncClient = httpx.AsyncClient


def _wav(width: int = 2, frames: int = 10) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(1000)
        w.writeframes((b"\x80" if width == 1 else b"\x01") * width * frames)
    return buf.getvalue()


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        async def no_db():
            yield None

        app = FastAPI()
        app.include_router(voice.router)
        app.dependency_overrides[get_db] = no_db
        self.client = TestClient(app)
        self.closed: list[str] = []
        for p in (
            patch.object(auth, "DEV_AUTH_BYPASS", True),
            patch.object(voice, "TTS_ENGINES", {"tts": BASE}),
            patch.object(voice, "_tts_base", lambda engine: BASE),
            patch.object(voice, "WHISPER_URL", WHISPER),
            patch.object(voice, "VOICE_S3_REHOST", False),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _route(self, handler):
        closed = self.closed

        class _Client(_RealAsyncClient):
            def __init__(self, *a, **kw):
                kw["transport"] = httpx.MockTransport(handler)
                super().__init__(*a, **kw)

            async def aclose(self):
                closed.append("client")
                await super().aclose()

        p = patch.object(voice.httpx, "AsyncClient", _Client)
        p.start()
        self.addCleanup(p.stop)


def _raise(exc_cls):
    def handler(request):
        raise exc_cls("boom at " + str(request.url), request=request)

    return handler


class TransportErrorTests(_Base):
    """Timeouts are 504, other transport failures 503, bad JSON 502 (not a raw 500)."""

    def _clone(self):
        return self.client.post(
            "/api/voice/clone",
            files=[("audio", ("a.wav", b"x", "audio/wav"))],
            data={"voice_name": "v", "replace": "true"},
        )

    def _transcribe(self):
        return self.client.post("/api/voice/transcribe", files={"audio": ("a.webm", b"x", "audio/webm")})

    def _analyze(self):
        return self.client.post("/api/voice/analyze", files=[("audio", ("a.wav", b"x", "audio/wav"))])

    def test_clone_timeout_and_transport_error(self) -> None:
        self._route(_raise(httpx.ReadTimeout))
        r = self._clone()
        self.assertEqual(r.status_code, 504, r.text)
        self.assertNotIn("tts-internal-host", r.text)

    def test_clone_write_error_is_503(self) -> None:
        self._route(_raise(httpx.WriteError))
        self.assertEqual(self._clone().status_code, 503)

    def test_clone_non_json_is_502(self) -> None:
        self._route(lambda req: httpx.Response(200, text="<html>oops</html>"))
        self.assertEqual(self._clone().status_code, 502)

    def test_transcribe_timeout_transport_and_bad_json(self) -> None:
        self._route(_raise(httpx.ReadTimeout))
        r = self._transcribe()
        self.assertEqual(r.status_code, 504, r.text)
        self.assertNotIn("whisper-internal", r.text)
        self._route(_raise(httpx.RemoteProtocolError))
        self.assertEqual(self._transcribe().status_code, 503)
        self._route(lambda req: httpx.Response(200, text="not json"))
        self.assertEqual(self._transcribe().status_code, 502)

    def test_transcribe_still_ok(self) -> None:
        self._route(lambda req: httpx.Response(200, json={"text": "hi", "language": "en"}))
        r = self._transcribe()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["text"], "hi")

    def test_transcribe_rejects_oversized_upload(self) -> None:
        calls = []
        self._route(lambda req: calls.append(req) or httpx.Response(200, json={}))
        with patch.object(voice, "VOICE_CLIP_MAX_BYTES", 4):
            r = self.client.post("/api/voice/transcribe", files={"audio": ("a.webm", b"x" * 10, "audio/webm")})
        self.assertEqual(r.status_code, 413, r.text)
        self.assertEqual(calls, [])

    def test_analyze_timeout_and_transport_error(self) -> None:
        self._route(_raise(httpx.ReadTimeout))
        self.assertEqual(self._analyze().status_code, 504)
        self._route(_raise(httpx.ReadError))
        self.assertEqual(self._analyze().status_code, 503)

    def test_upstream_http_error_still_relayed(self) -> None:
        self._route(lambda req: httpx.Response(422, text="bad clip"))
        r = self._analyze()
        self.assertEqual(r.status_code, 422)
        self.assertIn("bad clip", r.text)


class DeleteTests(_Base):
    def test_empty_204_is_success(self) -> None:
        self._route(lambda req: httpx.Response(204))
        r = self.client.delete("/api/voice/clones/alice")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"deleted": "alice"})

    def test_json_body_passed_through(self) -> None:
        self._route(lambda req: httpx.Response(200, json={"deleted": True, "id": "alice"}))
        self.assertEqual(self.client.delete("/api/voice/clones/alice").json(), {"deleted": True, "id": "alice"})


class NoLeakTests(_Base):
    def test_synthesise_503_hides_engine_urls(self) -> None:
        self._route(_raise(httpx.ConnectError))
        with patch.object(voice, "tts_engine_chain", lambda e: [("tts", BASE), ("alt", ALT)]):
            r = self.client.post("/api/voice/synthesise", json={"text": "hi", "pause_ms": 0})
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("host", r.text)

    def test_synthesise_falls_back_on_unreachable_engine(self) -> None:
        def handler(req):
            if req.url.host == "tts-internal-host":
                raise httpx.ConnectError("down", request=req)
            return httpx.Response(200, json={"audio_url": "", "duration": 1})

        self._route(handler)
        with patch.object(voice, "tts_engine_chain", lambda e: [("tts", BASE), ("alt", ALT)]):
            r = self.client.post("/api/voice/synthesise", json={"text": "hi", "pause_ms": 0})
        self.assertEqual(r.status_code, 200, r.text)

    def test_pcm_503_hides_engine_urls(self) -> None:
        self._route(_raise(httpx.ConnectError))
        with patch.object(voice, "tts_engine_chain", lambda e: [("tts", BASE)]):
            r = self.client.post("/api/voice/synthesise/pcm", json={"text": "hi"})
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("host", r.text)
        self.assertIn("client", self.closed)

    def _stream_events(self, handler):
        self._route(handler)
        with patch.object(voice, "tts_engine_chain", lambda e: [("tts", BASE)]):
            r = self.client.post("/api/voice/synthesise/stream", json={"text": "hi"})
        return [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: {")]

    def test_stream_http_error_is_generic(self) -> None:
        events = self._stream_events(lambda req: httpx.Response(500, text="Traceback at /srv/tts-internal-host"))
        self.assertEqual(events, [{"error": "TTS engine error", "status": 500, "chunk_index": 0}])

    def test_stream_other_error_is_generic(self) -> None:
        events = self._stream_events(lambda req: httpx.Response(200, text="not json"))
        self.assertEqual(events, [{"error": "TTS engine error", "chunk_index": 0}])


class PcmCleanupTests(_Base):
    def test_cleanup_attached_as_background_task(self) -> None:
        import asyncio

        self._route(lambda req: httpx.Response(200, content=b"\x00" * 8))

        async def run():
            with patch.object(voice, "tts_engine_chain", lambda e: [("tts", BASE)]):
                resp = await voice.synthesise_pcm(voice.SynthesiseRequest(text="hi"))
            # The body is never iterated (client went away before sending).
            self.assertIsNotNone(resp.background)
            await resp.background()
            await resp.background()  # idempotent

        asyncio.run(run())
        self.assertEqual(self.closed, ["client"])

    def test_streamed_body_closes_once(self) -> None:
        self._route(lambda req: httpx.Response(200, content=b"\x00" * 8))
        with patch.object(voice, "tts_engine_chain", lambda e: [("tts", BASE)]):
            r = self.client.post("/api/voice/synthesise/pcm", json={"text": "hi"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content, b"\x00" * 8)
        self.assertEqual(self.closed, ["client"])


class PacedFetchTests(_Base):
    def test_unreachable_audio_host_tries_next_engine(self) -> None:
        import asyncio

        def handler(req):
            if req.url.path == "/synthesise":
                return httpx.Response(200, json={"audio_url": f"http://{req.url.host}/a.wav"})
            if req.url.host == "tts-internal-host":
                raise httpx.ConnectError("down", request=req)
            return httpx.Response(200, content=_wav())

        self._route(handler)
        uploaded = []

        class _S3:
            def put_object(self, **kw):
                uploaded.append(kw)

        with patch.object(voice, "_voice_s3", lambda: _S3()):
            data = asyncio.run(
                voice._synthesise_paced([("tts", BASE), ("alt", ALT)], {}, ["One.", "Two."], 100)
            )
        self.assertEqual(data["engine"], "alt")
        self.assertEqual(len(uploaded), 1)

    def test_audio_fetch_http_error_degrades_to_single_shot(self) -> None:
        def handler(req):
            if req.url.path == "/synthesise":
                return httpx.Response(200, json={"audio_url": f"{BASE}/a.wav", "duration": 1})
            return httpx.Response(404)

        self._route(handler)
        with patch.object(voice, "VOICE_S3_REHOST", True), patch.object(
            voice, "VOICE_S3_PUBLIC_URL", BASE
        ), patch.object(voice, "tts_engine_chain", lambda e: [("tts", BASE)]):
            r = self.client.post("/api/voice/synthesise", json={"text": "One. Two.", "pause_ms": 100})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertNotIn("paced", r.json())


class RehostOriginTests(unittest.TestCase):
    """#81: echo's engines write to their own buckets on the same MinIO; those URLs are already reachable."""

    def _rehost(self, url):
        import asyncio

        fetched = []

        def handler(req):
            fetched.append(str(req.url))
            return httpx.Response(200, content=b"RIFF")

        class _Client(_RealAsyncClient):
            def __init__(self, *a, **kw):
                kw["transport"] = httpx.MockTransport(handler)
                super().__init__(*a, **kw)

        class _S3:
            def put_object(self, **kw):
                pass

        with patch.object(voice, "VOICE_S3_REHOST", True), \
                patch.object(voice, "VOICE_S3_PUBLIC_URL", "http://minio:9000/voice"), \
                patch.object(voice.httpx, "AsyncClient", _Client), \
                patch.object(voice, "_voice_s3", lambda: _S3()):
            return asyncio.run(voice._rehost_audio_url(url)), fetched

    def test_another_bucket_on_the_same_store_is_served_as_is(self) -> None:
        url = "http://minio:9000/voice-xtts/generated/voice/a.wav"
        self.assertEqual(self._rehost(url), (url, []))

    def test_another_host_is_rehosted(self) -> None:
        out, fetched = self._rehost("http://miso-host:8034/media/a.wav")
        self.assertTrue(out.startswith("http://minio:9000/voice/generated/voice/"), out)
        self.assertEqual(fetched, ["http://miso-host:8034/media/a.wav"])

    def test_a_prefix_lookalike_host_is_rehosted(self) -> None:
        out, fetched = self._rehost("http://minio:90001/voice/a.wav")
        self.assertEqual(len(fetched), 1)
        self.assertTrue(out.startswith("http://minio:9000/voice/"))


class EightBitSilenceTests(unittest.TestCase):
    def test_8bit_gap_is_unsigned_midpoint(self) -> None:
        out, _ = voice._concat_wavs_with_gaps([_wav(width=1), _wav(width=1)], 10)
        with wave.open(io.BytesIO(out), "rb") as w:
            frames = w.readframes(w.getnframes())
        self.assertEqual(set(frames), {0x80})

    def test_16bit_gap_is_zero(self) -> None:
        out, _ = voice._concat_wavs_with_gaps([_wav(), _wav()], 10)
        with wave.open(io.BytesIO(out), "rb") as w:
            frames = w.readframes(w.getnframes())
        self.assertIn(b"\x00\x00" * 10, frames)


if __name__ == "__main__":
    unittest.main()

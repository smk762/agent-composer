"""Voice clone proxy: companion_id is one safe path segment, overwrites are guarded here, clip uploads are bounded.

No network: httpx.AsyncClient is replaced by a recorder.
"""

import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import auth
from app.db import get_db
from app.routers import voice

BASE = "http://tts:8031"


class _Resp:
    def __init__(self, payload=None, status: int = 200):
        self._payload = payload if payload is not None else {}
        self.status_code = status
        self.content = b"zip"
        self.text = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            req = httpx.Request("GET", BASE)
            raise httpx.HTTPStatusError("err", request=req, response=httpx.Response(self.status_code, request=req))

    def json(self):
        return self._payload


def _recorder(listing=None, listing_status: int = 200):
    calls: list[tuple[str, str]] = []

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, **kw):
            calls.append(("GET", url))
            if url == f"{BASE}/clones":
                return _Resp({"clones": listing or []}, listing_status)
            return _Resp()

        async def delete(self, url, **kw):
            calls.append(("DELETE", url))
            return _Resp({"deleted": True})

        async def post(self, url, files=None, data=None, **kw):
            calls.append(("POST", url))
            return _Resp({"voice_clone_id": "x"})

    return _Client, calls


class _Base(unittest.TestCase):
    bypass = False

    def setUp(self) -> None:
        async def no_db():
            yield None

        app = FastAPI()
        app.include_router(voice.router)
        app.dependency_overrides[get_db] = no_db
        self.client = TestClient(app)
        for p in (
            patch.object(auth, "DEV_AUTH_BYPASS", self.bypass),
            patch.object(voice, "TTS_ENGINES", {"tts": BASE}),
            patch.object(voice, "_tts_base", lambda engine: BASE),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _with(self, listing=None, listing_status: int = 200):
        client_cls, calls = _recorder(listing, listing_status)
        p = patch.object(voice.httpx, "AsyncClient", client_cls)
        p.start()
        self.addCleanup(p.stop)
        return calls


class CompanionIdTests(_Base):
    bypass = True

    def test_download_refuses_query_injection(self) -> None:
        calls = self._with()
        r = self.client.get("/api/voice/clones/x%3Fa%3D1%23/download")
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(calls, [])

    def test_download_refuses_dot_segments(self) -> None:
        calls = self._with()
        r = self.client.get("/api/voice/clones/%2E%2E/download")
        self.assertIn(r.status_code, (400, 404), r.text)
        self.assertEqual(calls, [])

    def test_delete_refuses_query_injection(self) -> None:
        calls = self._with()
        r = self.client.delete("/api/voice/clones/x%3Fforce%3D1")
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(calls, [])

    def test_segment_validation(self) -> None:
        self.assertEqual(voice._companion_segment("alice_2-b"), "alice_2-b")
        for bad in ("..", ".", "", "a/b", "a?b", "a#b", "a b", "x" * 65):
            with self.assertRaises(voice.HTTPException, msg=bad):
                voice._companion_segment(bad)

    def test_valid_ids_reach_the_clone_route(self) -> None:
        calls = self._with()
        self.assertEqual(self.client.get("/api/voice/clones/alice/download").status_code, 200)
        self.assertEqual(self.client.delete("/api/voice/clones/alice").status_code, 200)
        self.assertEqual(calls, [("GET", f"{BASE}/clones/alice/download"), ("DELETE", f"{BASE}/clones/alice")])


class OverwriteGuardTests(_Base):
    bypass = False
    files = {"audio": ("a.wav", b"RIFF0000", "audio/wav")}

    def _clone(self, **data):
        return self.client.post("/api/voice/clone", files=self.files, data={"voice_name": "Alice"} | data)

    def test_existing_clone_is_409_without_replace(self) -> None:
        calls = self._with(listing=[{"companion_id": "alice"}])
        r = self._clone()
        self.assertEqual(r.status_code, 409, r.text)
        self.assertNotIn(("POST", f"{BASE}/clone-voice"), calls)

    def test_existing_clone_via_explicit_id_is_409(self) -> None:
        calls = self._with(listing=[{"companion_id": "bob"}])
        r = self._clone(companion_id="bob")
        self.assertEqual(r.status_code, 409, r.text)
        self.assertNotIn(("POST", f"{BASE}/clone-voice"), calls)

    def test_unreadable_clone_list_needs_auth(self) -> None:
        calls = self._with(listing_status=500)
        r = self._clone()
        self.assertEqual(r.status_code, 401, r.text)
        self.assertNotIn(("POST", f"{BASE}/clone-voice"), calls)

    def test_new_clone_needs_no_auth(self) -> None:
        calls = self._with(listing=[{"companion_id": "bob"}])
        r = self._clone()
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn(("POST", f"{BASE}/clone-voice"), calls)

    def test_bad_explicit_companion_id_is_400(self) -> None:
        calls = self._with()
        r = self._clone(companion_id="../x")
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(calls, [])


class ClipLimitTests(_Base):
    bypass = True

    def test_too_many_clips_is_413(self) -> None:
        calls = self._with()
        files = [("audio", (f"{i}.wav", b"RIFF", "audio/wav")) for i in range(3)]
        with patch.object(voice, "VOICE_MAX_CLIPS", 2):
            r = self.client.post("/api/voice/analyze", files=files)
        self.assertEqual(r.status_code, 413, r.text)
        self.assertEqual(calls, [])

    def test_oversized_clip_is_413(self) -> None:
        calls = self._with()
        files = [("audio", ("a.wav", b"R" * 2048, "audio/wav"))]
        with patch.object(voice, "VOICE_CLIP_MAX_BYTES", 1024):
            r = self.client.post("/api/voice/analyze", files=files)
            self.assertEqual(r.status_code, 413, r.text)
            r = self.client.post("/api/voice/clone", files=files, data={"voice_name": "New"})
            self.assertEqual(r.status_code, 413, r.text)
        self.assertNotIn(("POST", f"{BASE}/analyze-clips"), calls)
        self.assertNotIn(("POST", f"{BASE}/clone-voice"), calls)

    def test_clips_within_limits_are_forwarded(self) -> None:
        calls = self._with()
        files = [("audio", ("a.wav", b"RIFF", "audio/wav"))]
        self.assertEqual(self.client.post("/api/voice/analyze", files=files).status_code, 200)
        self.assertIn(("POST", f"{BASE}/analyze-clips"), calls)


if __name__ == "__main__":
    unittest.main()

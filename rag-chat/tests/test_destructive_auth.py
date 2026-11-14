"""Destructive voice/pipeline routes reject unauthenticated callers before touching a sidecar."""

import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import auth
from app.db import get_db
from app.routers import pipeline, voice


class DestructiveRoutesNeedAuthTest(unittest.TestCase):
    def setUp(self) -> None:
        async def no_db():
            yield None

        app = FastAPI()
        app.include_router(voice.router)
        app.include_router(pipeline.router)
        app.dependency_overrides[get_db] = no_db
        self.client = TestClient(app)
        p = patch.object(auth, "DEV_AUTH_BYPASS", False)
        p.start()
        self.addCleanup(p.stop)

    def test_clone_delete_requires_auth(self) -> None:
        self.assertEqual(self.client.delete("/api/voice/clones/abc").status_code, 401)

    def test_label_replace_requires_auth(self) -> None:
        r = self.client.put("/api/pipeline/labels", json={"labels": ["a"]})
        self.assertEqual(r.status_code, 401)

    def test_clone_replace_requires_auth(self) -> None:
        files = {"audio": ("a.wav", b"RIFF0000", "audio/wav")}
        r = self.client.post("/api/voice/clone", files=files, data={"voice_name": "Alice", "replace": "true"})
        self.assertEqual(r.status_code, 401)

    def test_clone_forwards_replace_to_the_engine(self) -> None:
        sent = []

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"voice_clone_id": "voice_clones/alice/prompt.pt"}

        class _Client:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, files=None, data=None):
                sent.append(data)
                return _Resp()

        files = {"audio": ("a.wav", b"RIFF0000", "audio/wav")}
        with patch.object(voice, "TTS_ENGINES", {"tts": "http://tts:8031"}), \
                patch.object(voice, "_tts_base", lambda engine: "http://tts:8031"), \
                patch.object(voice.httpx, "AsyncClient", _Client), patch.object(auth, "DEV_AUTH_BYPASS", True):
            for flag, expected in (("false", "false"), ("true", "true")):
                r = self.client.post("/api/voice/clone", files=files, data={"voice_name": "Alice", "replace": flag})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(sent[-1]["replace"], expected)
                self.assertEqual(sent[-1]["companion_id"], "alice")


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

"""INGEST_NONCE_STORE=redis with Redis down fails closed unless the SQLite fallback is opted into."""

import os
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("INGEST_NONCE_DB", os.path.join(tempfile.mkdtemp(prefix="ingest-test-"), "nonces.db"))

from app.nonce_store import NonceStore, RedisNonceStore


class UnreachableRedisTests(unittest.TestCase):
    """INGEST_NONCE_STORE=redis with Redis down fails closed unless the fallback is opted into."""

    def setUp(self) -> None:
        with mock.patch.dict("os.environ", {"INGEST_SHARED_SECRET": "test-only"}):
            from app import main
        self.main = main
        self.patches = [
            mock.patch.object(main, "NONCE_STORE_BACKEND", "redis"),
            mock.patch.object(main, "INGEST_REDIS_URL", "redis://127.0.0.1:1/0"),  # nothing listens
        ]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])

    def test_no_silent_downgrade_to_sqlite(self) -> None:
        store, backend = self.main._build_nonce_store()
        self.assertIsInstance(store, RedisNonceStore)
        self.assertEqual(backend, "redis")

    def test_signed_requests_are_refused_and_health_is_503(self) -> None:
        from fastapi import HTTPException
        from fastapi.testclient import TestClient

        store, backend = self.main._build_nonce_store()
        ts, nonce, body = str(int(time.time())), "abc", b"{}"
        with mock.patch.object(self.main, "_nonce_store", store), \
                mock.patch.object(self.main, "NONCE_BACKEND", backend):
            with self.assertRaises(HTTPException) as ctx:
                self.main.verify_signature(ts, nonce, self.main.sign_body(ts, nonce, body), body)
            self.assertEqual(ctx.exception.status_code, 503)
            r = TestClient(self.main.app).get("/health")
        self.assertEqual(r.status_code, 503)
        self.assertFalse(r.json()["ok"])

    def test_explicit_opt_in_keeps_the_sqlite_fallback(self) -> None:
        with mock.patch.object(self.main, "NONCE_ALLOW_SQLITE_FALLBACK", True, create=True):
            store, backend = self.main._build_nonce_store()
        self.assertIsInstance(store, NonceStore)
        self.assertEqual(backend, "sqlite-fallback")


if __name__ == "__main__":
    unittest.main()

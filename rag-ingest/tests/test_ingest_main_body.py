"""rag-ingest main: signed-body shape checks, redis URL redaction, the ingest UI's PKCS#7 pad."""

import json
import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("INGEST_NONCE_DB", os.path.join(tempfile.mkdtemp(prefix="ingest-test-"), "nonces.db"))

with mock.patch.dict("os.environ", {"INGEST_SHARED_SECRET": "test-only"}):
    from app import main  # main refuses to import without a secret

from fastapi.testclient import TestClient

HEADERS = {"X-Timestamp": "1", "X-Nonce": "n", "X-Signature": "s", "Content-Type": "application/json"}


class SignedBodyShapeTests(unittest.TestCase):
    def _post(self, body: bytes, require_encryption: bool = False):
        with mock.patch.object(main, "verify_signature", lambda *a, **k: None), \
                mock.patch.object(main, "REQUIRE_ENCRYPTION", require_encryption), \
                mock.patch.object(main, "ingest_docs") as ingest:
            r = TestClient(main.app).post("/ingest", content=body, headers=HEADERS)
        ingest.assert_not_called()
        return r

    def test_non_object_bodies_are_400(self) -> None:
        for body in (b"1", b"true", b"null", b"[1, 2]", b'"token"'):
            for enc in (False, True):
                with self.subTest(body=body, require_encryption=enc):
                    r = self._post(body, enc)
                    self.assertEqual(r.status_code, 400, r.text)
                    self.assertIn("JSON object", r.text)

    def test_envelope_holding_a_non_object_is_400(self) -> None:
        from cryptography.fernet import Fernet

        f = Fernet(Fernet.generate_key())
        body = json.dumps({"token": f.encrypt(b"[1, 2]").decode()}).encode()
        with mock.patch.object(main, "_fernet", lambda: f):
            r = self._post(body)
        self.assertEqual(r.status_code, 400, r.text)


class RedactedTests(unittest.TestCase):
    def test_password_is_masked(self) -> None:
        self.assertEqual(main._redacted("redis://:secret@host:6379/0"), "redis://:***@host:6379/0")

    def test_malformed_url_with_password_fails_closed(self) -> None:
        for url in ("redis://:secret@host:abc/0", "redis://:secret@host:99999/0"):
            with self.subTest(url=url):
                self.assertNotIn("secret", main._redacted(url))

    def test_url_without_password_is_unchanged(self) -> None:
        self.assertEqual(main._redacted("redis://host:6379/0"), "redis://host:6379/0")


class IngestUiPaddingTests(unittest.TestCase):
    def test_pkcs7_pad_adds_a_full_block_when_aligned(self) -> None:
        html = TestClient(main.app).get("/ui/ingest").text
        self.assertIn("const padLen = block - (buf.length % block);", html)
        self.assertNotIn("buf.length % block || block", html)


if __name__ == "__main__":
    unittest.main()

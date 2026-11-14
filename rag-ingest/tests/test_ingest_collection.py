"""Per-request collection for /ingest: allow-listed, never silently redirected."""

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

# app.main opens its nonce store at import, at /data/nonces.db by default. CI sets
# INGEST_NONCE_DB; other runners (pytest, the audit validator) may not, so default
# to a temp dir instead of failing with PermissionError on /data.
os.environ.setdefault("INGEST_NONCE_DB", os.path.join(tempfile.mkdtemp(prefix="ingest-test-"), "nonces.db"))

from app import ingest_pipeline as ip


class FakeQdrant:
    """Records which collection each upsert targets."""

    instances: list["FakeQdrant"] = []

    def __init__(self, url=None):
        self.created: list[str] = []
        self.upserts: list[tuple[str, int]] = []
        FakeQdrant.instances.append(self)

    async def get_collection(self, name):
        raise RuntimeError("missing")  # force create_collection

    async def create_collection(self, collection_name, vectors_config=None, **kw):
        self.created.append(collection_name)

    async def upsert(self, collection_name, points, wait=True):
        self.upserts.append((collection_name, len(points)))

    async def count(self, collection_name, count_filter=None, exact=True):
        return SimpleNamespace(count=0)

    async def delete(self, collection_name, points_selector, wait=True):
        raise AssertionError("nothing to delete in these tests")

    async def close(self):
        pass


class FakeEmbedder:
    async def embed(self, texts, profile="text"):
        return SimpleNamespace(vectors=[[0.1, 0.2, 0.3] for _ in texts], truncated=[False for _ in texts])


class AllowedCollectionsTests(unittest.TestCase):
    def test_default_always_allowed_extras_from_env(self) -> None:
        with mock.patch.dict("os.environ", {"INGEST_ALLOWED_COLLECTIONS": " audit_docs , ,other "}):
            self.assertEqual(ip.allowed_collections(), {ip.QDRANT_COLLECTION, "audit_docs", "other"})
        with mock.patch.dict("os.environ", {"INGEST_ALLOWED_COLLECTIONS": ""}):
            self.assertEqual(ip.allowed_collections(), {ip.QDRANT_COLLECTION})


class IngestCollectionTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeQdrant.instances.clear()
        self.patches = [
            mock.patch.object(ip, "AsyncQdrantClient", FakeQdrant),
            mock.patch.object(ip, "_text_client", FakeEmbedder()),
            mock.patch.object(ip, "RETRIEVAL_PROFILE", "text"),
            mock.patch.dict("os.environ", {"INGEST_ALLOWED_COLLECTIONS": "audit_docs"}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()

    def _ingest(self, **kw):
        docs = [{"id": "d1", "title": "t", "text": "hello world " * 20}]
        return asyncio.run(ip.ingest_docs(docs, source="ai-code-auditor", tags=["kind:wiki"], **kw))

    def test_text_docs_go_to_the_requested_collection(self) -> None:
        res = self._ingest(collection="audit_docs")
        q = FakeQdrant.instances[-1]
        self.assertEqual(res["collection"], "audit_docs")
        self.assertEqual(q.created, ["audit_docs"])
        self.assertTrue(q.upserts and all(c == "audit_docs" for c, _ in q.upserts))

    def test_default_collection_unchanged_without_a_request(self) -> None:
        res = self._ingest()
        self.assertEqual(res["collection"], ip.QDRANT_COLLECTION)
        self.assertTrue(all(c == ip.QDRANT_COLLECTION for c, _ in FakeQdrant.instances[-1].upserts))

    def test_disallowed_collection_is_refused_not_redirected(self) -> None:
        with self.assertRaises(ValueError):
            self._ingest(collection="someone_elses")
        self.assertEqual(FakeQdrant.instances, [])  # nothing was opened, let alone written


class IngestEndpointTests(unittest.TestCase):
    def test_embedder_mismatch_is_409(self) -> None:
        import json

        from fastapi.testclient import TestClient

        with mock.patch.dict("os.environ", {"INGEST_SHARED_SECRET": "test-only"}):
            from app import main

        async def mismatch(*a, **k):
            raise ip.EmbedderMismatchError("collection 'audit_docs' holds 3 point(s) embedded by infinity:x")

        body = json.dumps({"docs": [{"id": "d", "text": "x"}]}).encode()
        with mock.patch.object(main, "verify_signature", lambda *a, **k: None), \
                mock.patch.object(main, "ingest_docs", mismatch):
            r = TestClient(main.app).post("/ingest", content=body, headers={
                "X-Timestamp": "1", "X-Nonce": "n", "X-Signature": "s", "Content-Type": "application/json"})
        self.assertEqual(r.status_code, 409, r.text)
        self.assertIn("infinity:x", r.text)

    def test_endpoint_refuses_one_doc_sent_twice_with_400(self) -> None:
        import json

        from fastapi.testclient import TestClient

        with mock.patch.dict("os.environ", {"INGEST_SHARED_SECRET": "test-only"}):
            from app import main

        body = json.dumps({"docs": [{"id": "d", "text": "x"}, {"id": "d", "text": "y"}]}).encode()
        with mock.patch.object(main, "verify_signature", lambda *a, **k: None), \
                mock.patch.object(ip, "AsyncQdrantClient", FakeQdrant):
            FakeQdrant.instances.clear()
            r = TestClient(main.app).post("/ingest", content=body, headers={
                "X-Timestamp": "1", "X-Nonce": "n", "X-Signature": "s", "Content-Type": "application/json"})
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("same doc", r.text)
        self.assertEqual(FakeQdrant.instances, [])

    def test_endpoint_rejects_disallowed_collection_with_400(self) -> None:
        import json

        from fastapi.testclient import TestClient

        with mock.patch.dict("os.environ", {"INGEST_SHARED_SECRET": "test-only"}):
            from app import main  # main refuses to import without a secret

        body = json.dumps({"docs": [{"id": "d", "text": "x"}], "collection": "nope"}).encode()
        with mock.patch.object(main, "verify_signature", lambda *a, **k: None), \
                mock.patch.dict("os.environ", {"INGEST_ALLOWED_COLLECTIONS": "audit_docs"}), \
                mock.patch.object(main, "ingest_docs") as ingest:
            r = TestClient(main.app).post("/ingest", content=body, headers={
                "X-Timestamp": "1", "X-Nonce": "n", "X-Signature": "s", "Content-Type": "application/json"})
        self.assertEqual(r.status_code, 400)
        ingest.assert_not_called()


if __name__ == "__main__":
    unittest.main()

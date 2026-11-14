"""Replace on re-send (#24): a doc's re-send leaves exactly its new chunk set; failures keep the old one.

Runs against qdrant-client's in-memory mode, so the filters, counts and deletes are real.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from qdrant_client import AsyncQdrantClient

from app import ingest_pipeline as ip

COLL = ip.QDRANT_COLLECTION


class Embedder:
    """Deterministic 4-d vectors; can be told to fail on a chunk containing ``fail_on``."""

    def __init__(self):
        self.fail_on = None

    async def embed(self, texts, profile="text"):
        for t in texts:
            if self.fail_on and self.fail_on in t:
                raise RuntimeError("embedding backend down")
        return SimpleNamespace(vectors=[[1.0, float(len(t) % 7), 0.5, 0.25] for t in texts],
                               truncated=[False] * len(texts), dim=4)


class ReplaceOnResendTests(unittest.TestCase):
    def setUp(self):
        self.q = AsyncQdrantClient(location=":memory:")
        self.q.close = self._noop_close  # one shared client across ingests
        self.emb = Embedder()
        ip._INDEXED.clear()
        self.patches = [
            mock.patch.object(ip, "AsyncQdrantClient", lambda url=None: self.q),
            mock.patch.object(ip, "_text_client", self.emb),
            mock.patch.object(ip, "RETRIEVAL_PROFILE", "text"),
            mock.patch.object(ip, "CHUNK_SIZE", 100),
            mock.patch.object(ip, "CHUNK_OVERLAP", 0),
            mock.patch.object(ip, "UPSERT_BATCH_SIZE", 2),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    @staticmethod
    async def _noop_close():
        return None

    def ingest(self, docs, source="ai-code-auditor"):
        return asyncio.run(ip.ingest_docs(docs, source=source, tags=["kind:wiki"]))

    def points(self, doc_id=None):
        async def scroll():
            pts, _ = await self.q.scroll(COLL, limit=1000, with_payload=True)
            return pts
        pts = asyncio.run(scroll())
        return sorted(((p.payload["doc_id"], p.payload["source"], p.payload["chunk_index"], p.payload["text"])
                       for p in pts if doc_id is None or p.payload["doc_id"] == doc_id), key=str)

    def test_changed_resend_leaves_only_the_new_chunks(self):
        self.ingest([{"id": "a", "text": "A" * 250}, {"id": "b", "text": "B" * 50}])
        self.assertEqual(len(self.points("a")), 3)
        res = self.ingest([{"id": "a", "text": "a" * 150}])
        self.assertEqual(res["pruned"], 3)
        self.assertEqual([t[3] for t in self.points("a")], ["a" * 100, "a" * 50])
        self.assertEqual(len(self.points("b")), 1)  # other docs untouched

    def test_shorter_resend_drops_trailing_chunks_and_unchanged_resend_is_a_no_op(self):
        self.ingest([{"id": "a", "text": "x" * 100 + "y" * 100 + "z" * 100}])
        res = self.ingest([{"id": "a", "text": "x" * 100}])
        self.assertEqual(res["pruned"], 2)
        self.assertEqual([t[3] for t in self.points("a")], ["x" * 100])
        res = self.ingest([{"id": "a", "text": "x" * 100}])
        self.assertEqual((res["pruned"], len(self.points("a"))), (0, 1))

    def test_failure_mid_request_keeps_the_previous_version(self):
        self.ingest([{"id": "a", "text": "old " * 30}, {"id": "b", "text": "B" * 50}])
        before = self.points()
        self.emb.fail_on = "BOOM"
        with self.assertRaises(RuntimeError):
            self.ingest([{"id": "a", "text": "new " * 30}, {"id": "b", "text": "BOOM"}])
        after = self.points()
        self.assertEqual(after, before)  # nothing old deleted, nothing half-new written

    def test_id_less_docs_stay_append_only_and_sources_are_separate(self):
        self.ingest([{"text": "one"}])
        self.ingest([{"text": "two"}])
        self.assertEqual(len(self.points(None)), 2)
        self.ingest([{"id": "a", "text": "mine"}], source="s1")
        self.ingest([{"id": "a", "text": "theirs"}], source="s2")
        self.ingest([{"id": "a", "text": "mine v2"}], source="s1")
        self.assertEqual(sorted(t[1:] for t in self.points("a")), [("s1", 0, "mine v2"), ("s2", 0, "theirs")])

    def test_same_id_twice_in_one_request_is_refused(self):
        self.ingest([{"id": "a", "text": "old"}])
        with self.assertRaises(ip.IngestRequestError):
            self.ingest([{"id": "a", "text": "first"}, {"id": "a", "text": "second"}])
        self.assertEqual([t[3] for t in self.points("a")], ["old"])

    def test_id_zero_is_an_identity(self):
        self.ingest([{"id": 0, "text": "v1"}])
        self.ingest([{"id": 0, "text": "v2"}])
        self.assertEqual([t[3] for t in self.points("0")], ["v2"])

    def test_concurrent_resends_of_one_doc_leave_one_version(self):
        self.ingest([{"id": "a", "text": "v0"}])
        real = self.q

        class Yielding:  # yields before every Qdrant call, like a network client
            def __getattr__(self, name):
                attr = getattr(real, name)
                if not callable(attr):
                    return attr

                async def call(*a, **kw):
                    await asyncio.sleep(0)
                    return await attr(*a, **kw)
                return call

        async def both():
            await asyncio.gather(
                ip.ingest_docs([{"id": "a", "text": "v1"}], source="ai-code-auditor", tags=[]),
                ip.ingest_docs([{"id": "a", "text": "v2"}], source="ai-code-auditor", tags=[]),
            )
        with mock.patch.object(ip, "AsyncQdrantClient", lambda url=None: Yielding()):
            asyncio.run(both())
        self.assertEqual(len(self.points("a")), 1)

    def test_unembeddable_resend_replaces_and_is_reported(self):
        self.ingest([{"id": "a", "text": "old"}])

        class Overflow(Embedder):
            async def embed(self, texts, profile="text"):
                raise RuntimeError("input length exceeds the context length")
        with mock.patch.object(ip, "_text_client", Overflow()):
            res = self.ingest([{"id": "a", "text": "new"}])
        self.assertEqual((res["skipped_chunks"], res["incomplete_docs"], res["pruned"]), (1, ["a"], 1))
        self.assertEqual(self.points("a"), [])

    def test_payload_carries_ingested_at_and_indexes_exist(self):
        self.ingest([{"id": "a", "text": "x"}])
        self.ingest([{"id": "a", "text": "y"}])
        pts, _ = asyncio.run(self.q.scroll(COLL, limit=10, with_payload=True))
        self.assertIsInstance(pts[0].payload["ingested_at"], int)
        self.assertIn((COLL, "doc_id"), ip._INDEXED)

    def test_delete_docs(self):
        self.ingest([{"id": "a", "text": "A" * 150}, {"id": "b", "text": "b"}])
        self.ingest([{"id": "a", "text": "other source"}], source="else")
        res = asyncio.run(ip.delete_docs(["a", "a", "missing"], source="ai-code-auditor", collection=COLL))
        self.assertEqual((res["deleted"], res["docs"]), (2, 2))
        self.assertEqual([t[1] for t in self.points("a")], ["else"])
        self.assertEqual(len(self.points("b")), 1)


class CodeReplaceTests(ReplaceOnResendTests.__base__):
    """Code docs without an id are identified by (source, repo, path)."""

    def setUp(self):
        self.q = AsyncQdrantClient(location=":memory:")

        async def _close():
            return None

        self.q.close = _close
        ip._INDEXED.clear()
        self.patches = [
            mock.patch.object(ip, "AsyncQdrantClient", lambda url=None: self.q),
            mock.patch.object(ip, "_get_code_client", lambda: Embedder()),
            mock.patch.object(ip, "HYBRID_SPARSE", False),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def _code(self, src, repo="r1"):
        doc = {"kind": "code", "path": "pkg/mod.py", "repo": repo, "text": src}
        return asyncio.run(ip.ingest_docs([doc], source="ai-code-auditor", tags=[]))

    def _symbols(self):
        pts, _ = asyncio.run(self.q.scroll(ip.QDRANT_COLLECTION_CODE, limit=100, with_payload=True))
        return sorted((p.payload["repo"], p.payload["symbol"]) for p in pts)

    def test_code_resend_by_path(self):
        self._code("def a():\n    return 1\n\n\ndef b():\n    return 2\n")
        self._code("def a():\n    return 1\n", repo="r2")  # same path, other repo
        res = self._code("def a():\n    return 10\n")
        self.assertEqual(res["pruned"], 2)
        self.assertEqual(self._symbols(), [("r1", "a"), ("r2", "a")])

    def test_delete_by_path(self):
        self._code("def a():\n    return 1\n")
        self._code("def a():\n    return 1\n", repo="r2")
        res = asyncio.run(ip.delete_docs([], source="ai-code-auditor", collection=ip.QDRANT_COLLECTION_CODE,
                                         paths=["pkg/mod.py"], repo="r1"))
        self.assertEqual((res["deleted"], res["paths"]), (1, 1))
        self.assertEqual(self._symbols(), [("r2", "a")])

    def test_mixed_request_reports_both_collections(self):
        with mock.patch.object(ip, "_text_client", Embedder()), mock.patch.object(ip, "RETRIEVAL_PROFILE", "text"):
            res = asyncio.run(ip.ingest_docs(
                [{"id": "t", "text": "prose"}, {"kind": "code", "path": "m.py", "text": "def f():\n    return 1\n"}],
                source="s", tags=[]))
        self.assertEqual(res["collections"], [ip.QDRANT_COLLECTION, ip.QDRANT_COLLECTION_CODE])
        res = asyncio.run(ip.ingest_docs([{"kind": "code", "path": "m.py", "text": "def g():\n    return 2\n"}],
                                         source="s", tags=[]))
        self.assertEqual((res["collection"], res["embed_model"]), (ip.QDRANT_COLLECTION_CODE, ip.EMBED_MODEL_CODE))


class DeleteEndpointTests(unittest.TestCase):
    def _post(self, body):
        from fastapi.testclient import TestClient

        with mock.patch.dict("os.environ", {"INGEST_SHARED_SECRET": "test-only"}):
            from app import main
        with mock.patch.object(main, "verify_signature", lambda *a, **k: None), \
                mock.patch.dict("os.environ", {"INGEST_ALLOWED_COLLECTIONS": "extra"}), \
                mock.patch.object(main, "delete_docs") as dd:
            dd.return_value = {"ok": True, "deleted": 3}
            r = TestClient(main.app).post("/delete", content=json.dumps(body).encode(), headers={
                "X-Timestamp": "1", "X-Nonce": "n", "X-Signature": "s", "Content-Type": "application/json"})
        return r, dd

    def test_delete_routes_and_validates(self):
        r, dd = self._post({"doc_ids": ["a", " "], "source": "ai-code-auditor"})
        self.assertEqual(r.status_code, 200, r.text)
        dd.assert_called_once_with(["a"], source="ai-code-auditor", collection=ip.QDRANT_COLLECTION)
        r, dd = self._post({"doc_ids": ["a"], "source": "s", "kind": "code"})
        self.assertEqual(dd.call_args.kwargs["collection"], ip.QDRANT_COLLECTION_CODE)
        self.assertEqual(self._post({"doc_ids": ["a"], "source": "s", "collection": "nope"})[0].status_code, 400)
        self.assertEqual(self._post({"doc_ids": [], "source": "s"})[0].status_code, 422)
        self.assertEqual(self._post({"doc_ids": ["a"]})[0].status_code, 422)  # source is required
        self.assertEqual(self._post({"doc_ids": [" "], "source": "s"})[0].status_code, 422)
        # code: a collection would be ignored, so it's refused; paths delete id-less code docs
        self.assertEqual(self._post({"doc_ids": ["a"], "source": "s", "kind": "code", "collection": "x"})[0].status_code, 400)
        self.assertEqual(self._post({"paths": ["p.py"], "source": "s"})[0].status_code, 400)
        r, dd = self._post({"paths": ["p.py"], "repo": "r", "source": "s", "kind": "code"})
        self.assertEqual(r.status_code, 200, r.text)
        dd.assert_called_once_with([], source="s", collection=ip.QDRANT_COLLECTION_CODE, paths=["p.py"], repo="r")


if __name__ == "__main__":
    unittest.main()

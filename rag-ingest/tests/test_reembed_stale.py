"""reembed.sync drops a dst point whose src text became unusable (pack 1dd7e7f372)."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from qdrant_client import AsyncQdrantClient

from app import ingest_pipeline as ip
from app import reembed


class _Embedder:
    def __init__(self):
        self.calls = 0

    async def embed(self, texts, profile="text"):
        self.calls += len(texts)
        return SimpleNamespace(vectors=[[1.0, float(len(t) % 5), 0.5] for t in texts],
                               truncated=[False] * len(texts), dim=3)


class StaleTextTests(unittest.TestCase):
    def setUp(self):
        self.q = AsyncQdrantClient(location=":memory:")

        async def _close():
            return None
        self.q.close = _close
        ip._INDEXED.clear()
        ip._EMBEDDER_OK.clear()
        patches = [
            mock.patch.object(ip, "AsyncQdrantClient", lambda url=None: self.q),
            mock.patch.object(ip, "_text_client", _Embedder()),
            mock.patch.object(ip, "RETRIEVAL_PROFILE", "text"),
        ]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        asyncio.run(ip.ingest_docs([{"id": "a", "text": "alpha " * 10}, {"id": "b", "text": "beta"}],
                                   source="s", tags=["kind:wiki"]))

    def sync(self):
        self.emb = _Embedder()
        return asyncio.run(reembed.sync(ip.QDRANT_COLLECTION, "v2", scheme="nomic", client=self.q,
                                        embedder=self.emb, log=lambda *_: None))

    def dst(self):
        pts, _ = asyncio.run(self.q.scroll("v2", limit=100, with_payload=True))
        return {p.id: p.payload for p in pts}

    def test_source_text_emptied_removes_the_stale_dst_point(self):
        self.sync()
        pid = next(i for i, p in self.dst().items() if p["doc_id"] == "b")
        asyncio.run(self.q.set_payload(ip.QDRANT_COLLECTION, {"text": ""}, points=[pid]))
        r = self.sync()
        self.assertEqual((r["unembeddable"], r["delete"], r["failed"]), (1, 1, 0))
        self.assertNotIn(pid, self.dst())
        r = self.sync()  # converges: nothing left to do
        self.assertEqual((r["embed"], r["delete"], r["failed"], self.emb.calls), (0, 0, 0, 0))

    def test_failed_re_embed_removes_the_stale_dst_point(self):
        self.sync()
        pid = next(i for i, p in self.dst().items() if p["doc_id"] == "b")
        asyncio.run(self.q.set_payload(ip.QDRANT_COLLECTION, {"text": "beta, revised"}, points=[pid]))

        class _Down(_Embedder):
            async def embed(self, texts, profile="text"):
                raise RuntimeError("ollama down")

        r = asyncio.run(reembed.sync(ip.QDRANT_COLLECTION, "v2", scheme="nomic", client=self.q,
                                     embedder=_Down(), log=lambda *_: None))
        self.assertEqual((r["embed"], r["failed"], r["deleted"]), (1, 1, 1))
        self.assertNotIn(pid, self.dst())  # no stale "beta" left to serve
        r = self.sync()  # the next sync carries it across
        self.assertEqual((r["embedded"], r["failed"]), (1, 0))
        self.assertEqual(self.dst()[pid]["text"], "beta, revised")


if __name__ == "__main__":
    unittest.main()

"""Task prefixes (#32): the prefix reaches the embedder, never the stored text; re-embed sync."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from qdrant_client import AsyncQdrantClient

from app import embeddings as em
from app import ingest_pipeline as ip
from app import reembed


class _Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body or {"embedding": [0.1, 0.2]}, text

    def json(self):
        return self._body


class _Client:
    """Stands in for httpx.AsyncClient; overflows while the prompt is longer than ``limit``."""

    def __init__(self, limit=10_000):
        self.prompts, self.limit = [], limit

    async def post(self, url, json=None):
        self.prompts.append(json["prompt"])
        if len(json["prompt"]) > self.limit:
            return _Resp(400, text="input length exceeds the context length")
        return _Resp()


class PrefixTests(unittest.TestCase):
    def test_labels_and_prefixes(self):
        self.assertEqual(em.text_embedder_label("none"), f"ollama:{em.EMBED_MODEL}")
        self.assertEqual(em.text_embedder_label("nomic"), f"ollama:{em.EMBED_MODEL}+nomic-prefix")
        self.assertEqual(em.doc_prefix("nomic"), "search_document: ")
        self.assertEqual(em.doc_prefix("none"), "")

    def test_prefix_is_sent_and_budgeted(self):
        c = _Client()
        client = em.OllamaEmbeddingClient(prefix="search_document: ")
        with mock.patch.object(em, "EMBED_MAX_CHARS", 30):
            _, truncated = asyncio.run(client._embed_one(c, "x" * 20))
        self.assertEqual(c.prompts, ["search_document: " + "x" * 13])  # 30 - len(prefix)
        self.assertTrue(truncated)

    def test_shrinking_on_overflow_keeps_the_prefix(self):
        c = _Client(limit=60)
        client = em.OllamaEmbeddingClient(prefix="search_document: ")
        with mock.patch.object(em, "EMBED_MAX_CHARS", 0), mock.patch.object(em, "EMBED_MIN_CHARS", 10):
            asyncio.run(client._embed_one(c, "y" * 100))
        self.assertTrue(all(p.startswith("search_document: ") for p in c.prompts))
        self.assertLessEqual(len(c.prompts[-1]), 60)


class Embedder:
    def __init__(self):
        self.calls = 0

    async def embed(self, texts, profile="text"):
        self.calls += len(texts)
        return SimpleNamespace(vectors=[[1.0, float(len(t) % 5), 0.5] for t in texts],
                               truncated=[False] * len(texts), dim=3)


class ReembedSyncTests(unittest.TestCase):
    def setUp(self):
        self.q = AsyncQdrantClient(location=":memory:")

        async def _close():
            return None
        self.q.close = _close
        ip._INDEXED.clear()
        ip._EMBEDDER_OK.clear()
        self.patches = [
            mock.patch.object(ip, "AsyncQdrantClient", lambda url=None: self.q),
            mock.patch.object(ip, "_text_client", Embedder()),
            mock.patch.object(ip, "RETRIEVAL_PROFILE", "text"),
        ]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])
        self.ingest([{"id": "a", "text": "alpha " * 10}, {"id": "b", "text": "beta"}])

    def ingest(self, docs):
        return asyncio.run(ip.ingest_docs(docs, source="s", tags=["kind:wiki"]))

    def sync(self, **kw):
        self.emb = Embedder()
        return asyncio.run(reembed.sync(ip.QDRANT_COLLECTION, "v2", scheme="nomic", client=self.q,
                                        embedder=self.emb, log=lambda *_: None, **kw))

    def dst(self):
        pts, _ = asyncio.run(self.q.scroll("v2", limit=100, with_payload=True))
        return {p.id: p.payload for p in pts}

    def test_fill_then_idempotent_then_follow_changes(self):
        self.assertEqual(self.sync(dry_run=True)["embed"], 2)
        self.assertFalse(asyncio.run(self.q.collection_exists("v2")))  # dry run wrote nothing

        r = self.sync()
        self.assertEqual((r["embedded"], r["failed"]), (2, 0))
        dst = self.dst()
        self.assertEqual({p["embedder"] for p in dst.values()}, {em.text_embedder_label("nomic")})
        self.assertEqual(sorted(p["doc_id"] for p in dst.values()), ["a", "b"])
        self.assertNotIn("search_document", "".join(p["text"] for p in dst.values()))  # text stays clean

        r = self.sync()
        self.assertEqual((r["embed"], r["payload_only"], r["delete"], self.emb.calls), (0, 0, 0, 0))

        self.ingest([{"id": "a", "text": "alpha " * 10}, {"id": "b", "text": "beta v2"}])  # a: payload, b: text
        asyncio.run(ip.delete_docs(["a"], source="s", collection=ip.QDRANT_COLLECTION))
        self.ingest([{"id": "c", "text": "gamma"}])
        r = self.sync()
        self.assertEqual((r["embed"], r["delete"]), (2, 2))  # new b and c; old b and a gone
        self.assertEqual(sorted(p["text"] for p in self.dst().values()), ["beta v2", "gamma"])

    def test_payload_only_change_is_not_re_embedded(self):
        self.sync()
        asyncio.run(self.q.set_payload(ip.QDRANT_COLLECTION, {"tags": ["kind:wiki", "new"]}, points=ip.qm.Filter()))
        r = self.sync()
        self.assertEqual((r["embed"], r["payload_only"], self.emb.calls), (0, 2, 0))
        self.assertTrue(all("new" in p["tags"] for p in self.dst().values()))

    def test_refuses_a_dst_with_other_or_unlabelled_points(self):
        self.sync()
        asyncio.run(self.q.delete_payload("v2", keys=["embedder"], points=ip.qm.Filter()))
        ip._EMBEDDER_OK.clear()
        with self.assertRaises(ip.EmbedderMismatchError):
            self.sync()

    def test_strict_ingest_refuses_unlabelled_points(self):
        asyncio.run(self.q.delete_payload(ip.QDRANT_COLLECTION, keys=["embedder"], points=ip.qm.Filter()))
        ip._EMBEDDER_OK.clear()
        self.ingest([{"id": "d", "text": "lenient by default"}])
        with mock.patch.object(ip, "EMBEDDER_STRICT", True), self.assertRaises(ip.EmbedderMismatchError) as cm:
            self.ingest([{"id": "e", "text": "strict"}])
        self.assertIn("(unlabelled)", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

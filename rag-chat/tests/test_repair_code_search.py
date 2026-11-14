"""Repair related-code search (#35): embedded like the code collection, guarded by its label."""

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app import retrieval
from app.config import CODE_EMBEDDER, CODE_QUERY_PREFIX, EMBED_MODEL_CODE
from app.repair import code_search

DIFF = """--- a/cortex/retry.py
+++ b/cortex/retry.py
@@ -40,7 +40,9 @@ def with_retry(fn, attempts=3):
-    for i in range(attempts):
+    for attempt in range(attempts):
+        backoff = 2 ** attempt
+def jitter(delay):
"""


class CodeQueryTests(unittest.TestCase):
    def test_query_from_diff_uses_paths_symbols_and_changed_lines(self):
        q = code_search.code_query(DIFF)
        self.assertIn("files: cortex/retry.py", q)
        self.assertIn("jitter", q)
        self.assertIn("def with_retry(fn, attempts=3):", q)
        self.assertIn("backoff = 2 ** attempt", q)
        self.assertNotIn("+++", q)
        self.assertNotIn("@@", q)

    def test_query_from_pack_when_there_is_no_diff(self):
        block = "## Agent work pack\n- **location:** `cortex/retry.py:42`\n"
        q = code_search.code_query("", block, "Extract shared retry helper")
        self.assertEqual(q.splitlines()[0], "Extract shared retry helper")
        self.assertIn("files: cortex/retry.py:42", q)
        self.assertEqual(code_search.code_query("", ""), "")


class FetchTests(unittest.TestCase):
    def setUp(self):
        retrieval._embedder_cache.clear()
        self.addCleanup(retrieval._embedder_cache.clear)
        self.q = AsyncQdrantClient(location=":memory:")
        self.embedded = []

    async def embed(self, text):
        self.embedded.append(text)
        return [1.0, 0.0, 0.5]

    def _collection(self, label=CODE_EMBEDDER):
        async def setup():
            await self.q.create_collection("code", vectors_config={"dense": qm.VectorParams(size=3, distance=qm.Distance.COSINE)})
            await self.q.upsert("code", points=[
                qm.PointStruct(id=i, vector={"dense": [1.0, 0.1 * i, 0.5]}, payload={
                    "text": f"def f{i}(): pass", "repo": repo, "path": "pkg/mod.py", "start_line": 10 * i,
                    "end_line": 10 * i + 3, "symbol": f"f{i}", "embedder": label})
                for i, repo in enumerate(["r1", "r1", "r2"], 1)])
        asyncio.run(setup())

    def fetch(self, **kw):
        return asyncio.run(code_search.fetch_code_snippets("retry helper", collection="code", client=self.q,
                                                           embed=self.embed, **kw))

    def test_snippets_come_from_the_repo_with_their_source(self):
        self._collection()
        res = self.fetch(repo="r1")
        self.assertEqual(res.warning, "")
        self.assertEqual(len(res.snippets), 2)
        self.assertTrue(all(s.startswith("`r1/pkg/mod.py:") for s in res.snippets))
        self.assertIn("(f1)", "".join(res.snippets))
        self.assertEqual(self.embedded, ["retry helper"])

    def test_collection_from_another_embedder_gives_a_warning_not_snippets(self):
        self._collection(label="ollama:nomic-embed-text")
        res = self.fetch(repo="r1")
        self.assertEqual(res.snippets, [])
        self.assertIn("ollama:nomic-embed-text", res.warning)
        self.assertEqual(self.embedded, [])  # refused before embedding anything

    def test_missing_collection_is_silent(self):
        res = self.fetch()
        self.assertEqual((res.snippets, res.warning, self.embedded), ([], "", []))

    def test_query_is_embedded_by_infinity_with_the_code_model_never_ollama(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})

        real = httpx.AsyncClient

        async def no_ollama(_text):
            raise AssertionError("the code search must not use the text embedder")

        with patch("httpx.AsyncClient", lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(handler)})), \
                patch("app.routers.chat.ollama_embed", no_ollama):
            vec = asyncio.run(code_search.embed_code_query("retry helper"))
        self.assertEqual(vec, [0.1, 0.2, 0.3])
        self.assertTrue(str(seen[0].url).endswith("/embeddings"))
        body = json.loads(seen[0].content)
        self.assertEqual(body, {"model": EMBED_MODEL_CODE, "input": [CODE_QUERY_PREFIX + "retry helper"]})
        self.assertTrue(CODE_QUERY_PREFIX.startswith("Represent this query"))


if __name__ == "__main__":
    unittest.main()

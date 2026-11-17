"""Infinity idle-manager proxy tests: a real aiohttp upstream, no Infinity child.

Run from infinity-manager/: python -m unittest discover -s tests -v
(needs fastapi, aiohttp and httpx; the image ships them with infinity_emb).
"""
import gzip
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx  # noqa: E402
from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestServer  # noqa: E402

import manager  # noqa: E402

PAYLOAD = b'{"data": [{"embedding": [0.1, 0.2, 0.3]}], "padding": "' + b"x" * 2048 + b'"}'


class ProxyEncodingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def embeddings(request: web.Request) -> web.Response:
            # Like any upstream honouring the client's Accept-Encoding: a gzip body.
            body = gzip.compress(PAYLOAD)
            return web.Response(
                body=body,
                headers={"Content-Encoding": "gzip", "Content-Type": "application/json"},
            )

        upstream = web.Application()
        upstream.router.add_post("/embeddings", embeddings)
        self.server = TestServer(upstream)
        await self.server.start_server()

        self._saved = (manager.UPSTREAM, manager._ensure_started)
        manager.UPSTREAM = str(self.server.make_url("")).rstrip("/")

        async def _started() -> None:
            return None

        manager._ensure_started = _started

    async def asyncTearDown(self):
        manager.UPSTREAM, manager._ensure_started = self._saved
        await self.server.close()

    async def test_decompressed_body_is_not_labelled_gzip(self):
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://manager") as client:
            resp = await client.post(
                "/embeddings", content=b"{}", headers={"Accept-Encoding": "gzip"}
            )
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("content-encoding", resp.headers)
        self.assertEqual(int(resp.headers["content-length"]), len(PAYLOAD))
        self.assertEqual(resp.content, PAYLOAD)


if __name__ == "__main__":
    unittest.main()

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

        async def echo_query(request: web.Request) -> web.Response:
            return web.json_response({"a": request.query.getall("a", [])})

        upstream = web.Application()
        upstream.router.add_post("/embeddings", embeddings)
        upstream.router.add_get("/echo", echo_query)

        async def upstream_admin(request: web.Request) -> web.Response:
            return web.json_response({"reached": request.path})

        upstream.router.add_get("/admin", upstream_admin)
        upstream.router.add_get("/admin/{tail:.*}", upstream_admin)
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

    async def test_repeated_query_params_are_all_forwarded(self):
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://manager") as client:
            resp = await client.get("/echo?a=1&a=2")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"a": ["1", "2"]})

    async def _raw_get(self, raw_path: bytes) -> int:
        """Drive the ASGI app with a raw path, as a client that doesn't normalise URLs sends it."""
        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        path = raw_path.decode()
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
            "scheme": "http", "path": path, "raw_path": raw_path, "root_path": "",
            "query_string": b"", "headers": [], "client": ("127.0.0.1", 1), "server": ("manager", 80),
        }
        await manager.app(scope, receive, send)
        return next(m["status"] for m in sent if m["type"] == "http.response.start")

    async def test_upstream_admin_routes_are_not_proxied_in_any_spelling(self):
        spellings = (b"/admin", b"//admin/status", b"/./admin/status", b"/x/../admin/status", b"/admin/")
        statuses = {raw: await self._raw_get(raw) for raw in spellings}
        self.assertEqual(statuses, {raw: 404 for raw in spellings})


if __name__ == "__main__":
    unittest.main()

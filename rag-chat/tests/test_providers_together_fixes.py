"""Together: an item without b64_json is an error, never an empty image."""

import asyncio
import unittest
from unittest import mock

from app.providers import together


class _Resp:
    payload: dict = {}

    def raise_for_status(self):
        pass

    def json(self):
        return _Resp.payload


class _Client:
    calls: list = []

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kw):
        _Client.calls.append(json)
        return _Resp()


class TogetherFixTests(unittest.TestCase):
    def setUp(self):
        _Client.calls = []
        p = mock.patch.object(together.httpx, "AsyncClient", _Client)
        p.start()
        self.addCleanup(p.stop)
        self.provider = together.TogetherAIProvider(api_key="k")

    def test_url_only_item_is_an_error(self):
        _Resp.payload = {"data": [{"url": "https://example.invalid/a.png"}]}
        with self.assertRaises(RuntimeError):
            asyncio.run(self.provider.generate_image("a cat"))

    def test_no_items_is_an_error(self):
        _Resp.payload = {"data": []}
        with self.assertRaises(RuntimeError):
            asyncio.run(self.provider.generate_image("a cat"))

    def test_b64_item_decodes(self):
        _Resp.payload = {"data": [{"b64_json": "cG5n"}]}
        result = asyncio.run(self.provider.generate_image("a cat"))
        self.assertEqual(result.media[0].data, b"png")


if __name__ == "__main__":
    unittest.main()

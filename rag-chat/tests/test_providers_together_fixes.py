"""Together: an item without b64_json is an error, never an empty image."""

import asyncio
import base64
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

    def test_png_is_requested(self):
        _Resp.payload = {"data": [{"b64_json": base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()}]}
        asyncio.run(self.provider.generate_image("a cat"))
        self.assertEqual(_Client.calls[0]["output_format"], "png")

    def test_media_is_labelled_by_its_bytes(self):
        cases = {
            b"\x89PNG\r\n\x1a\n....": ("image/png", ".png"),
            b"\xff\xd8\xff\xe0\x00\x10JFIF": ("image/jpeg", ".jpg"),
            b"RIFF\x10\x00\x00\x00WEBPVP8 ": ("image/webp", ".webp"),
        }
        for raw, (mime, ext) in cases.items():
            _Resp.payload = {"data": [{"b64_json": base64.b64encode(raw).decode()}]}
            media = asyncio.run(self.provider.generate_image("a cat")).media[0]
            self.assertEqual((media.mime_type, media.filename[-len(ext):]), (mime, ext), raw)


if __name__ == "__main__":
    unittest.main()

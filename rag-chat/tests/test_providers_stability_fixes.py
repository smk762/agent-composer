"""Stability: strict size parsing, and the output's real dimensions are reported."""

import asyncio
import struct
import unittest
from unittest import mock

from app.providers import stability


def _png(w, h):
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", w, h) + b"\x08\x06\x00\x00\x00"


class _Resp:
    content = _png(1344, 768)

    def raise_for_status(self):
        pass


class _Client:
    calls: list = []

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, files=None, **kw):
        _Client.calls.append((url, files))
        return _Resp()


class StabilityFixTests(unittest.TestCase):
    def setUp(self):
        _Client.calls = []
        p = mock.patch.object(stability.httpx, "AsyncClient", _Client)
        p.start()
        self.addCleanup(p.stop)
        self.provider = stability.StabilityAIProvider(api_key="k")

    def test_malformed_size_is_a_lookup_error(self):
        for size in ("auto", "large", "1024x", "axb"):
            with self.assertRaises(LookupError, msg=size):
                asyncio.run(self.provider.generate_image("a cat", size=size))
        self.assertEqual(_Client.calls, [])

    def test_uppercase_and_empty_sizes_parse(self):
        asyncio.run(self.provider.generate_image("a cat", size="1920X1080"))
        self.assertEqual(_Client.calls[-1][1]["aspect_ratio"], (None, "16:9"))
        asyncio.run(self.provider.generate_image("a cat", size=""))
        self.assertEqual(_Client.calls[-1][1]["aspect_ratio"], (None, "1:1"))

    def test_reports_the_returned_dimensions_not_the_requested_ones(self):
        result = asyncio.run(self.provider.generate_image("a cat", size="1920x1080"))
        m = result.media[0]
        self.assertEqual((m.width, m.height), (1344, 768))

    def test_unreadable_output_leaves_dimensions_unset(self):
        self.assertEqual(stability._png_size(b"not a png"), (None, None))


if __name__ == "__main__":
    unittest.main()

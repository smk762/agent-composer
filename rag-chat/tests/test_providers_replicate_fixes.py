"""Replicate: strict size/n validation, abandoned predictions cancelled, no invented duration."""

import asyncio
import unittest
from unittest import mock

from app.providers import replicate


class _Resp:
    def __init__(self, payload=None, content=b"vid"):
        self._payload = payload or {}
        self.content = content
        self.headers = {"content-type": "video/mp4"}

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _Client:
    calls: list = []
    status = "starting"

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kw):
        _Client.calls.append(("POST", url, json))
        return _Resp({"id": "pred-1"})

    async def get(self, url, **kw):
        _Client.calls.append(("GET", url, None))
        if "/predictions/" in url:
            return _Resp({"status": _Client.status, "output": "https://example.invalid/v.mp4"})
        return _Resp()


class ReplicateFixTests(unittest.TestCase):
    def setUp(self):
        _Client.calls = []
        _Client.status = "starting"
        for p in (
            mock.patch.object(replicate.httpx, "AsyncClient", _Client),
            mock.patch.object(replicate, "_POLL_INTERVAL", 0),
            mock.patch.object(replicate, "_MAX_POLLS", 2),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.provider = replicate.ReplicateProvider(api_key="k")

    def test_malformed_size_is_a_lookup_error_before_any_call(self):
        for size in ("1024x", "large", "x768", "1024x768x2", "0x0"):
            with self.assertRaises(LookupError, msg=size):
                asyncio.run(self.provider.generate_image("a cat", size=size))
        self.assertEqual(_Client.calls, [])

    def test_uppercase_x_and_square_sizes_parse(self):
        _Client.status = "succeeded"
        asyncio.run(self.provider.generate_image("a cat", size="1024X768"))
        body = _Client.calls[0][2]["input"]
        self.assertEqual((body["width"], body["height"]), (1024, 768))

    def test_n_out_of_range_is_refused(self):
        for n in (0, -2, 5):
            with self.assertRaises(LookupError, msg=n):
                asyncio.run(self.provider.generate_image("a cat", n=n))
        self.assertEqual(_Client.calls, [])

    def test_timed_out_prediction_is_cancelled(self):
        with self.assertRaises(TimeoutError):
            asyncio.run(self.provider.generate_video("a cat"))
        self.assertIn(("POST", f"{replicate._API_BASE}/predictions/pred-1/cancel", None), _Client.calls)

    def test_video_duration_is_not_invented(self):
        _Client.status = "succeeded"
        result = asyncio.run(self.provider.generate_video("a cat", duration=9))
        self.assertIsNone(result.media[0].duration_seconds)


if __name__ == "__main__":
    unittest.main()

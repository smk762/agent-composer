"""OpenAI image requests with an n the API would reject never reach it."""

import asyncio
import unittest
from unittest import mock

from app.providers import openai


class _Resp:
    def raise_for_status(self):
        pass

    def json(self):
        return {"data": [{"b64_json": "cG5n"}]}


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


class OpenAINTests(unittest.TestCase):
    def setUp(self):
        _Client.calls = []
        p = mock.patch.object(openai.httpx, "AsyncClient", _Client)
        p.start()
        self.addCleanup(p.stop)
        self.provider = openai.OpenAIProvider(api_key="k")

    def test_dalle3_with_several_images_is_refused(self):
        with self.assertRaises(LookupError):
            asyncio.run(self.provider.generate_image("a cat", model="dall-e-3", n=2))
        self.assertEqual(_Client.calls, [])

    def test_non_positive_n_is_refused(self):
        for n in (0, -1):
            with self.assertRaises(LookupError):
                asyncio.run(self.provider.generate_image("a cat", n=n))
        self.assertEqual(_Client.calls, [])

    def test_dalle3_single_image_is_sent(self):
        asyncio.run(self.provider.generate_image("a cat", model="dall-e-3", n=1))
        self.assertEqual(_Client.calls[0]["n"], 1)


if __name__ == "__main__":
    unittest.main()

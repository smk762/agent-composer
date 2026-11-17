"""ComfyUI prompts we give up on are removed from the shared queue."""

import asyncio
import unittest
from unittest import mock

from app.providers import comfyui


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class _Client:
    calls: list = []
    running: list = []

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kw):
        _Client.calls.append(("GET", url, None))
        if url.endswith("/queue"):
            return _Resp(payload={"queue_running": _Client.running, "queue_pending": []})
        return _Resp(status_code=404)

    async def post(self, url, json=None, **kw):
        _Client.calls.append(("POST", url, json))
        if url.endswith("/prompt"):
            return _Resp(payload={"prompt_id": "p1"})
        return _Resp()


class ComfyUICancelTests(unittest.TestCase):
    def setUp(self):
        _Client.calls = []
        _Client.running = []
        patches = [
            mock.patch.object(comfyui.httpx, "AsyncClient", _Client),
            mock.patch.object(comfyui, "_POLL_INTERVAL", 0),
            mock.patch.object(comfyui, "_MAX_POLLS", 2),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.provider = comfyui.ComfyUIProvider()

    def _posts(self):
        return [(url.rsplit("/", 1)[-1], body) for method, url, body in _Client.calls if method == "POST"]

    def test_timed_out_pending_prompt_is_deleted_from_the_queue(self):
        with self.assertRaises(TimeoutError):
            asyncio.run(self.provider.generate_image("a cat"))
        posts = self._posts()
        self.assertIn(("queue", {"delete": ["p1"]}), posts)
        self.assertNotIn("interrupt", [name for name, _ in posts])

    def test_timed_out_running_prompt_is_interrupted(self):
        _Client.running = [[0, "p1", {}, {}, []]]
        with self.assertRaises(TimeoutError):
            asyncio.run(self.provider.generate_image("a cat"))
        posts = self._posts()
        self.assertIn(("interrupt", {"prompt_id": "p1"}), posts)
        self.assertIn(("queue", {"delete": ["p1"]}), posts)

    def test_cancelled_request_cancels_the_prompt(self):
        async def run():
            task = asyncio.create_task(self.provider.generate_image("a cat"))
            await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        with mock.patch.object(comfyui, "_POLL_INTERVAL", 10):
            asyncio.run(run())
        self.assertIn(("queue", {"delete": ["p1"]}), self._posts())


if __name__ == "__main__":
    unittest.main()

"""OpenAI video follows the /videos job API: create, poll, download content. No network."""

import asyncio
import unittest
from unittest import mock

import httpx

from app.providers import openai


def _run_with(handler, **kwargs):
    real = httpx.AsyncClient

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)

    provider = openai.OpenAIProvider(api_key="k")
    with mock.patch.object(openai.httpx, "AsyncClient", factory), \
            mock.patch.object(openai, "_VIDEO_POLL_INTERVAL", 0):
        return asyncio.run(provider.generate_video("a cat", **kwargs))


# pack-agent-composer-3aeaa36819
class OpenAIVideoJobTests(unittest.TestCase):
    def test_create_poll_download(self) -> None:
        seen = []
        polls = iter(["in_progress", "completed"])

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.method, request.url.path, request.headers.get("authorization")))
            if request.method == "POST":
                body = request.read().decode()
                fields = {name: (f'name="{name}"' in body) for name in ("model", "prompt", "seconds", "size")}
                seen.append(fields)
                seen.append(("sora-2" in body, "1280x720" in body))
                return httpx.Response(200, json={"id": "video_1", "status": "queued"})
            if request.url.path.endswith("/content"):
                return httpx.Response(200, content=b"MP4DATA", headers={"content-type": "video/mp4"})
            return httpx.Response(200, json={"id": "video_1", "status": next(polls), "seconds": "8",
                                             "size": "1280x720"})

        result = _run_with(handler, duration=8, aspect_ratio="16:9")
        self.assertEqual(seen, [
            ("POST", "/v1/videos", "Bearer k"),
            {"model": True, "prompt": True, "seconds": True, "size": True},
            (True, True),
            ("GET", "/v1/videos/video_1", "Bearer k"),
            ("GET", "/v1/videos/video_1", "Bearer k"),
            ("GET", "/v1/videos/video_1/content", "Bearer k"),
        ])
        self.assertEqual(
            [(m.data, m.mime_type, m.width, m.height, m.duration_seconds) for m in result.media],
            [(b"MP4DATA", "video/mp4", 1280, 720, 8.0)],
        )

    def test_failed_job_is_an_error_not_an_empty_result(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json={"id": "video_1", "status": "queued"})
            return httpx.Response(200, json={"id": "video_1", "status": "failed", "error": {"message": "x"}})

        with self.assertRaises(RuntimeError):
            _run_with(handler)

    def test_unsupported_duration_or_aspect_is_refused_before_any_call(self) -> None:
        def handler(request):
            raise AssertionError("no request expected")

        for kwargs in ({"duration": 5}, {"aspect_ratio": "1:1"}, {"model": "sora"}):
            with self.assertRaises(LookupError, msg=kwargs):
                _run_with(handler, **kwargs)


if __name__ == "__main__":
    unittest.main()

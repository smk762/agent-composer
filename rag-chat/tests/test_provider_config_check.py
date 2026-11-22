"""A provider's config check says *why* it failed: an unreachable API or a timeout is logged, not
read silently as a rejected key (ai-code-auditor pack-class-silent-failure). No network."""

import asyncio
import unittest
from unittest import mock

import httpx

from app.providers import comfyui, openai, replicate, stability, together

PROVIDERS = [
    (comfyui.ComfyUIProvider, {"comfyui_url": "http://comfy.invalid:8188"}),
    (openai.OpenAIProvider, {"api_key": "k"}),
    (replicate.ReplicateProvider, {"api_key": "k"}),
    (stability.StabilityAIProvider, {"api_key": "k"}),
    (together.TogetherAIProvider, {"api_key": "k"}),
]


class ConfigCheckTests(unittest.TestCase):
    def test_a_transport_error_is_false_and_logged(self) -> None:
        for cls, kw in PROVIDERS:
            with self.subTest(cls.__name__), mock.patch.object(
                    httpx.AsyncClient, "get", side_effect=httpx.ConnectError("connection refused")):
                with self.assertLogs("rag-chat.provider", "WARNING") as logs:
                    self.assertFalse(asyncio.run(cls(**kw).validate_config()))
                self.assertIn("ConnectError: connection refused", "\n".join(logs.output))

    def test_a_bug_is_not_read_as_a_bad_key(self) -> None:
        for cls, kw in PROVIDERS:
            with self.subTest(cls.__name__), mock.patch.object(
                    httpx.AsyncClient, "get", side_effect=TypeError("bad call")):
                with self.assertRaises(TypeError):
                    asyncio.run(cls(**kw).validate_config())


if __name__ == "__main__":
    unittest.main()

"""Per-request bounds on image generation, and ComfyUI advertising only what it can run. No network."""

import asyncio
import unittest
from unittest import mock

from pydantic import ValidationError

from app.models.schemas import ImageGenerateRequest
from app.providers import comfyui
from app.providers.base import Capability


class _NoNetwork:
    def __init__(self, *a, **kw) -> None:
        raise AssertionError("must be refused before any request to ComfyUI")


class ImageCountTests(unittest.TestCase):
    def test_n_is_bounded(self) -> None:
        self.assertEqual(ImageGenerateRequest(prompt="p").n, 1)
        self.assertEqual(ImageGenerateRequest(prompt="p", n=4).n, 4)
        for n in (0, -1, 5, 1000):
            with self.assertRaises(ValidationError, msg=f'n={n!r}'):
                ImageGenerateRequest(prompt="p", n=n)


class ComfyUITests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(comfyui.httpx, "AsyncClient", _NoNetwork)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.provider = comfyui.ComfyUIProvider()

    def test_only_txt2img_is_advertised(self) -> None:
        self.assertTrue(self.provider.supports(Capability.IMAGE_GENERATE))
        for cap in (Capability.IMAGE_EDIT, Capability.VIDEO_GENERATE):
            self.assertFalse(self.provider.supports(cap))
            self.assertEqual(self.provider.available_models(cap), [])

    def test_oversized_or_malformed_sizes_are_refused(self) -> None:
        for size in ("16384x16384", "64x64", "1023x1024", "abc", "1024x1024x3"):
            with self.assertRaises(LookupError, msg=f'size={size!r}'):
                asyncio.run(self.provider.generate_image("p", size=size))

    def test_image_count_is_capped(self) -> None:
        for n in (0, 5, 1000):
            with self.assertRaises(LookupError, msg=f'n={n!r}'):
                asyncio.run(self.provider.generate_image("p", n=n))


if __name__ == "__main__":
    unittest.main()

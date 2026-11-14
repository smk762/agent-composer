"""Request shapes the Stability provider sends. No network."""

import asyncio
import unittest
from unittest import mock

from app.providers import stability


class _Resp:
    content = b"png"

    def raise_for_status(self) -> None:
        pass


class _Client:
    """Stand-in for httpx.AsyncClient that records each POST."""

    calls: list[tuple[str, dict]] = []

    def __init__(self, *a, **kw) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, *, headers=None, files=None, **kw):
        _Client.calls.append((url, files))
        return _Resp()


def _run(coro):
    return asyncio.run(coro)


class StabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        _Client.calls = []
        patcher = mock.patch.object(stability.httpx, "AsyncClient", _Client)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.provider = stability.StabilityAIProvider(api_key="k")

    def test_default_sd3_goes_to_the_sd3_endpoint_with_model_field(self) -> None:
        _run(self.provider.generate_image("a cat"))
        url, form = _Client.calls[0]
        self.assertTrue(url.endswith("/stable-image/generate/sd3"))
        self.assertEqual(form["model"], (None, "sd3-large"))

    def test_core_and_ultra_have_their_own_endpoints(self) -> None:
        for model in ("core", "ultra"):
            _run(self.provider.generate_image("a cat", model=model))
            url, form = _Client.calls[-1]
            self.assertTrue(url.endswith(f"/stable-image/generate/{model}"))
            self.assertNotIn("model", form)

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaises(LookupError):
            _run(self.provider.generate_image("a cat", model="sdxl-9000"))
        self.assertEqual(_Client.calls, [])

    def test_maskless_edit_is_sd3_image_to_image(self) -> None:
        result = _run(self.provider.edit_image("make it blue", b"img"))
        url, form = _Client.calls[0]
        self.assertTrue(url.endswith("/stable-image/generate/sd3"))
        self.assertEqual(form["mode"], (None, "image-to-image"))
        self.assertIn("strength", form)
        self.assertEqual(form["model"], (None, "sd3-large"))
        self.assertNotIn("search-and-replace", url)
        self.assertEqual(result.model, "sd3-large")

    def test_masked_edit_is_inpaint(self) -> None:
        _run(self.provider.edit_image("fill", b"img", mask=b"mask"))
        url, form = _Client.calls[0]
        self.assertTrue(url.endswith("/stable-image/edit/inpaint"))
        self.assertIn("mask", form)

    def test_edit_models_are_not_silently_swapped(self) -> None:
        # Mask-less edits only run SD3; inpaint only runs core. Anything else is a 400, not a swap.
        with self.assertRaises(LookupError):
            _run(self.provider.edit_image("x", b"img", model="core"))
        with self.assertRaises(LookupError):
            _run(self.provider.edit_image("x", b"img", mask=b"m", model="sd3-large"))
        self.assertEqual(_Client.calls, [])
        self.assertEqual(_run(self.provider.edit_image("x", b"img", model="sd3-medium")).model, "sd3-medium")
        self.assertEqual(_run(self.provider.edit_image("x", b"img", mask=b"m")).model, "core")


if __name__ == "__main__":
    unittest.main()

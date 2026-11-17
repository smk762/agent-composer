"""Request shapes the Stability provider sends. No network."""

import asyncio
import unittest
from unittest import mock

from app.providers import openai, replicate, stability


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


class StabilityAspectTests(unittest.TestCase):
    ALLOWED = {"21:9", "16:9", "3:2", "5:4", "1:1", "4:5", "2:3", "9:16", "9:21"}

    def test_only_supported_aspect_ratios_are_sent(self) -> None:
        for w, h in ((1024, 768), (800, 600), (768, 1024), (1024, 1024), (1920, 1080), (640, 480)):
            with self.subTest(size=f"{w}x{h}"):
                self.assertIn(stability._closest_aspect(w, h), self.ALLOWED)

    def test_four_by_three_maps_symmetrically(self) -> None:
        landscape = stability._closest_aspect(1024, 768)
        portrait = stability._closest_aspect(768, 1024)
        self.assertEqual(landscape, ":".join(reversed(portrait.split(":"))))


class _JsonResp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.content = b""

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _JsonClient:
    """Stand-in for httpx.AsyncClient answering every POST with ``payload``."""

    calls: list[tuple[str, dict]] = []
    payload: dict = {}

    def __init__(self, *a, **kw) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, *, headers=None, files=None, json=None, **kw):
        _JsonClient.calls.append((url, files if files is not None else json))
        return _JsonResp(_JsonClient.payload)


class OpenAITests(unittest.TestCase):
    def setUp(self) -> None:
        _JsonClient.calls = []
        _JsonClient.payload = {"data": [{"b64_json": "cG5n"}]}
        patcher = mock.patch.object(openai.httpx, "AsyncClient", _JsonClient)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.provider = openai.OpenAIProvider(api_key="k")

    def test_dalle_edit_asks_for_base64(self) -> None:
        result = _run(self.provider.edit_image("x", b"img", model="dall-e-2"))
        _, form = _JsonClient.calls[0]
        self.assertEqual(form["response_format"], (None, "b64_json"))
        self.assertEqual(result.media[0].data, b"png")

    def test_url_only_items_are_an_error_not_empty_images(self) -> None:
        _JsonClient.payload = {"data": [{"url": "https://example.invalid/a.png"}]}
        with self.assertRaises(RuntimeError):
            _run(self.provider.edit_image("x", b"img", model="dall-e-2"))

    def test_video_with_an_input_image_is_refused(self) -> None:
        with self.assertRaises(LookupError):
            _run(self.provider.generate_video("a cat", image=b"frame"))
        self.assertEqual(_JsonClient.calls, [])


class ReplicateTests(unittest.TestCase):
    def setUp(self) -> None:
        _JsonClient.calls = []
        _JsonClient.payload = {"id": "pred-1"}
        patcher = mock.patch.object(replicate.httpx, "AsyncClient", _JsonClient)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.provider = replicate.ReplicateProvider(api_key="k")

    def test_malformed_model_never_reaches_the_api_url(self) -> None:
        for model in ("a/b/../../account", "owner/name?x=1", "x", "owner/name:not-a-version", "../predictions"):
            with self.subTest(model=model), self.assertRaises(LookupError):
                _run(self.provider.generate_image("a cat", model=model))
            with self.subTest(video=model), self.assertRaises(LookupError):
                _run(self.provider.generate_video("a cat", model=model))
        self.assertEqual(_JsonClient.calls, [])

    def test_known_and_well_formed_models_are_used(self) -> None:
        self.assertEqual(
            replicate._resolve_model("flux-schnell", replicate._IMAGE_MODELS), "black-forest-labs/flux-schnell"
        )
        self.assertEqual(replicate._resolve_model("owner/my-model", replicate._IMAGE_MODELS), "owner/my-model")

    def test_video_with_an_input_image_is_refused(self) -> None:
        with self.assertRaises(LookupError):
            _run(self.provider.generate_video("a cat", image=b"frame"))
        self.assertEqual(_JsonClient.calls, [])


if __name__ == "__main__":
    unittest.main()

"""Provider resolution: deterministic default, stored settings can't break it, real type checks."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from typing import Optional
from unittest import mock

from app.providers import registry
from app.providers.base import BaseProvider, Capability, ImageProvider


class _Image(ImageProvider):
    name = "fake-image"
    capabilities = [Capability.IMAGE_GENERATE, Capability.VIDEO_GENERATE]

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.settings = kwargs

    async def validate_config(self):
        return True

    async def generate_image(self, prompt, **kw):
        raise NotImplementedError


class _FakeDB:
    def __init__(self, rows):
        self.rows = rows
        self.stmts = []

    async def execute(self, stmt):
        self.stmts.append(stmt)
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: list(self.rows)))


def _row(**kw):
    base = dict(provider_name="fake-image", encrypted_api_key=None, settings_json=None, enabled=True)
    base.update(kw)
    return SimpleNamespace(**base)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(registry._PROVIDER_CLASSES, {"fake-image": _Image}, clear=True)
        p.start()
        self.addCleanup(p.stop)

    def test_settings_api_key_does_not_crash_instantiation(self):
        db = _FakeDB([_row(settings_json=json.dumps({"api_key": "x", "region": "eu"}))])
        p = asyncio.run(registry.ProviderRegistry.get_provider(db, "u", Capability.IMAGE_GENERATE))
        self.assertIsNone(p.api_key)
        self.assertEqual(p.settings, {"region": "eu"})

    def test_default_selection_query_is_ordered(self):
        db = _FakeDB([_row()])
        asyncio.run(registry.ProviderRegistry.get_provider(db, "u", Capability.IMAGE_GENERATE))
        self.assertIn("ORDER BY", str(db.stmts[0]).upper())

    def test_wrong_provider_type_is_an_error_even_without_asserts(self):
        db = _FakeDB([_row()])
        with mock.patch.object(registry, "VideoProvider", type("V", (BaseProvider,), {})):
            with self.assertRaises(LookupError):
                asyncio.run(registry.ProviderRegistry.get_video_provider(db, "u"))


if __name__ == "__main__":
    unittest.main()

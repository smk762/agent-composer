"""Repair validation helpers fail closed when no audit API URL is configured. No network, no model."""

import unittest
from unittest.mock import patch

import httpx

from app.repair import multi_model

real_async_client = httpx.AsyncClient
DIFF = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b"


def _patch_http(handler):
    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    return patch("httpx.AsyncClient", factory)


# pack-agent-composer-255bf84be4
class MissingAuditUrlFailsClosedTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_url_is_not_a_pass(self) -> None:
        for check in (multi_model._validate_via_api, multi_model._validate_via_tests):
            passed, errors = await check(DIFF, "repo", "")
            self.assertFalse(passed, check.__name__)
            self.assertIn("not configured", errors[0])

    async def test_empty_patch_still_passes(self) -> None:
        for check in (multi_model._validate_via_api, multi_model._validate_via_tests):
            self.assertEqual(await check("", "repo", ""), (True, []))


if __name__ == "__main__":
    unittest.main()

"""validate_patch verdicts: a payload with no 'passed' key is malformed, never a pass. No network, no model."""

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


# pack-agent-composer-7700a441bc
class MissingPassedTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, body):
        with _patch_http(lambda req: httpx.Response(200, json=body)):
            return await multi_model._validate_via_tests(DIFF, "repo", "http://audit")

    async def test_missing_passed_is_not_a_pass(self) -> None:
        results = [await self._run(body) for body in ({}, {"error": "boom"}, {"failure_summary": []})]
        self.assertEqual([ok for ok, _ in results], [False, False, False])
        self.assertTrue(all("malformed" in errors[0] for _, errors in results))

    async def test_explicit_pass_still_passes(self) -> None:
        self.assertEqual(await self._run({"passed": True, "failure_summary": []}), (True, []))


if __name__ == "__main__":
    unittest.main()

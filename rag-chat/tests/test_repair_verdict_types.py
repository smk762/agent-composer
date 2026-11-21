"""validate_patch verdicts: a non-bool passed or non-list summary is malformed. No network, no model."""

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


# pack-agent-composer-e2389c5756
class VerdictTypeTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, body):
        with _patch_http(lambda req: httpx.Response(200, json=body)):
            return await multi_model._validate_via_tests(DIFF, "repo", "http://audit")

    async def test_non_bool_passed_is_not_a_pass(self) -> None:
        for passed in ("false", "true", 1, 0, None):
            ok, errors = await self._run({"passed": passed, "failure_summary": ["x"]})
            self.assertIs(ok, False, passed)
            self.assertIn("malformed", errors[0])

    async def test_non_list_summary_is_malformed(self) -> None:
        for summary in ("tests failed", ["ok", 3], {"a": 1}):
            ok, errors = await self._run({"passed": False, "failure_summary": summary})
            self.assertIs(ok, False, summary)
            self.assertIn("malformed", errors[0])

    async def test_real_verdicts_still_work(self) -> None:
        self.assertEqual(await self._run({"passed": True, "failure_summary": []}), (True, []))
        self.assertEqual(await self._run({"passed": False, "failure_summary": ["E1"]}), (False, ["E1"]))


if __name__ == "__main__":
    unittest.main()

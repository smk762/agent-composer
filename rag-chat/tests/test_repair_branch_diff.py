"""_fetch_diff_from_branch: odd audit API answers become error messages, never exceptions."""

import json
import unittest
from unittest.mock import patch

import httpx

from app.routers.repair import _fetch_diff_from_branch

real_async_client = httpx.AsyncClient
URL = "http://audit.test:8765"


def _patch_http(handler):
    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    return patch("httpx.AsyncClient", factory)


async def _fetch(status: int, payload) -> tuple[str, str]:
    body = json.dumps(payload).encode()  # json=None would send no body at all
    with _patch_http(lambda r: httpx.Response(status, content=body)):
        return await _fetch_diff_from_branch("repo", "dev", URL)


class FetchDiffFromBranchTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_object_json_is_an_error(self) -> None:
        for payload in ([], "oops", None):
            diff, err = await _fetch(200, payload)
            self.assertEqual(diff, "")
            self.assertIn("unexpected JSON", err)

    async def test_null_fields_are_errors_not_crashes(self) -> None:
        diff, err = await _fetch(200, {"diff": None})
        self.assertEqual(diff, "")
        self.assertIn("empty", err)

        diff, err = await _fetch(400, {"error": None, "reason": "branch_not_found"})
        self.assertIn("status 400", err)
        self.assertIn("feature branch", err)

        diff, err = await _fetch(404, {"error": None})
        self.assertIsInstance(err, str)
        self.assertIn("not found", err)

    async def test_diff_is_returned(self) -> None:
        self.assertEqual(await _fetch(200, {"diff": "--- a\n+++ b\n"}), ("--- a\n+++ b\n", ""))


if __name__ == "__main__":
    unittest.main()

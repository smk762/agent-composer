"""Repair refs: a trailing newline does not pass the ref check. No network, no model."""

import unittest
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from app.repair.models import RepairRequest

real_async_client = httpx.AsyncClient
DIFF = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b"


def _patch_http(handler):
    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    return patch("httpx.AsyncClient", factory)


# pack-agent-composer-3f4e9dee00
class RefTrailingNewlineTests(unittest.TestCase):
    def test_trailing_newline_is_refused(self) -> None:
        for value in ("main\n", "origin/dev\n"):
            with self.assertRaises(ValidationError, msg=repr(value)):
                RepairRequest(repo="r", compare_branch=value)

    def test_plain_ref_still_accepted(self) -> None:
        self.assertEqual(RepairRequest(repo="r", compare_branch="main").compare_branch, "main")


if __name__ == "__main__":
    unittest.main()

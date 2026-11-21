"""Inline patch extraction splits only on newlines. No network, no model."""

import unittest
from unittest.mock import patch

import httpx

from app.repair.patch_applier import extract_patch

real_async_client = httpx.AsyncClient
DIFF = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b"


def _patch_http(handler):
    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    return patch("httpx.AsyncClient", factory)


# pack-agent-composer-f9fdf9b065
class InlineSplitTests(unittest.TestCase):
    def test_form_feed_inside_a_hunk_line_is_kept(self) -> None:
        resp = "Here:\n--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n ctx\x0cmore\n-a\n+b\nThanks"
        self.assertEqual(extract_patch(resp), "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n ctx\x0cmore\n-a\n+b\n")


if __name__ == "__main__":
    unittest.main()

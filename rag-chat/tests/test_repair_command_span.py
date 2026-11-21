"""Pack context: suggested test commands are rendered as safe inline code spans. No network, no model."""

import unittest
from unittest.mock import patch

import httpx

from app.repair.pack_client import build_pack_block

real_async_client = httpx.AsyncClient
DIFF = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b"


def _patch_http(handler):
    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    return patch("httpx.AsyncClient", factory)


# pack-agent-composer-298bb76630
class SuggestedCommandSpanTests(unittest.TestCase):
    def test_backticks_in_a_command_cannot_close_the_span(self) -> None:
        pack = {"work_item_id": "pack-x-1", "narrative": {"title": "t"}, "repo": {"name": "r"},
                "verification": {"suggested_test_commands": ["pytest -k `echo foo`"]}}
        block = build_pack_block(pack, "").block
        line = next(l for l in block.splitlines() if "echo foo" in l)
        self.assertIn("`` pytest -k `echo foo` ``", line)


if __name__ == "__main__":
    unittest.main()

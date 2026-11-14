"""The history page renders untrusted titles and summaries as text, not markup."""

import re
import unittest

from app.routers.ui import history_ui


class HistoryPageTests(unittest.TestCase):
    def test_cards_are_not_built_with_innerhtml(self) -> None:
        html = history_ui().body.decode("utf-8")
        script = html[html.index("function render(convs)"):html.index("async function loadConversations")]

        # Only the grid reset may touch innerHTML; nothing interpolates into it.
        self.assertEqual(re.findall(r"(\w+)\.innerHTML\s*=", script), ["gridEl"])
        self.assertIn('gridEl.innerHTML = ""', script)
        self.assertIn("textContent", script)
        self.assertIn("dataset.open = c.id", script)
        self.assertNotIn("${c.title", script.replace("{{", "{"))


if __name__ == "__main__":
    unittest.main()

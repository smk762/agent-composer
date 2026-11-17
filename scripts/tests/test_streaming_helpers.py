"""scripts/test_streaming.py: stream-chunk parsing helpers. Stdlib only:

    python3 -m unittest discover -s scripts/tests
"""

import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "streaming_smoke", Path(__file__).resolve().parent.parent / "test_streaming.py")
streaming_smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(streaming_smoke)


class DeltaTests(unittest.TestCase):
    def test_empty_or_missing_choices(self) -> None:
        self.assertEqual(streaming_smoke._delta({"choices": [], "usage": {}}), {})
        self.assertEqual(streaming_smoke._delta({}), {})

    def test_null_content_is_not_content(self) -> None:
        delta = streaming_smoke._delta({"choices": [{"delta": {"role": "assistant", "content": None}}]})
        self.assertIsNone(delta.get("content"))


class ToolCallMergeTests(unittest.TestCase):
    def test_fragments_merge_by_index(self) -> None:
        acc: dict = {}
        streaming_smoke._merge_tool_calls(acc, [
            {"index": 0, "id": "c1", "function": {"name": "search", "arguments": '{"q":'}}])
        streaming_smoke._merge_tool_calls(acc, [{"index": 0, "function": {"arguments": ' "x"}'}}])
        self.assertEqual(acc, {0: {"id": "c1", "name": "search", "arguments": '{"q": "x"}'}})


class NotCollectedTests(unittest.TestCase):
    def test_live_checks_are_not_test_prefixed(self) -> None:
        # pytest would collect test_* and ignore their False returns.
        self.assertFalse([n for n in vars(streaming_smoke) if n.startswith("test_")])


if __name__ == "__main__":
    unittest.main()

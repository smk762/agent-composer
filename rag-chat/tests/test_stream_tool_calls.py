"""Streamed tool calls are accumulated across chunks and stored as JSON. No network."""

import asyncio
import json
import unittest

from app.models.schemas import ToolCall, ToolCallFunction
from app.routers.chat import _tool_calls_json
from app.streaming import StreamError, stream_ollama_chat


class _Response:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _Stream:
    def __init__(self, lines):
        self._resp = _Response(lines)

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _Client:
    def __init__(self, lines):
        self._lines = lines

    def stream(self, method, url, json=None):
        return _Stream(self._lines)


def _call(name, args):
    return {"function": {"name": name, "arguments": args}}


def _collect(lines):
    async def go():
        return [c async for c in stream_ollama_chat("http://o", {}, "m", client=_Client(lines))]

    return asyncio.run(go())


class StreamToolCallTests(unittest.TestCase):
    def test_calls_in_separate_chunks_are_all_kept(self) -> None:
        lines = [
            json.dumps({"message": {"tool_calls": [_call("a", {"x": 1})]}, "done": False}),
            json.dumps({"message": {"tool_calls": [_call("b", {"y": 2})]}, "done": False}),
            json.dumps({"message": {}, "done": True}),
        ]

        async def collect():
            return [c async for c in stream_ollama_chat("http://o", {}, "m", client=_Client(lines))]

        chunks = asyncio.run(collect())
        calls = [tc for c in chunks for tc in (c.choices[0].delta.tool_calls or [])]
        self.assertEqual([tc["function"]["name"] for tc in calls], ["a", "b"])
        # What chat.py stores must load back as ToolCall, as GET /conversations/{id} does.
        stored = json.loads(_tool_calls_json(calls))
        self.assertEqual([ToolCall(**tc).function.name for tc in stored], ["a", "b"])

    def test_tool_call_deltas_carry_an_index(self) -> None:
        lines = [
            json.dumps({"message": {"tool_calls": [_call("a", {}), _call("b", {})]}, "done": False}),
            json.dumps({"message": {}, "done": True}),
        ]
        calls = [tc for c in _collect(lines) for tc in (c.choices[0].delta.tool_calls or [])]
        self.assertEqual([tc["index"] for tc in calls], [0, 1])

    def test_mid_stream_error_line_raises(self) -> None:
        lines = [
            json.dumps({"message": {"content": "partial"}, "done": False}),
            json.dumps({"error": "model runner has unexpectedly stopped"}),
        ]
        with self.assertRaisesRegex(StreamError, "unexpectedly stopped"):
            _collect(lines)

    def test_stream_closed_before_done_raises(self) -> None:
        lines = [json.dumps({"message": {"content": "partial"}, "done": False})]
        with self.assertRaises(StreamError):
            _collect(lines)

    def test_completed_stream_ends_with_stop(self) -> None:
        lines = [
            json.dumps({"message": {"content": "hi"}, "done": False}),
            json.dumps({"message": {}, "done": True}),
        ]
        self.assertEqual(_collect(lines)[-1].choices[0].finish_reason, "stop")

    def test_pydantic_tool_calls_serialise(self) -> None:
        tc = ToolCall(id="c1", type="function", function=ToolCallFunction(name="f", arguments="{}"))
        stored = json.loads(_tool_calls_json([tc]))
        self.assertEqual(stored[0]["function"]["name"], "f")

    def test_no_calls_is_none(self) -> None:
        self.assertIsNone(_tool_calls_json([]))

    def test_callers_payload_is_not_mutated(self) -> None:
        payload = {"model": "m", "stream": False}

        async def go():
            lines = [json.dumps({"message": {"content": "hi"}, "done": True})]
            return [c async for c in stream_ollama_chat("http://o", payload, "m", client=_Client(lines))]

        asyncio.run(go())
        self.assertEqual(payload, {"model": "m", "stream": False})


if __name__ == "__main__":
    unittest.main()

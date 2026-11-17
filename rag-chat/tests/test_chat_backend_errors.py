"""/chat and its Ollama helpers: backend failures are a generic 502 (Ollama's status and body stay
in the log), a request that fails on the model stores no conversation, and malformed stored
tool_calls don't break a conversation fetch. No network, no database."""

import contextlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from app.models.orm import Conversation, Message
from app.models.schemas import AuthContext, ChatRequest, Msg
from app.routers import chat

real_async_client = httpx.AsyncClient
AUTH = AuthContext(user_id="u1", via="dev")
SECRET = "model 'internal-model' not found on gpu-node-7"


class _Result:
    def __init__(self, value=None, rows=()):
        self._value = value
        self._rows = list(rows)

    def scalar_one_or_none(self):
        return self._value

    def scalars(self):
        return SimpleNamespace(all=lambda: self._rows)

    def fetchall(self):
        return []


class FakeDB:
    def __init__(self, conv=None, messages=()):
        self.conv = conv
        self.rows = list(messages)
        self.added = []
        self.commits = 0
        self.no_autoflush = contextlib.nullcontext()

    async def execute(self, stmt):
        sql = str(stmt)
        if sql.startswith("SELECT conversations.id, conversations.user_id"):
            return _Result(self.conv)
        if sql.startswith("SELECT messages.id"):
            return _Result(rows=self.rows)
        return _Result(None)

    def add(self, obj):
        if isinstance(obj, Conversation) and obj.id is None:
            obj.id = "conv-new"
        self.added.append(obj)

    def add_all(self, objs):
        self.added.extend(objs)

    async def flush(self):
        pass

    async def commit(self):
        self.commits += 1

    async def refresh(self, obj):
        pass

    def conversations(self):
        return [o for o in self.added if isinstance(o, Conversation)]


class _Base(unittest.IsolatedAsyncioTestCase):
    def ollama(self, handler):
        """Every httpx.AsyncClient answers through ``handler``; leases go nowhere."""

        @contextlib.asynccontextmanager
        async def fake_lease(*a, **kw):
            yield "http://ollama.test"

        async def no_ctx(msgs):
            return None

        async def resolve(m):
            return "m"

        def client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_async_client(*args, **kwargs)

        for p in (
            patch.object(chat, "ollama_lease", fake_lease),
            patch.object(chat, "rag_context_for", no_ctx),
            patch.object(chat, "_resolve_model", resolve),
            patch("httpx.AsyncClient", client),
            patch.object(chat, "CHAT_HISTORY_MAX_MSGS", 0),
        ):
            p.start()
            self.addCleanup(p.stop)

    def assert_generic_502(self, exc: HTTPException):
        self.assertEqual(exc.status_code, 502)
        self.assertNotIn("gpu-node-7", str(exc.detail))
        self.assertNotIn("Tried", str(exc.detail))


class BackendErrorTests(_Base):
    async def test_embed_backend_error_is_a_generic_502(self) -> None:
        self.ollama(lambda req: httpx.Response(404, text=SECRET))
        with self.assertRaises(HTTPException) as cm:
            await chat.ollama_embed("q")
        self.assert_generic_502(cm.exception)

    async def test_tags_backend_error_is_a_generic_502(self) -> None:
        self.ollama(lambda req: httpx.Response(500, text=SECRET))
        with self.assertRaises(HTTPException) as cm:
            await chat._ollama_tag_names()
        self.assert_generic_502(cm.exception)

    async def test_chat_backend_error_is_a_generic_502(self) -> None:
        self.ollama(lambda req: httpx.Response(400, text=SECRET))
        db = FakeDB()
        req = ChatRequest(messages=[Msg(role="user", content="hi")], stream=False)
        with self.assertRaises(HTTPException) as cm:
            await chat.chat(req, AUTH, db)
        self.assert_generic_502(cm.exception)


class NoEmptyConversationTests(_Base):
    async def test_failed_non_streaming_request_stores_no_conversation(self) -> None:
        self.ollama(lambda req: httpx.Response(500, text="boom"))
        db = FakeDB()
        req = ChatRequest(messages=[Msg(role="user", content="hi")], stream=False)
        with self.assertRaises(HTTPException):
            await chat.chat(req, AUTH, db)
        self.assertEqual(db.conversations(), [])
        self.assertEqual(db.commits, 0)

    async def test_unknown_model_in_streaming_stores_no_conversation(self) -> None:
        self.ollama(lambda req: httpx.Response(404, json={"error": "not found"}))
        db = FakeDB()
        req = ChatRequest(messages=[Msg(role="user", content="hi")], stream=True)
        resp = await chat.chat(req, AUTH, db)
        events = [e async for e in resp.body_iterator]
        self.assertIn("Model not found", events[0])
        self.assertEqual(db.conversations(), [])
        self.assertEqual(db.commits, 0)

    async def test_successful_non_streaming_request_stores_the_conversation(self) -> None:
        self.ollama(lambda req: httpx.Response(200, json={"message": {"content": "hello"}}))
        db = FakeDB()
        req = ChatRequest(messages=[Msg(role="user", content="hi")], stream=False)
        out = await chat.chat(req, AUTH, db)
        self.assertEqual(out.conversation_id, "conv-new")
        self.assertEqual(len(db.conversations()), 1)


class MalformedToolCallsTests(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_tool_calls_are_skipped(self) -> None:
        conv = Conversation(id="c1", user_id="u1", title=None, summary=None, model="m")
        good = [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        stored = ['"x"', "5", '{"a": 1}', "[1]", "[null]", "not json", json.dumps(good)]
        rows = [Message(conversation_id="c1", role="assistant", content=str(i), tool_calls=tc, seq=i)
                for i, tc in enumerate(stored)]
        out = await chat.get_conversation("c1", AUTH, FakeDB(conv=conv, messages=rows))
        self.assertEqual([m.tool_calls for m in out.messages[:-1]], [None] * (len(stored) - 1))
        self.assertEqual(out.messages[-1].tool_calls[0].function.name, "f")


if __name__ == "__main__":
    unittest.main()

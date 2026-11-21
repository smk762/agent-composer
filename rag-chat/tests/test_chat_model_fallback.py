"""/chat model fallback: the streaming probe asks the leased Ollama (not OLLAMA_URL), and the
conversation stores the model that actually answered. No network, no database."""

import contextlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from sqlalchemy.dialects import postgresql

from app import streaming
from app.models.orm import Conversation
from app.models.schemas import AuthContext, ChatRequest, Msg
from app.routers import chat

real_async_client = httpx.AsyncClient
AUTH = AuthContext(user_id="u1", via="dev")
HOME, SPILL = "http://home.test", "http://spill.test"


class _Result:
    def __init__(self, value=None):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def scalars(self):
        return SimpleNamespace(all=lambda: [])

    def fetchall(self):
        return []


class FakeDB:
    def __init__(self, conv=None):
        self.conv = conv
        self.added = []
        self.no_autoflush = contextlib.nullcontext()

    async def execute(self, stmt):
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        if sql.startswith("SELECT conversations.id, conversations.user_id"):
            return _Result(self.conv)
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
        pass

    async def refresh(self, obj):
        pass

    def conversations(self):
        return [o for o in self.added if isinstance(o, Conversation)]


class ModelFallbackTests(unittest.IsolatedAsyncioTestCase):
    """HOME (OLLAMA_URL) has ``m``; the leased SPILL node only has ``m:latest``."""

    def setUp(self) -> None:
        self.streamed: list[tuple[str, str]] = []
        self.chat_calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            host = f"{request.url.scheme}://{request.url.host}"
            body = json.loads(request.content or b"{}")
            name = body.get("name") or body.get("model")
            have = {"m"} if host == HOME else {"m:latest"}
            if request.url.path == "/api/show":
                return httpx.Response(200 if name in have else 404, json={})
            if request.url.path == "/api/chat":
                self.chat_calls.append(name)
                if name in have:
                    return httpx.Response(200, json={"message": {"content": "hello"}})
                return httpx.Response(404, json={"error": f"model '{name}' not found"})
            return httpx.Response(404, json={})

        def client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_async_client(*args, **kwargs)

        @contextlib.asynccontextmanager
        async def spill_lease(*a, **kw):
            yield SPILL

        async def fake_stream(base_url, payload, model, conv_id):
            self.streamed.append((base_url, model))
            delta = SimpleNamespace(content="hi", tool_calls=None)
            yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)],
                                  model_dump_json=lambda **kw: '{"c": "hi"}')

        async def resolve(m):
            return m

        async def no_ctx(msgs):
            return None

        async def no_save(*a, **kw):
            return None

        for p in (
            patch.object(streaming, "stream_ollama_chat", fake_stream),
            patch.object(chat, "ollama_lease", spill_lease),
            patch.object(chat, "OLLAMA_URL", HOME),
            patch.object(chat, "_resolve_model", resolve),
            patch.object(chat, "rag_context_for", no_ctx),
            patch.object(chat, "_save_stream_turn", no_save),
            patch("httpx.AsyncClient", client),
            patch.object(chat, "CHAT_HISTORY_MAX_MSGS", 0),
        ):
            p.start()
            self.addCleanup(p.stop)

    async def test_stream_probe_uses_the_leased_backend(self) -> None:
        db = FakeDB()
        req = ChatRequest(messages=[Msg(role="user", content="hi")], stream=True, model="m")
        resp = await chat.chat(req, AUTH, db)
        events = [e async for e in resp.body_iterator]
        self.assertNotIn("Model not found", "".join(events))
        self.assertEqual(self.streamed, [(SPILL, "m:latest")])
        self.assertEqual(db.conversations()[0].model, "m:latest")

    async def test_non_stream_conversation_stores_the_fallback_model(self) -> None:
        conv = Conversation(id="c1", user_id="u1", model="m")
        db = FakeDB(conv)
        req = ChatRequest(messages=[Msg(role="user", content="hi")], stream=False, conversation_id="c1")
        out = await chat.chat(req, AUTH, db)
        self.assertEqual(out.model, "m:latest")
        self.assertEqual(self.chat_calls, ["m", "m:latest"])
        self.assertEqual(conv.model, "m:latest")


if __name__ == "__main__":
    unittest.main()

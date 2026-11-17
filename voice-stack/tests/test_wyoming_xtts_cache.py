"""wyoming_bridge.xtts_tts caches: deleted clones drop out of _CLONE_KEYS
(4cbf1941c9), and concurrent first calls fill _SPEAKER_CACHE once (1186fef1b7)."""

import asyncio

import pytest

pytest.importorskip("wyoming")

from wyoming_bridge import xtts_tts as tts


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _fake_client(monkeypatch, handler):
    calls = []

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url):
            calls.append(url)
            return await handler(url)

    monkeypatch.setattr(tts.httpx, "AsyncClient", _Client)
    return calls


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    monkeypatch.setattr(tts, "_CLONE_KEYS", {})
    monkeypatch.setattr(tts, "_SPEAKER_CACHE", [])
    if hasattr(tts, "_SPEAKER_LOCK"):
        monkeypatch.setattr(tts, "_SPEAKER_LOCK", asyncio.Lock())


def _clones(*ids):
    return {"clones": [{"companion_id": i, "voice_clone_id": f"voice_clones/{i}/v2.pt"} for i in ids]}


def test_deleted_clone_drops_out_of_the_key_cache(monkeypatch):
    listing = {"payload": _clones("alice", "bob")}

    async def handler(url):
        return _Resp(listing["payload"])

    _fake_client(monkeypatch, handler)
    asyncio.run(tts._fetch_clones())
    assert tts._clone_key("bob") == "voice_clones/bob/v2.pt"

    listing["payload"] = _clones("alice")  # bob deleted on the XTTS side
    asyncio.run(tts._fetch_clones())
    assert tts._CLONE_KEYS == {"alice": "voice_clones/alice/v2.pt"}
    assert tts._clone_key("bob") == "voice_clones/bob/prompt.pt"


def test_failed_clone_fetch_keeps_the_cache(monkeypatch):
    tts._CLONE_KEYS["alice"] = "voice_clones/alice/v2.pt"

    async def handler(url):
        return _Resp({}, status=503)

    _fake_client(monkeypatch, handler)
    assert asyncio.run(tts._fetch_clones()) == []
    assert tts._CLONE_KEYS == {"alice": "voice_clones/alice/v2.pt"}


def test_concurrent_first_calls_fetch_speakers_once(monkeypatch):
    async def handler(url):
        await asyncio.sleep(0.05)
        return _Resp({"speakers": [{"name": "Ana"}, {"name": "Ben"}]})

    calls = _fake_client(monkeypatch, handler)

    async def run():
        return await asyncio.gather(*(tts._fetch_speakers() for _ in range(5)))

    results = asyncio.run(run())
    assert len(calls) == 1
    assert tts._SPEAKER_CACHE == ["Ana", "Ben"]
    assert all(r == ["Ana", "Ben"] for r in results)

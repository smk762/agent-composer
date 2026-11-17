"""whisper_server idle eviction: never mid-transcription, and never skipped.

No model is loaded: ``_load_model`` and ``_transcribe_sync`` are replaced, so
these run without faster-whisper or a GPU.
"""

import asyncio
import threading
import time

import pytest

import whisper_server as ws


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(ws, "_model", None)
    monkeypatch.setattr(ws, "_last_used", 0.0)
    monkeypatch.setattr(ws, "_active", 0)
    monkeypatch.setattr(ws, "_evict_task", None)
    monkeypatch.setattr(ws, "_lock", asyncio.Lock())
    monkeypatch.setattr(ws, "_load_model", lambda: object())
    unloads = []
    real_unload = ws._unload_model

    def counting_unload():
        unloads.append(ws._model)
        real_unload()

    monkeypatch.setattr(ws, "_unload_model", counting_unload)
    return unloads


async def _until(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


def test_model_is_not_evicted_while_a_transcription_runs(fake_model, monkeypatch):
    """4f67369a02: a transcription longer than the keep-alive keeps its model."""
    monkeypatch.setattr(ws, "STT_KEEP_ALIVE_GPU", 0.05)
    release = threading.Event()
    seen = {}

    def slow_transcribe(path, language, model):
        release.wait(5)
        seen["model_at_end"] = ws._model
        return {"text": "", "language": "en", "duration": 0, "duration_seconds": 0, "confidence": 1}

    monkeypatch.setattr(ws, "_transcribe_sync", slow_transcribe)

    async def run():
        t = asyncio.create_task(ws._run_transcription("x.wav", None, "/transcribe"))
        await _until(lambda: ws._active == 1)
        await asyncio.sleep(0.2)  # four keep-alives
        assert ws._model is not None, "evicted mid-transcription"
        release.set()
        await t
        assert seen["model_at_end"] is not None
        # idle time counts from the end of the work: unloaded only after that
        await _until(lambda: ws._model is None)

    asyncio.run(run())
    assert len(fake_model) == 1


def test_zero_keep_alive_still_waits_for_the_transcription(fake_model, monkeypatch):
    monkeypatch.setattr(ws, "STT_KEEP_ALIVE_GPU", 0)
    release = threading.Event()

    def slow_transcribe(path, language, model):
        release.wait(5)
        return {"text": "", "language": "en", "duration": 0, "duration_seconds": 0, "confidence": 1}

    monkeypatch.setattr(ws, "_transcribe_sync", slow_transcribe)

    async def run():
        t = asyncio.create_task(ws._run_transcription("x.wav", None, "/transcribe"))
        await _until(lambda: ws._active == 1)
        await asyncio.sleep(0.05)
        assert ws._model is not None
        release.set()
        await t
        await _until(lambda: ws._model is None)

    asyncio.run(run())


def test_eviction_rechecks_instead_of_giving_up(fake_model, monkeypatch):
    """16869a07fb: a wake-up that finds the model used since (without a restart)
    sleeps for the rest of the keep-alive instead of leaving the model loaded."""
    monkeypatch.setattr(ws, "STT_KEEP_ALIVE_GPU", 0.1)

    async def run():
        await ws._ensure_model()
        await asyncio.sleep(0.05)
        # A use that refreshed _last_used without restarting the timer (same clock as the server).
        clock = time.time if ws._last_used > 1e9 else time.monotonic
        monkeypatch.setattr(ws, "_last_used", clock())
        await _until(lambda: ws._model is None, timeout=1.0)

    asyncio.run(run())


def test_early_wakeup_still_unloads(fake_model, monkeypatch):
    """16869a07fb: the loop's sleep may end a hair before the keep-alive has passed."""
    monkeypatch.setattr(ws, "STT_KEEP_ALIVE_GPU", 0.1)
    real_sleep = asyncio.sleep

    async def early_sleep(seconds, *a, **kw):
        await real_sleep(max(0.0, seconds - 0.02), *a, **kw)

    monkeypatch.setattr(ws.asyncio, "sleep", early_sleep)

    async def run():
        await ws._ensure_model()
        await _until(lambda: ws._model is None, timeout=1.0)

    asyncio.run(run())

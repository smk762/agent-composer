"""wyoming_bridge.whisper_stt: the per-connection audio buffer is bounded (00140d05e9)."""

import asyncio

import pytest

pytest.importorskip("wyoming")

from wyoming.audio import AudioChunk, AudioStart, AudioStop  # noqa: E402

from wyoming_bridge import whisper_stt as stt  # noqa: E402


class _Writer:
    def __init__(self):
        self.data = bytearray()

    def write(self, b):
        self.data += b

    def writelines(self, lines):
        for b in lines:
            self.data += b

    async def drain(self):
        pass

    def close(self):
        pass

    async def wait_closed(self):
        pass


def _handler():
    return stt.WhisperEventHandler(asyncio.StreamReader(), _Writer())


def _chunk(seconds: float) -> AudioChunk:
    n = int(seconds * stt.TARGET_RATE)
    return AudioChunk(rate=stt.TARGET_RATE, width=2, channels=1, audio=b"\x01\x00" * n)


def test_buffer_stops_growing_at_the_cap(monkeypatch):
    monkeypatch.setattr(stt, "MAX_SECONDS", 1.0, raising=False)
    monkeypatch.setattr(stt, "MAX_BUFFER_BYTES", stt.TARGET_RATE * 2, raising=False)

    async def run():
        h = _handler()
        await h.handle_event(AudioStart(rate=stt.TARGET_RATE, width=2, channels=1).event())
        for _ in range(50):
            await h.handle_event(_chunk(0.3).event())
        return h

    h = asyncio.run(run())
    assert len(h._buffer) == stt.TARGET_RATE * 2


def test_capped_stream_still_transcribes_its_first_seconds(monkeypatch):
    monkeypatch.setattr(stt, "MAX_SECONDS", 1.0, raising=False)
    monkeypatch.setattr(stt, "MAX_BUFFER_BYTES", stt.TARGET_RATE * 2, raising=False)
    sent = []

    async def fake_transcribe(self):
        sent.append(len(self._buffer))
        return "hi", "en"

    monkeypatch.setattr(stt.WhisperEventHandler, "_transcribe", fake_transcribe)

    async def run():
        h = _handler()
        await h.handle_event(AudioStart(rate=stt.TARGET_RATE, width=2, channels=1).event())
        for _ in range(10):
            await h.handle_event(_chunk(0.3).event())
        await h.handle_event(AudioStop().event())
        return h

    h = asyncio.run(run())
    assert sent == [stt.TARGET_RATE * 2]
    assert len(h._buffer) == 0


def test_short_utterance_is_buffered_whole():
    async def run():
        h = _handler()
        await h.handle_event(AudioStart(rate=stt.TARGET_RATE, width=2, channels=1).event())
        await h.handle_event(_chunk(0.5).event())
        return h

    assert len(asyncio.run(run())._buffer) == stt.TARGET_RATE  # 0.5 s of 16-bit mono

"""wyoming_bridge.whisper_stt: the per-connection audio buffer is bounded (00140d05e9)."""

import asyncio

import pytest

pytest.importorskip("wyoming")

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming_bridge import whisper_stt as stt


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


# ── 8247e6a4b1 / e3150402f8: per-request language and buffer ─────────────────


def _record_requests(monkeypatch):
    sent = []

    async def fake_transcribe(self):
        sent.append((self._language, bytes(self._buffer)))
        return "hi", self._language or None

    monkeypatch.setattr(stt.WhisperEventHandler, "_transcribe", fake_transcribe)
    return sent


async def _utterance(h, language=None, seconds=0.1):
    from wyoming.asr import Transcribe

    await h.handle_event(Transcribe(language=language).event())
    await h.handle_event(AudioStart(rate=stt.TARGET_RATE, width=2, channels=1).event())
    await h.handle_event(_chunk(seconds).event())
    await h.handle_event(AudioStop().event())


def test_language_does_not_stick_across_requests(monkeypatch):
    monkeypatch.setattr(stt, "DEFAULT_LANGUAGE", "")
    sent = _record_requests(monkeypatch)

    async def run():
        h = _handler()
        await _utterance(h, "de")
        await _utterance(h, None)

    asyncio.run(run())
    assert [lang for lang, _ in sent] == ["de", ""]


def test_language_falls_back_to_the_configured_default(monkeypatch):
    monkeypatch.setattr(stt, "DEFAULT_LANGUAGE", "en")
    sent = _record_requests(monkeypatch)

    async def run():
        h = _handler()
        await _utterance(h, "fr")
        await _utterance(h, None)
        assert h._language == "en"

    asyncio.run(run())
    assert [lang for lang, _ in sent] == ["fr", "en"]


def test_second_audio_stop_does_not_resend_the_utterance(monkeypatch):
    sent = _record_requests(monkeypatch)

    async def run():
        h = _handler()
        await _utterance(h)
        await h.handle_event(AudioStop().event())  # no new audio
        await h.handle_event(_chunk(0.05).event())  # chunks without a fresh audio-start
        await h.handle_event(AudioStop().event())

    asyncio.run(run())
    assert len(sent[0][1]) == int(0.1 * stt.TARGET_RATE) * 2
    assert sent[1][1] == b""
    assert len(sent[2][1]) == int(0.05 * stt.TARGET_RATE) * 2


def test_buffer_is_cleared_even_when_transcription_raises(monkeypatch):
    async def failing(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(stt.WhisperEventHandler, "_transcribe", failing)

    async def run():
        h = _handler()
        await h.handle_event(AudioStart(rate=stt.TARGET_RATE, width=2, channels=1).event())
        await h.handle_event(_chunk(0.1).event())
        with pytest.raises(RuntimeError):
            await h.handle_event(AudioStop().event())
        return h

    assert len(asyncio.run(run())._buffer) == 0

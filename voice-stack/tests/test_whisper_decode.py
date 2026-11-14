"""ffmpeg decode path and decoder-backed /health (whisper_server, issue #42)."""

import shutil
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

import whisper_server as ws

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(autouse=True)
def _reset_decoder_cache():
    ws._decoder_check.update(at=0.0, result=None)
    yield
    ws._decoder_check.update(at=0.0, result=None)


def _tone_wav(path, rate=24000, seconds=0.5):
    t = np.arange(int(rate * seconds)) / rate
    pcm = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


@needs_ffmpeg
def test_decode_resamples_24k_wav_to_16k_float(tmp_path):
    p = tmp_path / "tone.wav"
    _tone_wav(p)
    audio = ws._decode_audio(str(p))
    assert audio.dtype == np.float32
    assert abs(len(audio) - 8000) < 80
    assert 0.3 < np.abs(audio).max() <= 1.0


@needs_ffmpeg
def test_decode_garbage_raises_decode_error(tmp_path):
    p = tmp_path / "junk.wav"
    p.write_bytes(b"not audio at all")
    with pytest.raises(ws.AudioDecodeError):
        ws._decode_audio(str(p))


def test_transcribe_passes_decoded_samples_not_path(tmp_path, monkeypatch):
    samples = np.zeros(16000, dtype=np.float32)
    monkeypatch.setattr(ws, "_decode_audio", lambda path: samples)
    seen = {}

    class Info:
        language = "en"
        duration = 1.0
        language_probability = 0.9

    class Seg:
        text = " hello "

    class Model:
        def transcribe(self, audio, **kw):
            seen["audio"] = audio
            return [Seg()], Info()

    out = ws._transcribe_sync("ignored.wav", None, Model())
    assert seen["audio"] is samples
    assert out["text"] == "hello"


def test_undecodable_upload_is_400(monkeypatch):
    async def fake_model():
        return object()

    def bad_decode(path):
        raise ws.AudioDecodeError("ffmpeg exit 1: Invalid data")

    monkeypatch.setattr(ws, "_ensure_model", fake_model)
    monkeypatch.setattr(ws, "_decode_audio", bad_decode)
    r = TestClient(ws.app).post("/transcribe", files={"audio": ("x.wav", b"junk", "audio/wav")})
    assert r.status_code == 400
    assert "Could not decode audio" in r.json()["error"]


@needs_ffmpeg
def test_health_ok_when_decoder_works():
    r = TestClient(ws.app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["decoder"] == {"ok": True}
    assert body["model_state"] == "unloaded"


def test_health_503_when_decoder_broken(monkeypatch):
    def broken(path):
        raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")

    monkeypatch.setattr(ws, "_decode_audio", broken)
    r = TestClient(ws.app).get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "error"
    assert "metadata_errors" in body["decoder"]["error"]


def test_health_caches_decoder_check(monkeypatch):
    calls = []

    def counting(path):
        calls.append(path)
        return np.zeros(16000, dtype=np.float32)

    monkeypatch.setattr(ws, "_decode_audio", counting)
    client = TestClient(ws.app)
    client.get("/health")
    client.get("/health")
    assert len(calls) == 1

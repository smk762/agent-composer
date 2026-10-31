import contextlib
import io
import unittest
import wave
from unittest import mock

import httpx
from fastapi import HTTPException

from app.routers import voice


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"audio_url": None, "duration": 1.0}


class _FakeClient:
    """Stands in for httpx.AsyncClient on the single-shot synth path."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def post(self, url: str, json: dict) -> _FakeResponse:
        return _FakeResponse()


def _req() -> voice.SynthesiseRequest:
    return voice.SynthesiseRequest(text="One. Two. Three.", pause_ms=200)


def _pcm_wav(*, rate: int = 24000, channels: int = 1, width: int = 2, frames: int = 240) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x00" * (frames * channels * width))
    return buf.getvalue()


@contextlib.contextmanager
def _synth_patches(paced: mock.AsyncMock):
    with (
        mock.patch.object(voice, "_require_tts"),
        mock.patch.object(voice, "VOICE_S3_REHOST", True),
        mock.patch.object(voice, "tts_engine_chain", return_value=[("xtts", "http://xtts")]),
        mock.patch.object(voice, "_synthesise_paced", paced),
        mock.patch.object(voice.httpx, "AsyncClient", _FakeClient),
    ):
        yield


class ConcatWavTests(unittest.TestCase):
    def test_matching_wavs_insert_silence(self) -> None:
        combined, seconds = voice._concat_wavs_with_gaps(
            [_pcm_wav(frames=240), _pcm_wav(frames=240)],
            gap_ms=100,
        )
        with wave.open(io.BytesIO(combined), "rb") as w:
            self.assertEqual(w.getframerate(), 24000)
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getsampwidth(), 2)
            # 240 + 100ms silence (2400 frames) + 240
            self.assertEqual(w.getnframes(), 240 + 2400 + 240)
        self.assertAlmostEqual(seconds, (240 + 2400 + 240) / 24000)

    def test_mismatched_format_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            voice._concat_wavs_with_gaps(
                [_pcm_wav(rate=24000), _pcm_wav(rate=16000)],
                gap_ms=50,
            )
        self.assertIn("format mismatch", str(ctx.exception))


class PacedFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_wav_or_fetch_failure_falls_back_to_single_shot(self) -> None:
        for exc in (wave.Error("not PCM"), httpx.ReadError("audio fetch failed"), ValueError("WAV format mismatch")):
            with self.subTest(exc=type(exc).__name__):
                paced = mock.AsyncMock(side_effect=exc)
                with _synth_patches(paced):
                    data = await voice.synthesise(_req())
                paced.assert_awaited_once()
                self.assertEqual(data["duration"], 1.0)
                self.assertNotIn("paced", data)

    async def test_http_exception_from_pacing_is_not_swallowed(self) -> None:
        paced = mock.AsyncMock(side_effect=HTTPException(503, detail="No reachable TTS engine"))
        with _synth_patches(paced), self.assertRaises(HTTPException) as ctx:
            await voice.synthesise(_req())
        self.assertEqual(ctx.exception.status_code, 503)

    async def test_unexpected_errors_surface(self) -> None:
        paced = mock.AsyncMock(side_effect=RuntimeError("bug"))
        with _synth_patches(paced), self.assertRaises(RuntimeError):
            await voice.synthesise(_req())


if __name__ == "__main__":
    unittest.main()

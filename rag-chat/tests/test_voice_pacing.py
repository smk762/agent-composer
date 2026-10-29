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


class PacedFallbackTests(unittest.IsolatedAsyncioTestCase):
    def _patches(self, paced: mock.AsyncMock):
        return (
            mock.patch.object(voice, "_require_tts"),
            mock.patch.object(voice, "VOICE_S3_REHOST", True),
            mock.patch.object(voice, "tts_engine_chain", return_value=[("xtts", "http://xtts")]),
            mock.patch.object(voice, "_synthesise_paced", paced),
            mock.patch.object(voice.httpx, "AsyncClient", _FakeClient),
        )

    async def _run(self, paced: mock.AsyncMock):
        patches = self._patches(paced)
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return await voice.synthesise(_req())

    async def test_wav_or_fetch_failure_falls_back_to_single_shot(self) -> None:
        for exc in (wave.Error("not PCM"), httpx.ReadError("audio fetch failed")):
            with self.subTest(exc=type(exc).__name__):
                paced = mock.AsyncMock(side_effect=exc)
                data = await self._run(paced)
                paced.assert_awaited_once()
                self.assertEqual(data["duration"], 1.0)
                self.assertNotIn("paced", data)
                mock.patch.stopall()

    async def test_http_exception_from_pacing_is_not_swallowed(self) -> None:
        paced = mock.AsyncMock(side_effect=HTTPException(503, detail="No reachable TTS engine"))
        with self.assertRaises(HTTPException) as ctx:
            await self._run(paced)
        self.assertEqual(ctx.exception.status_code, 503)

    async def test_unexpected_errors_surface(self) -> None:
        paced = mock.AsyncMock(side_effect=RuntimeError("bug"))
        with self.assertRaises(RuntimeError):
            await self._run(paced)


if __name__ == "__main__":
    unittest.main()

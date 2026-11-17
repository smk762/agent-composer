"""Wyoming ASR bridge → Whisper sidecar.

Speaks the Wyoming protocol on a TCP port, buffers the incoming audio stream
(`audio-start` → `audio-chunk`* → `audio-stop`), resamples it to 16 kHz mono
16-bit, and POSTs it to the Whisper sidecar's `POST /transcribe` multipart
endpoint. The resulting text is returned as a Wyoming `transcript`.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import wave
from functools import partial
from typing import Optional

import httpx
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, AsrModel, AsrProgram, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger("wyoming_whisper")

WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper:8030").rstrip("/")
MODEL_NAME = os.getenv("STT_MODEL", "large-v3-turbo")
DEFAULT_LANGUAGE = os.getenv("WYOMING_STT_LANGUAGE", "").strip()  # empty → autodetect
REQUEST_TIMEOUT = float(os.getenv("WYOMING_STT_TIMEOUT", "120"))

# Whisper works best on 16 kHz mono 16-bit; normalise whatever HA sends.
TARGET_RATE = 16000
TARGET_WIDTH = 2
TARGET_CHANNELS = 1

# Longest utterance buffered per audio stream. Past it further chunks are
# dropped (the first MAX_SECONDS are still transcribed), so a client that never
# sends audio-stop can't grow the buffer without bound.
MAX_SECONDS = float(os.getenv("WYOMING_STT_MAX_SECONDS", "120"))
MAX_BUFFER_BYTES = int(MAX_SECONDS * TARGET_RATE * TARGET_WIDTH * TARGET_CHANNELS)

LANGS = [
    "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja",
    "ko", "ar", "hi", "pl", "tr", "cs", "hu",
]


def _build_info() -> Info:
    attr = Attribution(
        name="faster-whisper", url="https://github.com/SYSTRAN/faster-whisper"
    )
    return Info(
        asr=[
            AsrProgram(
                name="whisper",
                description="faster-whisper STT (CTranslate2)",
                attribution=attr,
                installed=True,
                version=None,
                models=[
                    AsrModel(
                        name=MODEL_NAME,
                        description=MODEL_NAME,
                        attribution=attr,
                        installed=True,
                        version=None,
                        languages=list(LANGS),
                    )
                ],
            )
        ]
    )


def _pcm_to_wav(pcm: bytes, rate: int, width: int, channels: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setframerate(rate)
        wav_file.setsampwidth(width)
        wav_file.setnchannels(channels)
        wav_file.writeframes(pcm)
    return buf.getvalue()


class WhisperEventHandler(AsyncEventHandler):
    """Handles one Wyoming client connection."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._buffer = bytearray()
        self._truncated = False
        self._converter = AudioChunkConverter(
            rate=TARGET_RATE, width=TARGET_WIDTH, channels=TARGET_CHANNELS
        )
        self._language = DEFAULT_LANGUAGE

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(_build_info().event())
            return True

        if Transcribe.is_type(event.type):
            # Per request: a Transcribe without a language means the default
            # (or autodetect), never the previous request's language.
            self._language = Transcribe.from_event(event).language or DEFAULT_LANGUAGE
            return True

        if AudioStart.is_type(event.type):
            self._buffer = bytearray()
            self._truncated = False
            self._converter = AudioChunkConverter(
                rate=TARGET_RATE, width=TARGET_WIDTH, channels=TARGET_CHANNELS
            )
            return True

        if AudioChunk.is_type(event.type):
            room = MAX_BUFFER_BYTES - len(self._buffer)
            if room <= 0:
                if not self._truncated:
                    self._truncated = True
                    _LOGGER.warning("audio stream passed %.0f s; dropping the rest", MAX_SECONDS)
                return True
            chunk = self._converter.convert(AudioChunk.from_event(event))
            audio = chunk.audio
            if len(audio) > room:
                room -= room % (TARGET_WIDTH * TARGET_CHANNELS)  # whole samples only
                audio = audio[:room]
            self._buffer.extend(audio)
            return True

        if AudioStop.is_type(event.type):
            try:
                text, language = await self._transcribe()
            finally:
                # Don't hold (or re-send) the utterance, and don't carry this
                # request's language into the next one.
                self._buffer = bytearray()
                self._truncated = False
                self._language = DEFAULT_LANGUAGE
            await self.write_event(
                Transcript(text=text, language=language or None).event()
            )
            return True

        return True

    async def _transcribe(self) -> tuple[str, Optional[str]]:
        if not self._buffer:
            return "", None

        wav_bytes = _pcm_to_wav(
            bytes(self._buffer), TARGET_RATE, TARGET_WIDTH, TARGET_CHANNELS
        )
        params = {"language": self._language} if self._language else {}
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.post(
                    f"{WHISPER_URL}/transcribe",
                    files={"audio": ("speech.wav", wav_bytes, "audio/wav")},
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()
                return (data.get("text") or "").strip(), data.get("language")
        except Exception as exc:  # noqa: BLE001 — return empty transcript on failure
            _LOGGER.error("transcription failed: %s", exc)
            return "", None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Wyoming ASR bridge for Whisper")
    parser.add_argument("--uri", default=os.getenv("WYOMING_URI", "tcp://0.0.0.0:10300"))
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    _LOGGER.info("Wyoming Whisper bridge listening on %s → %s", args.uri, WHISPER_URL)

    server = AsyncServer.from_uri(args.uri)
    await server.run(partial(WhisperEventHandler))


if __name__ == "__main__":
    asyncio.run(main())

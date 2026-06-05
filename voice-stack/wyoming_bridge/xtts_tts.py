"""Wyoming TTS bridge → XTTS sidecar.

Speaks the Wyoming protocol on a TCP port and translates `synthesize` requests
into calls to the XTTS HTTP sidecar's streaming endpoint
(`POST /synthesise/stream`, raw 24 kHz mono float32 PCM). The float samples are
converted to signed 16-bit and re-emitted as Wyoming AudioChunks so Home
Assistant can consume them as a normal `tts` engine.

Voices advertised in the Wyoming `info` response:
  * one built-in XTTS studio speaker (``XTTS_DEFAULT_SPEAKER``)
  * every stored clone from ``GET /clones`` as ``clone:<companion_id>``

When HA selects a ``clone:<id>`` voice the bridge sends ``voice_clone_id`` to
XTTS; any other voice name is forwarded as a built-in ``speaker``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from functools import partial
from typing import Any, Optional

import httpx
import numpy as np
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

_LOGGER = logging.getLogger("wyoming_xtts")

XTTS_URL = os.getenv("XTTS_URL", "http://xtts:8031").rstrip("/")
DEFAULT_SPEAKER = os.getenv("XTTS_DEFAULT_SPEAKER", "Claribel Dervla")
DEFAULT_VOICE = os.getenv("WYOMING_TTS_DEFAULT_VOICE", "").strip()
DEFAULT_LANGUAGE = os.getenv("WYOMING_TTS_LANGUAGE", "en").strip() or "en"
REQUEST_TIMEOUT = float(os.getenv("WYOMING_TTS_TIMEOUT", "300"))
# Advertise all built-in studio speakers (off by default — fetching them wakes
# the XTTS GPU; clones + the default speaker are always advertised regardless).
LIST_SPEAKERS = os.getenv("WYOMING_TTS_LIST_SPEAKERS", "0").strip().lower() in (
    "1", "true", "yes", "on",
)

# XTTS native stream output is 24 kHz mono float32; Wyoming wants signed 16-bit.
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
CHANNELS = 1

# Languages XTTS-v2 supports (ISO codes accepted by the sidecar's normaliser).
SUPPORTED_LANGS = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
    "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
]


def _attribution() -> Attribution:
    return Attribution(name="Coqui XTTS-v2", url="https://github.com/coqui-ai/TTS")


def _voice(name: str, description: str) -> TtsVoice:
    return TtsVoice(
        name=name,
        description=description,
        attribution=_attribution(),
        installed=True,
        version=None,
        languages=list(SUPPORTED_LANGS),
    )


# companion_id → full S3 key (the value XTTS expects as `voice_clone_id`, e.g.
# "voice_clones/<id>/prompt.pt"). Populated whenever we list clones; used to
# resolve the key at synth time even across connections.
_CLONE_KEYS: dict[str, str] = {}
# Built-in studio speaker names, cached once (the list is static; fetching it
# wakes the XTTS GPU, so we only do it when WYOMING_TTS_LIST_SPEAKERS is set).
_SPEAKER_CACHE: list[str] = []


async def _fetch_clones() -> list[dict]:
    """List stored clones from XTTS. Cheap (S3 metadata) — safe on every Describe."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{XTTS_URL}/clones")
            resp.raise_for_status()
            clones = resp.json().get("clones", []) or []
    except Exception as exc:  # noqa: BLE001 — degrade gracefully if XTTS is down
        _LOGGER.warning("could not fetch clones from %s: %s", XTTS_URL, exc)
        return []

    for clone in clones:
        cid = (clone.get("companion_id") or "").strip()
        key = (clone.get("voice_clone_id") or "").strip()
        if cid and key:
            _CLONE_KEYS[cid] = key
    return clones


async def _fetch_speakers() -> list[str]:
    """Enumerate built-in studio speakers (cached). Wakes the XTTS GPU once."""
    if _SPEAKER_CACHE:
        return _SPEAKER_CACHE
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{XTTS_URL}/speakers")
            resp.raise_for_status()
            names = [s.get("name") for s in resp.json().get("speakers", []) if s.get("name")]
    except Exception as exc:  # noqa: BLE001 — model may still be loading
        _LOGGER.warning("could not fetch speakers from %s: %s", XTTS_URL, exc)
        return []
    _SPEAKER_CACHE.extend(names)
    return names


def _clone_key(companion_id: str) -> str:
    """Resolve an advertised clone id to the full S3 key XTTS loads it from.

    XTTS treats ``voice_clone_id`` as the literal S3 object key
    (``voice_clones/<companion_id>/prompt.pt``), not the bare id — passing the
    bare id yields a 422. Prefer the key learned from ``/clones``; fall back to
    the well-known layout when we have not listed clones on this connection.
    """
    cid = companion_id.strip()
    if "/" in cid:  # already a full key
        return cid
    return _CLONE_KEYS.get(cid) or f"voice_clones/{cid}/prompt.pt"


async def _build_info() -> Info:
    names: list[str] = await _fetch_speakers() if LIST_SPEAKERS else []
    if DEFAULT_SPEAKER not in names:
        names = [DEFAULT_SPEAKER, *names]

    voices: list[TtsVoice] = [
        _voice(name, f"XTTS studio speaker ({name})") for name in names
    ]
    for clone in await _fetch_clones():
        cid = (clone.get("companion_id") or "").strip()
        if not cid:
            continue
        label = clone.get("name") or cid
        voices.append(_voice(f"clone:{cid}", f"Cloned voice: {label}"))

    return Info(
        tts=[
            TtsProgram(
                name="xtts",
                description="XTTS-v2 (zero-shot voice cloning + streaming)",
                attribution=_attribution(),
                installed=True,
                version=None,
                voices=voices,
                supports_synthesize_streaming=False,
            )
        ]
    )


def _float_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


class XttsEventHandler(AsyncEventHandler):
    """Handles one Wyoming client connection."""

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            info = await _build_info()
            await self.write_event(info.event())
            return True

        if Synthesize.is_type(event.type):
            await self._synthesize(Synthesize.from_event(event))
            return True

        return True

    def _resolve_payload(self, synth: Synthesize) -> dict[str, Any]:
        voice_name: Optional[str] = None
        language = DEFAULT_LANGUAGE
        if synth.voice is not None:
            voice_name = synth.voice.name
            if synth.voice.language:
                language = synth.voice.language
        voice_name = (voice_name or DEFAULT_VOICE or DEFAULT_SPEAKER).strip()

        payload: dict[str, Any] = {"text": synth.text.strip(), "language": language}
        if voice_name.startswith("clone:"):
            payload["voice_clone_id"] = _clone_key(voice_name.split(":", 1)[1])
        else:
            payload["speaker"] = voice_name
        return payload

    async def _synthesize(self, synth: Synthesize) -> None:
        if not (synth.text or "").strip():
            return

        payload = self._resolve_payload(synth)
        _LOGGER.debug("synthesize: %s", {k: v for k, v in payload.items() if k != "text"})

        await self.write_event(
            AudioStart(rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS).event()
        )

        leftover = b""
        timestamp = 0
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                async with client.stream(
                    "POST", f"{XTTS_URL}/synthesise/stream", json=payload
                ) as resp:
                    resp.raise_for_status()
                    async for raw in resp.aiter_bytes():
                        if not raw:
                            continue
                        buf = leftover + raw
                        # float32 = 4 bytes; keep any trailing partial sample.
                        usable = len(buf) - (len(buf) % 4)
                        leftover = buf[usable:]
                        if usable == 0:
                            continue
                        f32 = np.frombuffer(buf[:usable], dtype="<f4")
                        chunk = AudioChunk(
                            rate=SAMPLE_RATE,
                            width=SAMPLE_WIDTH,
                            channels=CHANNELS,
                            audio=_float_to_pcm16(f32),
                            timestamp=timestamp,
                        )
                        await self.write_event(chunk.event())
                        timestamp += chunk.milliseconds
        except Exception as exc:  # noqa: BLE001 — always close the stream cleanly
            _LOGGER.error("synthesis failed: %s", exc)
        finally:
            await self.write_event(AudioStop(timestamp=timestamp).event())


async def main() -> None:
    parser = argparse.ArgumentParser(description="Wyoming TTS bridge for XTTS")
    parser.add_argument("--uri", default=os.getenv("WYOMING_URI", "tcp://0.0.0.0:10200"))
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    _LOGGER.info("Wyoming XTTS bridge listening on %s → %s", args.uri, XTTS_URL)

    server = AsyncServer.from_uri(args.uri)
    await server.run(partial(XttsEventHandler))


if __name__ == "__main__":
    asyncio.run(main())

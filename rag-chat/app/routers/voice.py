"""Voice proxy router — forwards STT and TTS requests to Dragon services.

Endpoints:
  POST /api/voice/transcribe        — proxy audio to Whisper STT
  POST /api/voice/synthesise        — proxy text to Qwen3-TTS
  POST /api/voice/synthesise/stream — SSE: sentence-chunked TTS
  GET  /api/voice/speakers          — list available TTS speakers
  GET  /api/voice/health            — fan-out health check
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

import httpx
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import TTS_URL, VOICE_TIMEOUT, WHISPER_URL, log

router = APIRouter(prefix="/api/voice", tags=["voice"])

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _require_whisper():
    if not WHISPER_URL:
        raise HTTPException(503, detail="WHISPER_URL not configured")


def _require_tts():
    if not TTS_URL:
        raise HTTPException(503, detail="TTS_URL not configured")


# ── Request / response models ────────────────────────────────────────────────

class SynthesiseRequest(BaseModel):
    text: str
    voice_clone_id: Optional[str] = None
    speaker: Optional[str] = None
    voice_description: Optional[str] = None
    instruction: Optional[str] = None
    emotion: str = "neutral"
    language: str = Field(default="en")


class SynthesiseStreamRequest(BaseModel):
    """Same as SynthesiseRequest but explicitly for streaming."""
    text: str
    speaker: Optional[str] = None
    voice_description: Optional[str] = None
    instruction: Optional[str] = None
    emotion: str = "neutral"
    language: str = Field(default="en")


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: Optional[str] = Query(default=None),
):
    """Proxy audio to Whisper STT on Dragon."""
    _require_whisper()

    audio_bytes = await audio.read()
    files = {"audio": (audio.filename or "audio.webm", audio_bytes, audio.content_type or "audio/webm")}
    params = {}
    if language:
        params["language"] = language

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
            r = await client.post(f"{WHISPER_URL}/transcribe", files=files, params=params)
            r.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(503, detail="Whisper STT unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    data = r.json()
    log.info("voice: transcribe elapsed=%dms lang=%s", elapsed_ms, data.get("language"))
    return data


@router.post("/synthesise")
async def synthesise(req: SynthesiseRequest):
    """Proxy text to TTS on Dragon."""
    _require_tts()

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
            r = await client.post(f"{TTS_URL}/synthesise", json=req.model_dump(exclude_none=True))
            r.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(503, detail="TTS server unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    data = r.json()
    log.info("voice: synthesise elapsed=%dms duration=%.1fs", elapsed_ms, data.get("duration", 0))
    return data


@router.post("/synthesise/stream")
async def synthesise_stream(req: SynthesiseStreamRequest):
    """SSE streaming TTS — split text into sentences and synthesise each chunk.

    Each SSE event contains a JSON object with the audio_url for one chunk,
    or an error/done sentinel.
    """
    _require_tts()

    sentences = [s.strip() for s in _SENTENCE_RE.split(req.text) if s.strip()]
    if not sentences:
        sentences = [req.text]

    base_payload = req.model_dump(exclude_none=True, exclude={"text"})

    async def event_stream():
        for i, sentence in enumerate(sentences):
            payload = {**base_payload, "text": sentence}
            try:
                async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
                    r = await client.post(f"{TTS_URL}/synthesise", json=payload)
                    r.raise_for_status()
                data = r.json()
                data["chunk_index"] = i
                data["text"] = sentence
                yield f"data: {json.dumps(data)}\n\n"
            except httpx.ConnectError:
                yield f"data: {json.dumps({'error': 'TTS server unreachable', 'chunk_index': i})}\n\n"
                return
            except httpx.HTTPStatusError as exc:
                yield f"data: {json.dumps({'error': exc.response.text, 'chunk_index': i})}\n\n"
                return
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc), 'chunk_index': i})}\n\n"
                return
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/speakers")
async def speakers():
    """List available TTS speakers."""
    _require_tts()

    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
            r = await client.get(f"{TTS_URL}/speakers")
            r.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(503, detail="TTS server unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)

    return r.json()


@router.get("/health")
async def voice_health():
    """Fan-out health check to both voice services."""
    results: dict = {"whisper": None, "tts": None}

    async with httpx.AsyncClient(timeout=5) as client:
        if WHISPER_URL:
            try:
                r = await client.get(f"{WHISPER_URL}/health")
                results["whisper"] = r.json() if r.status_code == 200 else {"status": "error", "code": r.status_code}
            except Exception as exc:
                results["whisper"] = {"status": "unreachable", "error": str(exc)}
        else:
            results["whisper"] = {"status": "not_configured"}

        if TTS_URL:
            try:
                r = await client.get(f"{TTS_URL}/health")
                results["tts"] = r.json() if r.status_code == 200 else {"status": "error", "code": r.status_code}
            except Exception as exc:
                results["tts"] = {"status": "unreachable", "error": str(exc)}
        else:
            results["tts"] = {"status": "not_configured"}

    all_ok = all(
        isinstance(v, dict) and v.get("status") == "ok"
        for v in results.values()
        if v and v.get("status") != "not_configured"
    )

    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ok" if all_ok else "degraded", **results},
    )

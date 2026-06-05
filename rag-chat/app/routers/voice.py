"""Voice proxy router — forwards STT and TTS requests to Dragon services.

Endpoints:
  POST /api/voice/transcribe        — proxy audio to Whisper STT
  POST /api/voice/synthesise        — proxy text to the TTS sidecar
  POST /api/voice/synthesise/stream — SSE: sentence-chunked TTS (one WAV/sentence)
  POST /api/voice/synthesise/pcm    — raw PCM passthrough for live playback (XTTS)
  POST /api/voice/analyze           — score candidate clips, recommend subset + order
  POST /api/voice/clone             — create a voice clone from one or more clips
  GET  /api/voice/clones            — list stored voice clones
  GET  /api/voice/clones/{id}/download — download a clone (prompt.pt + meta.json) as a zip
  POST /api/voice/clones/import     — import a downloaded voice archive (zip)
  GET  /api/voice/speakers          — list available TTS speakers
  GET  /api/voice/health            — fan-out health check
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
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
    voice_clone_id: Optional[str] = None
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


@router.post("/synthesise/pcm")
async def synthesise_pcm(req: SynthesiseRequest):
    """Low-latency passthrough of the TTS sidecar's raw PCM stream.

    Proxies `{TTS_URL}/synthesise/stream` byte-for-byte (24 kHz mono float32,
    `pcm_f32le`) so the browser can begin playback on the first chunk (~0.2 s)
    instead of waiting for whole-sentence WAVs. Forwards the upstream audio
    layout headers. Only the XTTS-v2 sidecar implements this; the legacy Qwen
    sidecar returns 404, which surfaces here as an upstream error.
    """
    _require_tts()

    payload = req.model_dump(exclude_none=True)
    client = httpx.AsyncClient(timeout=VOICE_TIMEOUT)
    try:
        upstream_req = client.build_request(
            "POST", f"{TTS_URL}/synthesise/stream", json=payload
        )
        upstream = await client.send(upstream_req, stream=True)
    except httpx.ConnectError:
        await client.aclose()
        raise HTTPException(503, detail="TTS server unreachable")
    except Exception:
        await client.aclose()
        raise

    if upstream.status_code != 200:
        body = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        detail = body.decode(errors="ignore") or "TTS stream error"
        raise HTTPException(upstream.status_code, detail=detail)

    headers = {
        "X-Sample-Rate": upstream.headers.get("x-sample-rate", "24000"),
        "X-Audio-Format": upstream.headers.get("x-audio-format", "pcm_f32le"),
        "X-Channels": upstream.headers.get("x-channels", "1"),
        "Cache-Control": "no-store",
    }

    t0 = time.monotonic()

    async def body_iter():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()
            log.info("voice: synthesise/pcm streamed elapsed=%dms", int((time.monotonic() - t0) * 1000))

    return StreamingResponse(body_iter(), media_type="application/octet-stream", headers=headers)


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


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    return slug or f"voice-{int(time.time())}"


@router.post("/clone")
async def clone_voice(
    audio: list[UploadFile] = File(...),
    voice_name: str = Form(...),
    companion_id: Optional[str] = Form(default=None),
    reference_text: str = Form(default=""),
):
    """Create a voice clone from one or more reference clips.

    Forwards every uploaded clip to the TTS sidecar's `/clone-voice`, which
    averages the speaker embedding across clips for a more robust clone than a
    single short take. `companion_id` defaults to a slug of `voice_name`.
    """
    _require_tts()

    if not audio:
        raise HTTPException(422, detail="At least one reference clip is required")

    cid = (companion_id or "").strip() or _slugify(voice_name)

    files = []
    for i, clip in enumerate(audio):
        data = await clip.read()
        files.append(
            ("audio", (clip.filename or f"clip-{i}.wav", data, clip.content_type or "audio/wav"))
        )
    form = {"companion_id": cid, "voice_name": voice_name, "reference_text": reference_text}

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
            r = await client.post(f"{TTS_URL}/clone-voice", files=files, data=form)
            r.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(503, detail="TTS server unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    data = r.json()
    log.info(
        "voice: clone elapsed=%dms id=%s clips=%d", elapsed_ms, cid, len(audio)
    )
    return data


@router.post("/analyze")
async def analyze_clips(audio: list[UploadFile] = File(...)):
    """Score candidate reference clips (no clone created).

    Forwards clips to the TTS sidecar's `/analyze-clips`, which returns a per-clip
    quality/consistency score, a recommended subset, and an optimal ordering.
    """
    _require_tts()

    if not audio:
        raise HTTPException(422, detail="No clips provided")

    files = []
    for i, clip in enumerate(audio):
        data = await clip.read()
        files.append(
            ("audio", (clip.filename or f"clip-{i}.wav", data, clip.content_type or "audio/wav"))
        )

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
            r = await client.post(f"{TTS_URL}/analyze-clips", files=files)
            r.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(503, detail="TTS server unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)

    log.info("voice: analyze elapsed=%dms clips=%d", int((time.monotonic() - t0) * 1000), len(audio))
    return r.json()


@router.get("/clones")
async def clones():
    """List stored voice clones from the TTS sidecar."""
    _require_tts()

    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
            r = await client.get(f"{TTS_URL}/clones")
            r.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(503, detail="TTS server unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)

    return r.json()


@router.post("/clones/import")
async def import_clone(
    archive: UploadFile = File(...),
    voice_name: str = Form(default=""),
    companion_id: Optional[str] = Form(default=None),
):
    """Import a previously downloaded voice archive (zip) into the TTS sidecar."""
    _require_tts()

    data = await archive.read()
    files = {"archive": (archive.filename or "voice.zip", data, archive.content_type or "application/zip")}
    form = {"voice_name": voice_name or "", "companion_id": companion_id or ""}

    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
            r = await client.post(f"{TTS_URL}/clones/import", files=files, data=form)
            r.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(503, detail="TTS server unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)

    return r.json()


@router.get("/clones/{companion_id}/download")
async def download_clone(companion_id: str):
    """Download a clone's latents + metadata sidecar as a zip."""
    _require_tts()

    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
            r = await client.get(f"{TTS_URL}/clones/{companion_id}/download")
            r.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(503, detail="TTS server unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)

    return Response(
        content=r.content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{companion_id}.zip"'},
    )


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

"""Voice proxy router — forwards STT and TTS requests to the voice sidecars.

TTS is a registry of interchangeable engines (e.g. local XTTS-v2, remote Miso)
that all speak the same sidecar contract. Every TTS endpoint accepts an
``engine`` selector (JSON field on synth requests, ``?engine=`` query param
elsewhere); omitting it uses the configured default engine.

Endpoints:
  GET  /api/voice/engines           — list configured TTS engines + the default
  POST /api/voice/transcribe        — proxy audio to Whisper STT
  POST /api/voice/synthesise        — proxy text to the selected TTS engine
  POST /api/voice/synthesise/stream — SSE: sentence-chunked TTS (one WAV/sentence)
  POST /api/voice/synthesise/pcm    — raw PCM passthrough for live playback
  POST /api/voice/analyze           — score candidate clips, recommend subset + order
  POST /api/voice/clone             — create a voice clone from one or more clips
  GET  /api/voice/clones            — list stored voice clones
  GET  /api/voice/clones/{id}/download — download a clone (prompt.pt + meta.json) as a zip
  POST /api/voice/clones/import     — import a downloaded voice archive (zip)
  GET  /api/voice/speakers          — list available TTS speakers
  GET  /api/voice/health            — fan-out health check (whisper + all engines)
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

from app.config import (
    TTS_DEFAULT_ENGINE,
    TTS_ENGINE_LABELS,
    TTS_ENGINES,
    VOICE_TIMEOUT,
    WHISPER_URL,
    log,
    tts_engine_url,
)

router = APIRouter(prefix="/api/voice", tags=["voice"])

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Fields that are routing hints for *this* proxy, not part of the sidecar
# contract — stripped before forwarding the JSON body upstream.
_PROXY_ONLY_FIELDS = {"engine"}


def _require_whisper():
    if not WHISPER_URL:
        raise HTTPException(503, detail="WHISPER_URL not configured")


def _require_tts():
    if not TTS_ENGINES:
        raise HTTPException(503, detail="No TTS engine configured")


def _tts_base(engine: Optional[str]) -> str:
    """Resolve the upstream base URL for a (possibly unknown) engine name."""
    base = tts_engine_url(engine)
    if not base:
        raise HTTPException(503, detail="No TTS engine configured")
    return base


# ── Request / response models ────────────────────────────────────────────────

class SynthesiseRequest(BaseModel):
    text: str
    engine: Optional[str] = None
    voice_clone_id: Optional[str] = None
    speaker: Optional[str] = None
    voice_description: Optional[str] = None
    instruction: Optional[str] = None
    emotion: str = "neutral"
    language: str = Field(default="en")


class SynthesiseStreamRequest(BaseModel):
    """Same as SynthesiseRequest but explicitly for streaming."""
    text: str
    engine: Optional[str] = None
    voice_clone_id: Optional[str] = None
    speaker: Optional[str] = None
    voice_description: Optional[str] = None
    instruction: Optional[str] = None
    emotion: str = "neutral"
    language: str = Field(default="en")


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/engines")
async def engines():
    """List the configured TTS engines and which one is the default.

    The UI uses this to populate the engine picker; every other voice endpoint
    accepts the returned ``id`` as an ``engine`` selector (query param or JSON
    field). Internal URLs are intentionally not exposed.
    """
    return {
        "engines": [
            {
                "id": name,
                "label": TTS_ENGINE_LABELS.get(name, name),
                "default": name == TTS_DEFAULT_ENGINE,
            }
            for name in TTS_ENGINES
        ],
        "default": TTS_DEFAULT_ENGINE,
    }


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
    """Proxy text to the selected TTS engine."""
    _require_tts()
    base = _tts_base(req.engine)
    payload = req.model_dump(exclude_none=True, exclude=_PROXY_ONLY_FIELDS)

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
            r = await client.post(f"{base}/synthesise", json=payload)
            r.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(503, detail="TTS server unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    data = r.json()
    log.info(
        "voice: synthesise engine=%s elapsed=%dms duration=%.1fs",
        req.engine or TTS_DEFAULT_ENGINE, elapsed_ms, data.get("duration", 0),
    )
    return data


@router.post("/synthesise/stream")
async def synthesise_stream(req: SynthesiseStreamRequest):
    """SSE streaming TTS — split text into sentences and synthesise each chunk.

    Each SSE event contains a JSON object with the audio_url for one chunk,
    or an error/done sentinel.
    """
    _require_tts()
    base = _tts_base(req.engine)

    sentences = [s.strip() for s in _SENTENCE_RE.split(req.text) if s.strip()]
    if not sentences:
        sentences = [req.text]

    base_payload = req.model_dump(exclude_none=True, exclude={"text", *_PROXY_ONLY_FIELDS})

    async def event_stream():
        for i, sentence in enumerate(sentences):
            payload = {**base_payload, "text": sentence}
            try:
                async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
                    r = await client.post(f"{base}/synthesise", json=payload)
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
    layout headers. The XTTS-v2 and Miso sidecars implement this; the legacy Qwen
    sidecar returns 404, which surfaces here as an upstream error.
    """
    _require_tts()
    base = _tts_base(req.engine)

    payload = req.model_dump(exclude_none=True, exclude=_PROXY_ONLY_FIELDS)
    client = httpx.AsyncClient(timeout=VOICE_TIMEOUT)
    try:
        upstream_req = client.build_request(
            "POST", f"{base}/synthesise/stream", json=payload
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
async def speakers(engine: Optional[str] = Query(default=None)):
    """List available TTS speakers for the selected engine."""
    _require_tts()
    base = _tts_base(engine)

    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
            r = await client.get(f"{base}/speakers")
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
    engine: Optional[str] = Query(default=None),
):
    """Create a voice clone from one or more reference clips on the chosen engine.

    Forwards every uploaded clip to the engine's `/clone-voice`. XTTS averages
    the speaker embedding across clips for a more robust clone than a single
    short take; Miso is one-shot (best single clip). `companion_id` defaults to
    a slug of `voice_name`.
    """
    _require_tts()
    base = _tts_base(engine)

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
            r = await client.post(f"{base}/clone-voice", files=files, data=form)
            r.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(503, detail="TTS server unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    data = r.json()
    log.info(
        "voice: clone engine=%s elapsed=%dms id=%s clips=%d",
        engine or TTS_DEFAULT_ENGINE, elapsed_ms, cid, len(audio),
    )
    return data


@router.post("/analyze")
async def analyze_clips(
    audio: list[UploadFile] = File(...),
    engine: Optional[str] = Query(default=None),
):
    """Score candidate reference clips on the chosen engine (no clone created).

    Forwards clips to the engine's `/analyze-clips`, which returns a per-clip
    quality/consistency score, a recommended subset, and an optimal ordering.
    """
    _require_tts()
    base = _tts_base(engine)

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
            r = await client.post(f"{base}/analyze-clips", files=files)
            r.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(503, detail="TTS server unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)

    log.info("voice: analyze elapsed=%dms clips=%d", int((time.monotonic() - t0) * 1000), len(audio))
    return r.json()


@router.get("/clones")
async def clones(engine: Optional[str] = Query(default=None)):
    """List stored voice clones from the selected TTS engine."""
    _require_tts()
    base = _tts_base(engine)

    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
            r = await client.get(f"{base}/clones")
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
    engine: Optional[str] = Query(default=None),
):
    """Import a previously downloaded voice archive (zip) into the chosen engine."""
    _require_tts()
    base = _tts_base(engine)

    data = await archive.read()
    files = {"archive": (archive.filename or "voice.zip", data, archive.content_type or "application/zip")}
    form = {"voice_name": voice_name or "", "companion_id": companion_id or ""}

    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
            r = await client.post(f"{base}/clones/import", files=files, data=form)
            r.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(503, detail="TTS server unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)

    return r.json()


@router.get("/clones/{companion_id}/download")
async def download_clone(companion_id: str, engine: Optional[str] = Query(default=None)):
    """Download a clone's latents + metadata sidecar as a zip."""
    _require_tts()
    base = _tts_base(engine)

    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
            r = await client.get(f"{base}/clones/{companion_id}/download")
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
    """Fan-out health check to Whisper and every configured TTS engine.

    ``engines`` carries each engine's health individually; ``tts`` mirrors the
    default engine for back-compat with older UI code. Overall status is OK when
    Whisper (if configured) and the default engine are both healthy — a
    non-default engine being down only degrades that engine, not the page.
    """
    results: dict = {"whisper": None, "tts": None, "engines": {}, "default_engine": TTS_DEFAULT_ENGINE}

    async with httpx.AsyncClient(timeout=5) as client:
        if WHISPER_URL:
            try:
                r = await client.get(f"{WHISPER_URL}/health")
                results["whisper"] = r.json() if r.status_code == 200 else {"status": "error", "code": r.status_code}
            except Exception as exc:
                results["whisper"] = {"status": "unreachable", "error": str(exc)}
        else:
            results["whisper"] = {"status": "not_configured"}

        for name, url in TTS_ENGINES.items():
            try:
                r = await client.get(f"{url}/health")
                results["engines"][name] = r.json() if r.status_code == 200 else {"status": "error", "code": r.status_code}
            except Exception as exc:
                results["engines"][name] = {"status": "unreachable", "error": str(exc)}

    if not TTS_ENGINES:
        results["tts"] = {"status": "not_configured"}
    else:
        results["tts"] = results["engines"].get(TTS_DEFAULT_ENGINE, {"status": "not_configured"})

    def _ok(entry) -> bool:
        return isinstance(entry, dict) and entry.get("status") in ("ok", "not_configured")

    all_ok = _ok(results["whisper"]) and _ok(results["tts"])

    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ok" if all_ok else "degraded", **results},
    )

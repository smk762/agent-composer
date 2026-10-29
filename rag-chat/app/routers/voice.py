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
  DELETE /api/voice/clones/{id}     — permanently delete a stored voice clone
  GET  /api/voice/clones/{id}/download — download a clone (prompt.pt + meta.json) as a zip
  POST /api/voice/clones/import     — import a downloaded voice archive (zip)
  GET  /api/voice/speakers          — list available TTS speakers
  GET  /api/voice/health            — fan-out health check (whisper + all engines)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from app.config import (
    MAX_SENTENCE_PAUSE_MS,
    TTS_DEFAULT_ENGINE,
    TTS_ENGINE_LABELS,
    TTS_ENGINES,
    TTS_FALLBACK_ENGINE,
    VOICE_S3_ACCESS_KEY_ID,
    VOICE_S3_BUCKET,
    VOICE_S3_ENDPOINT_URL,
    VOICE_S3_PUBLIC_URL,
    VOICE_S3_REHOST,
    VOICE_S3_SECRET_ACCESS_KEY,
    VOICE_TIMEOUT,
    WHISPER_URL,
    default_sentence_pause_ms,
    log,
    tts_engine_chain,
    tts_engine_url,
)

# Connection-level failures that mean "this engine/host is down" — safe to retry
# on the next engine in the fallback chain. HTTP error *responses* (4xx/5xx) are
# not retried: the engine is up and deliberately rejected the request.
_UNREACHABLE = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout)

router = APIRouter(prefix="/api/voice", tags=["voice"])

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Fields that are routing hints for *this* proxy, not part of the sidecar
# contract — stripped before forwarding the JSON body upstream. ``pause_ms`` is
# handled here (sentence pacing), never sent to the engine.
_PROXY_ONLY_FIELDS = {"engine", "pause_ms"}


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


# ── Rendered-audio re-hosting ────────────────────────────────────────────────
# A remote engine (e.g. Miso on another subnet) returns an audio_url on its own
# host, which a LAN browser may not reach. When the shared voice bucket is
# configured, re-host such audio into MinIO so playback always comes from the
# same S3_PUBLIC_URL origin the XTTS sidecar uses.

_voice_s3_client = None


def _voice_s3():
    global _voice_s3_client
    if _voice_s3_client is None:
        import boto3

        _voice_s3_client = boto3.client(
            "s3",
            endpoint_url=VOICE_S3_ENDPOINT_URL,
            aws_access_key_id=VOICE_S3_ACCESS_KEY_ID,
            aws_secret_access_key=VOICE_S3_SECRET_ACCESS_KEY,
        )
    return _voice_s3_client


async def _rehost_audio_url(audio_url: Optional[str]) -> Optional[str]:
    """Return a browser-reachable URL for a rendered clip.

    If re-hosting is disabled, the URL is empty, or it's already served from the
    shared voice bucket (S3_PUBLIC_URL), the URL is returned untouched. Otherwise
    the clip is downloaded and re-uploaded to the voice bucket under the same
    ``generated/voice/`` prefix the XTTS sidecar uses, and the S3_PUBLIC_URL is
    returned. Best-effort: any failure falls back to the original URL so
    synthesis never breaks on a re-host hiccup.
    """
    if not VOICE_S3_REHOST or not audio_url:
        return audio_url
    if audio_url.startswith(VOICE_S3_PUBLIC_URL):
        return audio_url
    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
            r = await client.get(audio_url)
            r.raise_for_status()
            data = r.content
            content_type = r.headers.get("content-type", "audio/wav")
        key = f"generated/voice/{uuid.uuid4().hex}.wav"
        await asyncio.to_thread(
            lambda: _voice_s3().put_object(
                Bucket=VOICE_S3_BUCKET, Key=key, Body=data, ContentType=content_type
            )
        )
        new_url = f"{VOICE_S3_PUBLIC_URL}/{key}"
        log.info("voice: rehosted audio (%d bytes) %s -> %s", len(data), audio_url, new_url)
        return new_url
    except Exception as exc:
        log.warning("voice: audio re-host failed (%s); serving original url %s", exc, audio_url)
        return audio_url


# ── Sentence pacing (inter-sentence breath) ──────────────────────────────────
# CSM-style engines (Miso) run multi-sentence text as one fast utterance with no
# gap between sentences, and ignore emotion/speed. To add a "breath", the proxy
# can synthesise each sentence separately and stitch the clips with a silence
# gap, then re-host the combined WAV in the shared voice bucket.

def _concat_wavs_with_gaps(wav_blobs: list[bytes], gap_ms: int) -> tuple[bytes, float]:
    """Concatenate same-format PCM WAV blobs with ``gap_ms`` silence between them.

    Returns ``(combined_wav_bytes, total_seconds)``. All blobs must share sample
    rate / width / channels (they do — same engine renders every sentence).
    """
    import io
    import wave

    frames: list[bytes] = []
    params = None
    for blob in wav_blobs:
        with wave.open(io.BytesIO(blob), "rb") as w:
            if params is None:
                params = w.getparams()
            frames.append(w.readframes(w.getnframes()))

    if params is None:
        raise ValueError("no audio to concatenate")

    gap_frames = max(0, int(params.framerate * gap_ms / 1000))
    silence = b"\x00" * (gap_frames * params.sampwidth * params.nchannels)

    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(params.nchannels)
        w.setsampwidth(params.sampwidth)
        w.setframerate(params.framerate)
        for i, fr in enumerate(frames):
            if i > 0 and silence:
                w.writeframes(silence)
            w.writeframes(fr)

    frame_bytes = params.sampwidth * params.nchannels
    total_frames = sum(len(f) // frame_bytes for f in frames) + gap_frames * max(0, len(frames) - 1)
    return out.getvalue(), total_frames / params.framerate


async def _synthesise_paced(chain, base_payload: dict, sentences: list[str], gap_ms: int) -> dict:
    """Render each sentence on a single resolved engine, stitch with silence.

    Resolves a reachable engine on the first sentence (walking the fallback
    chain) and stays on it, so the whole utterance keeps one voice. Uploads the
    combined WAV to the voice bucket and returns the standard synth response.
    """
    resolved: Optional[tuple] = None
    wav_blobs: list[bytes] = []
    last_exc: Optional[Exception] = None

    async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
        for sentence in sentences:
            payload = {**base_payload, "text": sentence}
            candidates = [resolved] if resolved else list(chain)
            blob = None
            for name, base in candidates:
                try:
                    r = await client.post(f"{base}/synthesise", json=payload)
                    r.raise_for_status()
                except _UNREACHABLE as exc:
                    last_exc = exc
                    log.warning("voice: paced engine=%s unreachable, trying next: %s", name, exc)
                    continue
                except httpx.HTTPStatusError as exc:
                    raise HTTPException(exc.response.status_code, detail=exc.response.text)
                data = r.json()
                url = data.get("audio_url")
                if not url:
                    raise HTTPException(502, detail="engine returned no audio_url to pace")
                ar = await client.get(url)
                ar.raise_for_status()
                resolved = (name, base)
                blob = ar.content
                break
            if blob is None:
                raise HTTPException(503, detail=f"No reachable TTS engine ({last_exc})")
            wav_blobs.append(blob)

    combined, total_seconds = await asyncio.to_thread(_concat_wavs_with_gaps, wav_blobs, gap_ms)
    key = f"generated/voice/{uuid.uuid4().hex}.wav"
    await asyncio.to_thread(
        lambda: _voice_s3().put_object(
            Bucket=VOICE_S3_BUCKET, Key=key, Body=combined, ContentType="audio/wav"
        )
    )
    return {
        "audio_url": f"{VOICE_S3_PUBLIC_URL}/{key}",
        "duration": round(total_seconds, 3),
        "duration_seconds": round(total_seconds, 3),
        "format": "wav",
        "engine": resolved[0] if resolved else None,
        "paced": True,
        "pause_ms": gap_ms,
        "sentences": len(sentences),
    }


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
    # Playback rate hint forwarded to the engine (reliable on XTTS; Miso accepts
    # it but doesn't honour it reliably). None → engine default.
    speed: Optional[float] = None
    # Proxy-side sentence pacing: silence (ms) inserted between sentences so
    # multi-sentence text gets a breath. None → per-engine default (Miso on).
    # 0 disables. Requires the shared voice bucket to store the stitched result.
    pause_ms: Optional[int] = None


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
    speed: Optional[float] = None


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
                "fallback": name == TTS_FALLBACK_ENGINE,
            }
            for name in TTS_ENGINES
        ],
        "default": TTS_DEFAULT_ENGINE,
        "fallback": TTS_FALLBACK_ENGINE or None,
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
    """Proxy text to the selected TTS engine (falling back if it's unreachable)."""
    _require_tts()
    chain = tts_engine_chain(req.engine)
    if not chain:
        raise HTTPException(503, detail="No TTS engine configured")
    payload = req.model_dump(exclude_none=True, exclude=_PROXY_ONLY_FIELDS)

    t0 = time.monotonic()

    # Sentence pacing: when a per-sentence breath is wanted (explicit pause_ms or
    # the engine's default, e.g. Miso), render sentences separately and stitch
    # them with silence. Needs the voice bucket to host the combined clip and
    # only helps when there's more than one sentence.
    sentences = [s.strip() for s in _SENTENCE_RE.split(req.text) if s.strip()]
    gap_ms = req.pause_ms if req.pause_ms is not None else default_sentence_pause_ms(chain[0][0])
    if gap_ms and gap_ms > 0 and len(sentences) > 1 and VOICE_S3_REHOST:
        gap_ms = min(int(gap_ms), MAX_SENTENCE_PAUSE_MS)
        base_payload = {k: v for k, v in payload.items() if k != "text"}
        try:
            data = await _synthesise_paced(chain, base_payload, sentences, gap_ms)
        except HTTPException:
            raise
        except Exception as exc:
            # Pacing is cosmetic. A non-PCM WAV, a failed audio fetch, or a bucket
            # that is down must not turn a working synth into a 500, so fall back
            # to a single-shot render (as re-hosting already does).
            log.warning("voice: paced synthesis failed (%s); using single-shot render", exc)
        else:
            log.info(
                "voice: synthesise paced engine=%s sentences=%d gap=%dms elapsed=%dms duration=%.1fs",
                data.get("engine"), len(sentences), gap_ms,
                int((time.monotonic() - t0) * 1000), data.get("duration", 0),
            )
            return data

    last_exc: Optional[Exception] = None
    for name, base in chain:
        try:
            async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
                r = await client.post(f"{base}/synthesise", json=payload)
                r.raise_for_status()
        except _UNREACHABLE as exc:
            last_exc = exc
            log.warning("voice: synthesise engine=%s unreachable, trying next: %s", name, exc)
            continue
        except httpx.HTTPStatusError as exc:
            raise HTTPException(exc.response.status_code, detail=exc.response.text)

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        data = r.json()
        if data.get("audio_url"):
            data["audio_url"] = await _rehost_audio_url(data["audio_url"])
        log.info(
            "voice: synthesise engine=%s elapsed=%dms duration=%.1fs",
            name, elapsed_ms, data.get("duration", 0),
        )
        return data

    raise HTTPException(503, detail=f"No reachable TTS engine ({last_exc})")


@router.post("/synthesise/stream")
async def synthesise_stream(req: SynthesiseStreamRequest):
    """SSE streaming TTS — split text into sentences and synthesise each chunk.

    Each SSE event contains a JSON object with the audio_url for one chunk,
    or an error/done sentinel.
    """
    _require_tts()
    chain = tts_engine_chain(req.engine)
    if not chain:
        raise HTTPException(503, detail="No TTS engine configured")

    sentences = [s.strip() for s in _SENTENCE_RE.split(req.text) if s.strip()]
    if not sentences:
        sentences = [req.text]

    base_payload = req.model_dump(exclude_none=True, exclude={"text", *_PROXY_ONLY_FIELDS})

    async def event_stream():
        # Resolve a reachable engine on the first chunk (walking the fallback
        # chain), then stay on it for the remaining sentences so a stream never
        # mixes voices mid-utterance.
        resolved: Optional[tuple] = None
        for i, sentence in enumerate(sentences):
            payload = {**base_payload, "text": sentence}
            candidates = [resolved] if resolved else list(chain)
            sent = False
            for name, base in candidates:
                try:
                    async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
                        r = await client.post(f"{base}/synthesise", json=payload)
                        r.raise_for_status()
                    resolved = (name, base)
                    data = r.json()
                    if data.get("audio_url"):
                        data["audio_url"] = await _rehost_audio_url(data["audio_url"])
                    data["chunk_index"] = i
                    data["text"] = sentence
                    data["engine"] = name
                    yield f"data: {json.dumps(data)}\n\n"
                    sent = True
                    break
                except _UNREACHABLE as exc:
                    log.warning("voice: stream engine=%s unreachable, trying next: %s", name, exc)
                    continue
                except httpx.HTTPStatusError as exc:
                    yield f"data: {json.dumps({'error': exc.response.text, 'chunk_index': i})}\n\n"
                    return
                except Exception as exc:
                    yield f"data: {json.dumps({'error': str(exc), 'chunk_index': i})}\n\n"
                    return
            if not sent:
                yield f"data: {json.dumps({'error': 'No reachable TTS engine', 'chunk_index': i})}\n\n"
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
    chain = tts_engine_chain(req.engine)
    if not chain:
        raise HTTPException(503, detail="No TTS engine configured")

    payload = req.model_dump(exclude_none=True, exclude=_PROXY_ONLY_FIELDS)

    # Open the upstream stream, walking the fallback chain on connection errors.
    client = httpx.AsyncClient(timeout=VOICE_TIMEOUT)
    upstream = None
    last_exc: Optional[Exception] = None
    for name, base in chain:
        try:
            upstream_req = client.build_request("POST", f"{base}/synthesise/stream", json=payload)
            upstream = await client.send(upstream_req, stream=True)
            break
        except _UNREACHABLE as exc:
            last_exc = exc
            log.warning("voice: pcm engine=%s unreachable, trying next: %s", name, exc)
            continue
        except Exception:
            await client.aclose()
            raise
    if upstream is None:
        await client.aclose()
        raise HTTPException(503, detail=f"No reachable TTS engine ({last_exc})")

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


@router.delete("/clones/{companion_id}")
async def delete_clone(companion_id: str, engine: Optional[str] = Query(default=None)):
    """Permanently delete a stored voice clone from the selected TTS engine.

    Engine-specific (like the other clone ops): the request goes only to the
    named engine, never the fallback chain. An engine whose sidecar has no
    delete route surfaces its upstream 404/405 to the caller.
    """
    _require_tts()
    base = _tts_base(engine)

    try:
        async with httpx.AsyncClient(timeout=VOICE_TIMEOUT) as client:
            r = await client.delete(f"{base}/clones/{companion_id}")
            r.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(503, detail="TTS server unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)

    log.info("voice: delete engine=%s id=%s", engine or TTS_DEFAULT_ENGINE, companion_id)
    return r.json()


@router.get("/health")
async def voice_health():
    """Fan-out health check to Whisper and every configured TTS engine.

    ``engines`` carries each engine's health individually; ``tts`` mirrors the
    default engine for back-compat with older UI code. Overall status is OK when
    Whisper (if configured) is healthy *and* at least one engine in the default
    fallback chain (default → fallback) is reachable. This mirrors the actual
    synth path (``tts_engine_chain``): a request only fails when every engine in
    the chain is down, so a reachable fallback keeps the page healthy even while
    the preferred engine's host is offline. ``serving_engine`` reports which
    chain engine would currently handle a default-routed request.
    """
    results: dict = {
        "whisper": None,
        "tts": None,
        "engines": {},
        "default_engine": TTS_DEFAULT_ENGINE,
        "serving_engine": None,
    }

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

    def _ok(entry) -> bool:
        return isinstance(entry, dict) and entry.get("status") in ("ok", "not_configured")

    if not TTS_ENGINES:
        results["tts"] = {"status": "not_configured"}
        tts_ok = True
    else:
        results["tts"] = results["engines"].get(TTS_DEFAULT_ENGINE, {"status": "not_configured"})
        # Healthy if any engine in the default routing chain is reachable —
        # a live fallback keeps synthesis working, so the page isn't degraded.
        tts_ok = False
        for name, _base in tts_engine_chain(None):
            if _ok(results["engines"].get(name)):
                tts_ok = True
                results["serving_engine"] = name
                break

    all_ok = _ok(results["whisper"]) and tts_ok

    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ok" if all_ok else "degraded", **results},
    )

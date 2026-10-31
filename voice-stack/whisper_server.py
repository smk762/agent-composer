"""
faster-whisper STT server — FastAPI, port 8030.

GPU-accelerated speech-to-text using CTranslate2 via faster-whisper.

GPU memory lifecycle:
  Model is loaded on first request (lazy) and evicted after
  STT_KEEP_ALIVE_GPU seconds of inactivity, freeing VRAM for other
  services.  Set to -1 to keep resident forever (original behaviour).

Endpoints:
  POST /transcribe      — multipart file upload (playact-engine, rag-chat)
  POST /transcriptions   — JSON body with audio_url or audio_base64 (kimini-api)
  GET  /health           — readiness check

Env vars:
  STT_MODEL           — model id (default "large-v3-turbo")
  STT_DEVICE          — "cuda" (default) or "cpu"
  STT_COMPUTE_TYPE    — "float16" (default), "int8", "int8_float16", "float32"
  STT_VAD_FILTER      — "1" (default) to enable Silero VAD pre-filter
  STT_BEAM_SIZE       — beam size for decoding (default 5)
  STT_KEEP_ALIVE_GPU  — seconds on GPU before eviction. -1 = forever. Default 120.
"""

import asyncio
import base64
import gc
import os
import tempfile
import time
from typing import Optional

import httpx
import structlog
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

STT_MODEL = os.getenv("STT_MODEL", "large-v3-turbo")
STT_DEVICE = os.getenv("STT_DEVICE", "cuda")
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "float16")
STT_VAD_FILTER = os.getenv("STT_VAD_FILTER", "1") == "1"
STT_BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", "5"))
STT_DOWNLOAD_TIMEOUT = int(os.getenv("STT_DOWNLOAD_TIMEOUT", "30"))
STT_KEEP_ALIVE_GPU: float = float(os.getenv("STT_KEEP_ALIVE_GPU", "120"))

_model = None
_last_used: float = 0.0
_lock = asyncio.Lock()
_evict_task: Optional[asyncio.Task] = None


def _load_model():
    from faster_whisper import WhisperModel

    logger.info(
        "loading_whisper",
        model=STT_MODEL,
        device=STT_DEVICE,
        compute_type=STT_COMPUTE_TYPE,
    )
    model = WhisperModel(
        STT_MODEL,
        device=STT_DEVICE,
        compute_type=STT_COMPUTE_TYPE,
    )
    logger.info("whisper_loaded")
    return model


def _unload_model():
    global _model
    if _model is not None:
        del _model
        _model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("whisper_unloaded")


async def _ensure_model():
    global _model, _last_used, _evict_task
    async with _lock:
        if _model is None:
            _model = await asyncio.to_thread(_load_model)
        _last_used = time.time()
        if _evict_task and not _evict_task.done():
            _evict_task.cancel()
        if STT_KEEP_ALIVE_GPU >= 0:
            _evict_task = asyncio.create_task(_eviction_loop())
    return _model


async def _eviction_loop():
    try:
        if STT_KEEP_ALIVE_GPU == 0:
            async with _lock:
                _unload_model()
            return
        await asyncio.sleep(STT_KEEP_ALIVE_GPU)
        async with _lock:
            if time.time() - _last_used >= STT_KEEP_ALIVE_GPU:
                _unload_model()
    except asyncio.CancelledError:
        pass


def _transcribe_sync(path: str, language: str | None, model) -> dict:
    """Run transcription in a thread — faster-whisper is not async.

    Returns extended result with all fields needed by both endpoints.
    """

    kwargs: dict = {
        "beam_size": STT_BEAM_SIZE,
        "vad_filter": STT_VAD_FILTER,
    }
    if language:
        kwargs["language"] = language

    segments, info = model.transcribe(path, **kwargs)
    text = " ".join(seg.text.strip() for seg in segments)

    return {
        "text": text,
        "language": info.language or "en",
        "duration": round(info.duration, 2),
        "duration_seconds": round(info.duration, 3),
        "confidence": round(info.language_probability, 4),
    }


def _gpu_memory_info() -> dict | None:
    """Return GPU memory stats if CUDA is available."""
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            total = torch.cuda.get_device_properties(0).total_mem / 1024**2
            return {
                "allocated_mb": round(allocated, 1),
                "reserved_mb": round(reserved, 1),
                "total_mb": round(total, 1),
            }
    except Exception:
        pass
    return None


async def _run_transcription(tmp_path: str, language: str | None, endpoint: str) -> dict:
    """Shared transcription runner with GPU error handling."""
    model = await _ensure_model()
    t0 = time.monotonic()
    try:
        result = await asyncio.to_thread(_transcribe_sync, tmp_path, language, model)
    except RuntimeError as exc:
        err_str = str(exc).lower()
        if "out of memory" in err_str or "cuda" in err_str:
            gpu_info = _gpu_memory_info()
            logger.error("stt_gpu_oom", error=str(exc), gpu=gpu_info, endpoint=endpoint)
            return JSONResponse(
                status_code=503,
                content={"error": "GPU OOM", "detail": str(exc), "gpu": gpu_info},
            )
        raise
    except Exception as exc:
        logger.error("stt_transcription_failed", error=str(exc), exc_info=True, endpoint=endpoint)
        return JSONResponse(
            status_code=500,
            content={"error": f"Transcription failed: {exc}"},
        )

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "stt_transcribed",
        endpoint=endpoint,
        language=result["language"],
        duration=result["duration"],
        elapsed_ms=elapsed_ms,
        text_len=len(result["text"]),
    )
    return result


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Voice Stack — Whisper STT", version="1.1.0")


# ── POST /transcribe — multipart (playact-engine, rag-chat) ──────────────────

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Query(default=None, description="ISO 639-1 language hint"),
):
    audio_bytes = await audio.read()
    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        result = await _run_transcription(tmp_path, language, "/transcribe")
    finally:
        os.unlink(tmp_path)

    if isinstance(result, JSONResponse):
        return result

    return {"text": result["text"], "language": result["language"], "duration": result["duration"]}


# ── POST /transcriptions — JSON body (kimini-api) ────────────────────────────

class TranscriptionRequest(BaseModel):
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    language: Optional[str] = None


@app.post("/transcriptions")
async def transcriptions(req: TranscriptionRequest):
    """JSON endpoint for kimini-api and similar consumers.

    Accepts audio_url (downloaded server-side) or audio_base64 (decoded
    in-process). Returns duration_seconds and confidence alongside the
    standard text and language fields.
    """
    if req.audio_url and req.audio_base64:
        raise HTTPException(422, detail="Provide audio_url or audio_base64, not both")
    if not req.audio_url and not req.audio_base64:
        raise HTTPException(400, detail="audio_url or audio_base64 is required")

    suffix = ".wav"

    if req.audio_url:
        try:
            async with httpx.AsyncClient(timeout=STT_DOWNLOAD_TIMEOUT) as client:
                resp = await client.get(req.audio_url)
                resp.raise_for_status()
                audio_bytes = resp.content
        except httpx.ConnectError as exc:
            raise HTTPException(502, detail=f"Cannot reach audio URL: {exc}")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(502, detail=f"Audio download failed: HTTP {exc.response.status_code}")
        except httpx.TimeoutException:
            raise HTTPException(502, detail=f"Audio download timed out ({STT_DOWNLOAD_TIMEOUT}s)")

        # Infer suffix from URL path or content-type
        from urllib.parse import urlparse
        url_path = urlparse(req.audio_url).path
        ext = os.path.splitext(url_path)[1]
        if ext:
            suffix = ext
    else:
        raw = req.audio_base64
        if "," in raw:
            raw = raw.split(",", 1)[1]
        try:
            audio_bytes = base64.b64decode(raw)
        except Exception as exc:
            raise HTTPException(400, detail=f"Invalid base64: {exc}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        result = await _run_transcription(tmp_path, req.language, "/transcriptions")
    finally:
        os.unlink(tmp_path)

    if isinstance(result, JSONResponse):
        return result

    return {
        "text": result["text"],
        "language": result["language"],
        "duration_seconds": result["duration_seconds"],
        "confidence": result["confidence"],
    }


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    loaded = _model is not None
    return {
        "status": "ok",
        "model": STT_MODEL,
        "device": STT_DEVICE,
        "compute_type": STT_COMPUTE_TYPE,
        "model_loaded": loaded,
        "model_state": "gpu" if loaded else "unloaded",
        "keep_alive_gpu": STT_KEEP_ALIVE_GPU,
    }


@app.post("/admin/evict")
async def admin_evict():
    """Force-unload model, freeing VRAM immediately."""
    async with _lock:
        _unload_model()
    return {"status": "evicted", "model_state": "unloaded"}


@app.post("/admin/wake")
async def admin_wake():
    """Pre-load model onto GPU."""
    await _ensure_model()
    return {"status": "ready", "model_state": "gpu"}

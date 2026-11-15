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
  GET  /health           — readiness check; 503 when the audio decoder is broken

Env vars:
  STT_MODEL           — model id (default "large-v3-turbo")
  STT_DEVICE          — "cuda" (default) or "cpu"
  STT_COMPUTE_TYPE    — "float16" (default), "int8", "int8_float16", "float32"
  STT_VAD_FILTER      — "1" (default) to enable Silero VAD pre-filter
  STT_BEAM_SIZE       — beam size for decoding (default 5)
  STT_KEEP_ALIVE_GPU  — seconds on GPU before eviction. -1 = forever. Default 120.
  STT_DOWNLOAD_TIMEOUT — seconds allowed for an audio_url download (default 30).
  STT_DECODE_TIMEOUT  — seconds allowed for the ffmpeg decode of one upload (default 120).
  STT_AUDIO_URL_ALLOWED_HOSTS — comma-separated `host` or `host:port` entries
                        that /transcriptions may fetch even though they resolve
                        to a private/loopback address (e.g. `minio:9000` or the
                        stateful node's MinIO). Default empty: only http(s) URLs
                        whose host resolves to public addresses are fetched.

audio_url SSRF policy (/transcriptions):
  Only http/https URLs are fetched. Unless the host is in
  STT_AUDIO_URL_ALLOWED_HOSTS, every address the hostname resolves to must be
  public (no loopback, private, link-local/metadata, CGNAT, reserved, multicast
  or unspecified ranges, IPv4 or IPv6, incl. IPv4-mapped IPv6), and the request
  is pinned to the vetted IP (Host header + TLS SNI keep the original name), so
  a DNS rebind between the check and the fetch can't redirect it. Redirects are
  not followed, and every failure returns the same generic 400/502 message.
"""

import asyncio
import base64
import gc
import hmac
import ipaddress
import os
import socket
import subprocess
import sys
import tempfile
import time
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import httpx
import numpy as np
import structlog
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
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


def _parse_allowed_hosts(raw: str) -> frozenset[str]:
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


STT_AUDIO_URL_ALLOWED_HOSTS = _parse_allowed_hosts(os.getenv("STT_AUDIO_URL_ALLOWED_HOSTS", ""))

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


SAMPLE_RATE = 16000  # what faster-whisper expects for an ndarray input
STT_DECODE_TIMEOUT = float(os.getenv("STT_DECODE_TIMEOUT", "120"))
HEALTH_DECODE_TTL_S = 60.0


class AudioDecodeError(Exception):
    """ffmpeg could not decode the audio."""


def _decode_audio(path: str) -> np.ndarray:
    """Decode any ffmpeg-readable file to 16 kHz mono float32.

    Done with the ffmpeg CLI instead of faster-whisper's PyAV-based
    ``decode_audio``, so a PyAV API change can't break every upload (#42)."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "0",
        "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=STT_DECODE_TIMEOUT, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioDecodeError(f"ffmpeg failed: {exc}") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()[-300:]
        raise AudioDecodeError(f"ffmpeg exit {proc.returncode}: {err}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _silence_wav(path: str, seconds: float = 1.0, rate: int = 24000) -> None:
    import wave

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(rate * seconds))


def _decoder_self_test() -> dict:
    """Decode a second of generated silence through the real decode path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        _silence_wav(path)
        samples = _decode_audio(path)
        if abs(len(samples) - SAMPLE_RATE) > SAMPLE_RATE // 100:
            return {"ok": False, "error": f"decoded {len(samples)} samples, expected ~{SAMPLE_RATE}"}
        return {"ok": True}
    except Exception as exc:  # any failure here means /transcribe is broken too
        return {"ok": False, "error": str(exc)}
    finally:
        os.unlink(path)


_decoder_check: dict = {"at": 0.0, "result": None}


async def _decoder_status() -> dict:
    now = time.monotonic()
    if _decoder_check["result"] is None or now - _decoder_check["at"] >= HEALTH_DECODE_TTL_S:
        _decoder_check["result"] = await asyncio.to_thread(_decoder_self_test)
        _decoder_check["at"] = now
    return _decoder_check["result"]


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

    audio = _decode_audio(path)
    segments, info = model.transcribe(audio, **kwargs)
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
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
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
        logger.error("stt_transcription_failed", error=str(exc), exc_info=True, endpoint=endpoint)
        return JSONResponse(
            status_code=500,
            content={"error": f"Transcription failed: {exc}"},
        )
    except AudioDecodeError as exc:
        logger.warning("stt_decode_failed", error=str(exc), endpoint=endpoint)
        return JSONResponse(
            status_code=400,
            content={"error": f"Could not decode audio: {exc}"},
        )
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


# ── audio_url fetch (SSRF-guarded) ────────────────────────────────────────────

_AUDIO_URL_REJECTED = "audio_url is not an allowed http(s) URL"
_AUDIO_URL_FAILED = "Could not download audio_url"


class AudioURLError(Exception):
    """audio_url rejected or not fetchable; `status` is the HTTP code to return."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _ip_is_public(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def _host_allowlisted(host: str, port: int, allowed: frozenset[str]) -> bool:
    host = host.lower()
    if ":" in host:  # IPv6 literal: entries use the bracketed form
        host = f"[{host}]"
    return host in allowed or f"{host}:{port}" in allowed


async def _resolve_host(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


async def _vet_audio_url(url: str, allowed: Optional[frozenset[str]] = None) -> tuple[list[str], dict, dict]:
    """Validate `url` and return (fetch_urls, headers, extensions) for httpx.

    For a non-allowlisted host there is one fetch URL per vetted address, in
    resolver order, each targeting that IP directly (tried in turn, like a
    normal connect falls back across a dual-stack host's addresses); the Host
    header and TLS SNI (``sni_hostname``) keep the original name so virtual
    hosting and certificate checks still work."""
    allowed = STT_AUDIO_URL_ALLOWED_HOSTS if allowed is None else allowed
    try:
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        host = parts.hostname
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError:
        raise AudioURLError(400, _AUDIO_URL_REJECTED)
    if scheme not in ("http", "https") or not host or parts.username or parts.password:
        raise AudioURLError(400, _AUDIO_URL_REJECTED)

    if _host_allowlisted(host, port, allowed):
        return [url], {}, {}

    try:
        addrs = await _resolve_host(host, port)
    except (OSError, UnicodeError):
        raise AudioURLError(400, _AUDIO_URL_REJECTED)
    if not addrs or not all(_ip_is_public(a) for a in addrs):
        logger.warning("stt_audio_url_rejected", host=host, reason="non_public_address")
        raise AudioURLError(400, _AUDIO_URL_REJECTED)

    pinned: list[str] = []
    for addr in addrs:
        ip = addr.split("%", 1)[0]
        ip_host = f"[{ip}]" if ":" in ip else ip
        netloc = f"{ip_host}:{parts.port}" if parts.port else ip_host
        target = urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))
        if target not in pinned:
            pinned.append(target)
    host_header = parts.netloc.rsplit("@", 1)[-1]
    extensions = {"sni_hostname": host} if scheme == "https" else {}
    return pinned, {"Host": host_header}, extensions


async def _fetch_audio_url(url: str) -> bytes:
    fetch_urls, headers, extensions = await _vet_audio_url(url)
    resp = None
    try:
        async with httpx.AsyncClient(
            timeout=STT_DOWNLOAD_TIMEOUT, follow_redirects=False, trust_env=False
        ) as client:
            for i, fetch_url in enumerate(fetch_urls):
                try:
                    resp = await client.get(fetch_url, headers=headers, extensions=extensions)
                    break
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    # This address is unreachable (e.g. IPv6 in an IPv4-only network): try the next.
                    if i == len(fetch_urls) - 1:
                        raise
    except httpx.HTTPError as exc:
        logger.warning("stt_audio_url_fetch_failed", error_type=type(exc).__name__)
        raise AudioURLError(502, _AUDIO_URL_FAILED)
    if resp is None or resp.status_code != 200:
        logger.warning("stt_audio_url_fetch_failed", status=getattr(resp, "status_code", None))
        raise AudioURLError(502, _AUDIO_URL_FAILED)
    return resp.content


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
            audio_bytes = await _fetch_audio_url(req.audio_url)
        except AudioURLError as exc:
            raise HTTPException(exc.status, detail=exc.detail)

        # Infer suffix from URL path or content-type
        url_path = urlsplit(req.audio_url).path
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
    """Readiness without loading the model: the decoder is exercised on a
    second of silence (cached HEALTH_DECODE_TTL_S) so a broken decode path
    reports 503 instead of ``ok`` (#42). A full transcription would pull the
    model onto the GPU on every poll."""
    loaded = _model is not None
    decoder = await _decoder_status()
    body = {
        "status": "ok" if decoder["ok"] else "error",
        "decoder": decoder,
        "model": STT_MODEL,
        "device": STT_DEVICE,
        "compute_type": STT_COMPUTE_TYPE,
        "model_loaded": loaded,
        "model_state": "gpu" if loaded else "unloaded",
        "keep_alive_gpu": STT_KEEP_ALIVE_GPU,
        "gpu": _resident_gpu(),
    }
    if not decoder["ok"]:
        return JSONResponse(status_code=503, content=body)
    return body


# ── Resident contract (argus-iris argus-http adapter) ─────────────────────────
# /health carries a ``gpu`` block, POST /admin/unload frees the model, and
# /admin/wake preloads it. Set ARGUS_ADMIN_TOKEN to require
# ``Authorization: Bearer <token>`` on the admin routes (open when unset).


def _require_admin(authorization: Optional[str] = Header(default=None)) -> None:
    token = os.environ.get("ARGUS_ADMIN_TOKEN", "")
    if token and not hmac.compare_digest((authorization or "").encode(), f"Bearer {token}".encode()):
        raise HTTPException(401, detail="admin token required")


def _free_vram_mb() -> Optional[int]:
    """Device-wide free VRAM, only when this process already holds a CUDA
    context: creating one just to answer /health would cost VRAM itself."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            return int(torch.cuda.mem_get_info()[0] / 1024**2)
    except Exception:
        pass
    return None


def _resident_gpu() -> dict:
    loaded = _model is not None
    return {
        "loaded": loaded,
        "model_state": "gpu" if loaded else "unloaded",
        "idle_s": round(time.time() - _last_used, 1) if _last_used else None,
        "free_vram_mb": _free_vram_mb(),
    }


async def _unload_now() -> dict:
    global _evict_task
    async with _lock:
        was_loaded = _model is not None
        _unload_model()
        if _evict_task and not _evict_task.done():
            _evict_task.cancel()
    return {"unloaded": was_loaded, "model_state": "unloaded", "gpu": _resident_gpu()}


@app.post("/admin/evict", dependencies=[Depends(_require_admin)])
async def admin_evict():
    """Force-unload model, freeing VRAM immediately (no CPU stage here)."""
    return {"status": "evicted", **await _unload_now()}


@app.post("/admin/unload", dependencies=[Depends(_require_admin)])
async def admin_unload():
    """Fully unload the model; it reloads lazily on the next request."""
    return {"status": "unloaded", **await _unload_now()}


@app.post("/admin/wake", dependencies=[Depends(_require_admin)])
async def admin_wake():
    """Pre-load model onto GPU."""
    await _ensure_model()
    return {"status": "ready", "model_state": "gpu", "gpu": _resident_gpu()}

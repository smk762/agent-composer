"""
XTTS-v2 server — FastAPI, port 8031 (drop-in for the Qwen3-TTS sidecar).

Coqui XTTS-v2 (via the maintained `coqui-tts` fork) for zero-shot voice cloning
with streaming output. Exposes the same HTTP contract as the Qwen3-TTS server so
rag-chat's voice proxy, the `voice` MinIO bucket, and the UI need no changes:

  POST /synthesise         — text → audio, uploaded to S3, returns {audio_url,...}
  POST /synthesise/stream  — text → chunked 24 kHz PCM (real within-sentence
                             streaming via XTTS inference_stream)
  POST /analyze-clips      — score candidate clips, recommend subset + ordering
  POST /clone-voice        — one or more reference clips → conditioning latents in S3
  GET  /clones             — list stored voice clones (from meta.json sidecars)
  GET  /clones/{id}/download — download a clone's prompt.pt + meta.json as a zip
  POST /clones/import      — import a downloaded voice archive (zip) back into S3
  GET  /speakers           — built-in XTTS studio speakers
  GET  /health             — status + GPU memory
  POST /admin/{evict,unload,wake}

GPU memory lifecycle (three-state, like the Qwen sidecar / ModernBERT):
  unloaded → cpu → gpu

  On request the model is promoted to GPU; after XTTS_KEEP_ALIVE_GPU idle
  seconds it demotes to CPU, after a further XTTS_KEEP_ALIVE_CPU it fully
  unloads. XTTS is a single nn.Module so device moves are a plain .to().

A "voice clone" here is the pair (gpt_cond_latent, speaker_embedding) computed
from the reference clip by get_conditioning_latents(); we torch.save it to S3
under the same `voice_clones/{companion_id}/prompt.pt` key the Qwen path used.

Env vars:
  XTTS_MODEL           — Coqui model id (default tts_models/multilingual/multi-dataset/xtts_v2)
  XTTS_DEVICE          — "cuda:0" (default) or "cpu"
  XTTS_PRELOAD         — "1" to load to GPU at startup, "0" (default) lazy
  XTTS_USE_DEEPSPEED   — "1" to enable DeepSpeed inference (default "0")
  XTTS_KEEP_ALIVE_GPU  — idle seconds on GPU before demoting to CPU (default 120)
  XTTS_KEEP_ALIVE_CPU  — idle seconds on CPU before full unload (default 600)
  XTTS_TEMPERATURE     — sampling temperature (default 0.7)
  XTTS_STREAM_CHUNK_SIZE — GPT tokens per streamed audio chunk (default 20)
  S3_ENDPOINT_URL / S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY /
  S3_BUCKET_NAME / S3_PUBLIC_URL
"""

import asyncio
import gc
import io
import json
import math
import os
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

import structlog
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

XTTS_MODEL = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
DEVICE = os.getenv("XTTS_DEVICE", "cuda:0")
PRELOAD = os.getenv("XTTS_PRELOAD", "0") == "1"
USE_DEEPSPEED = os.getenv("XTTS_USE_DEEPSPEED", "0") == "1"
KEEP_ALIVE_GPU: float = float(os.getenv("XTTS_KEEP_ALIVE_GPU", "120"))
KEEP_ALIVE_CPU: float = float(os.getenv("XTTS_KEEP_ALIVE_CPU", "600"))
TEMPERATURE: float = float(os.getenv("XTTS_TEMPERATURE", "0.7"))
STREAM_CHUNK_SIZE: int = int(os.getenv("XTTS_STREAM_CHUNK_SIZE", "20"))

# Reference-clip preprocessing (applied at clone time, before latents).
PREPROCESS: bool = os.getenv("XTTS_PREPROCESS", "1") == "1"
PREPROCESS_VAD: bool = os.getenv("XTTS_PREPROCESS_VAD", "1") == "1"
PREPROCESS_SR: int = int(os.getenv("XTTS_PREPROCESS_SR", "24000"))
PREPROCESS_TARGET_LUFS: float = float(os.getenv("XTTS_PREPROCESS_LUFS", "-23.0"))
PREPROCESS_HIGHPASS_HZ: float = float(os.getenv("XTTS_PREPROCESS_HIGHPASS_HZ", "85"))
# Cascade N/2 biquads → ~12 dB/oct per pass; 4 ⇒ 24 dB/oct to knock down HVAC rumble.
PREPROCESS_HIGHPASS_ORDER: int = int(os.getenv("XTTS_PREPROCESS_HIGHPASS_ORDER", "4"))
PREPROCESS_MAX_SECONDS: float = float(os.getenv("XTTS_PREPROCESS_MAX_SECONDS", "30"))
PREPROCESS_VAD_PAD: float = float(os.getenv("XTTS_PREPROCESS_VAD_PAD", "0.1"))
# Stationary denoise (e.g. HVAC hum) seeded from VAD gaps; gated on estimated SNR.
PREPROCESS_DENOISE: bool = os.getenv("XTTS_PREPROCESS_DENOISE", "1") == "1"
PREPROCESS_DENOISE_PROP: float = float(os.getenv("XTTS_PREPROCESS_DENOISE_PROP", "0.75"))
PREPROCESS_DENOISE_SNR_DB: float = float(os.getenv("XTTS_PREPROCESS_DENOISE_SNR_DB", "30"))

# Conditioning-latent extraction. The speaker embedding is averaged over ALL
# clips, but the GPT conditioning latent only draws on the first `gpt_cond_len`
# seconds of the concatenation — XTTS's own default is 6, which under-uses a
# multi-clip reference set, so we default to 30 (matching XTTS high-level
# inference) and order clips best-first so that window is the strongest audio.
GPT_COND_LEN: int = int(os.getenv("XTTS_GPT_COND_LEN", "30"))
GPT_COND_CHUNK_LEN: int = int(os.getenv("XTTS_GPT_COND_CHUNK_LEN", "4"))
MAX_REF_LEN: int = int(os.getenv("XTTS_MAX_REF_LEN", "30"))
# Score (0–100) below which a clip is excluded from the default selection.
ANALYZE_SCORE_THRESHOLD: float = float(os.getenv("XTTS_ANALYZE_THRESHOLD", "60"))

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "minioadmin")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "minioadmin")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "voice")
S3_PUBLIC_URL = os.getenv("S3_PUBLIC_URL", "http://localhost:9000/voice")

SAMPLE_RATE = 24000  # XTTS-v2 output sample rate

# XTTS-v2 supported languages → our public ISO codes accepted on the wire.
# Qwen used "zh"; XTTS expects "zh-cn", so we normalise.
ISO_TO_XTTS: dict[str, str] = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it",
    "pt": "pt", "pl": "pl", "tr": "tr", "ru": "ru", "nl": "nl",
    "cs": "cs", "ar": "ar", "ja": "ja", "hu": "hu", "ko": "ko",
    "hi": "hi", "zh": "zh-cn", "zh-cn": "zh-cn",
    # tolerate full names from older callers
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "russian": "ru", "japanese": "ja",
    "korean": "ko", "chinese": "zh-cn",
}

_CLONE_CACHE_MAX = 50

# ── GPU memory lifecycle ─────────────────────────────────────────────────────


class TTSState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    CPU = "cpu"
    GPU = "gpu"
    ERROR = "error"


class XTTSModelManager:
    """Three-state lifecycle: unloaded → cpu → gpu (mirrors the Qwen sidecar)."""

    def __init__(self) -> None:
        self.keep_alive_gpu = KEEP_ALIVE_GPU
        self.keep_alive_cpu = KEEP_ALIVE_CPU
        self._state = TTSState.UNLOADED
        self._error: Optional[str] = None
        self._model = None
        self._config = None
        self._last_used: float = 0.0
        self._lock = asyncio.Lock()
        self._evict_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> TTSState:
        return self._state

    @property
    def model(self):
        return self._model

    @property
    def is_ready(self) -> bool:
        return self._state in (TTSState.CPU, TTSState.GPU)

    def _load_to_cpu(self) -> None:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.utils.manage import ModelManager

        logger.info("loading_xtts", model=XTTS_MODEL, target="cpu")

        # Download (no-op if cached) and resolve the local checkpoint dir + config.
        model_path, config_path, _ = ModelManager().download_model(XTTS_MODEL)
        if config_path is None:
            config_path = os.path.join(str(model_path), "config.json")

        config = XttsConfig()
        config.load_json(str(config_path))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=str(model_path),
            use_deepspeed=USE_DEEPSPEED,
        )
        model.eval()
        self._config = config
        self._model = model
        logger.info("xtts_loaded_cpu")

    def _move_to_device(self, device: str) -> None:
        import torch

        if self._model is None:
            return
        if device == "cpu":
            self._model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            self._model.to(torch.device(device))

    async def ensure_gpu(self) -> None:
        """Guarantee the model is on GPU and reset the idle timer."""
        async with self._lock:
            if self._state == TTSState.GPU:
                self._last_used = time.time()
                self._arm_evict()
                return

            if self._state in (TTSState.UNLOADED, TTSState.ERROR):
                self._state = TTSState.LOADING
                try:
                    await asyncio.to_thread(self._load_to_cpu)
                    self._state = TTSState.CPU
                    self._error = None
                except Exception as exc:
                    self._state = TTSState.ERROR
                    self._error = str(exc)
                    logger.error("xtts_load_failed", error=str(exc))
                    raise

            if self._state == TTSState.CPU:
                if DEVICE.startswith("cuda"):
                    logger.info("xtts_promoting_to_gpu", device=DEVICE)
                    await asyncio.to_thread(self._move_to_device, DEVICE)
                    self._state = TTSState.GPU
                    logger.info("xtts_on_gpu")
                else:
                    self._state = TTSState.GPU  # CPU-only: treat as ready

            self._last_used = time.time()
            self._arm_evict()

    async def demote_to_cpu(self) -> None:
        async with self._lock:
            if self._state != TTSState.GPU:
                return
            if DEVICE == "cpu":
                return
            logger.info("xtts_demoting_to_cpu")
            await asyncio.to_thread(self._move_to_device, "cpu")
            self._state = TTSState.CPU
            logger.info("xtts_on_cpu")

    async def unload(self) -> None:
        async with self._lock:
            if self._evict_task and not self._evict_task.done():
                self._evict_task.cancel()
            if self._model is not None:
                del self._model
                self._model = None
            self._config = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._state = TTSState.UNLOADED
            logger.info("xtts_fully_unloaded")

    def _arm_evict(self) -> None:
        if self._evict_task and not self._evict_task.done():
            self._evict_task.cancel()
        self._evict_task = asyncio.create_task(self._eviction_loop())

    async def _eviction_loop(self) -> None:
        try:
            if self.keep_alive_gpu == 0:
                await self.demote_to_cpu()
            elif self.keep_alive_gpu > 0:
                await asyncio.sleep(self.keep_alive_gpu)
                if time.time() - self._last_used >= self.keep_alive_gpu:
                    await self.demote_to_cpu()

            if self.keep_alive_cpu == 0:
                await self.unload()
            elif self.keep_alive_cpu > 0:
                await asyncio.sleep(self.keep_alive_cpu)
                if time.time() - self._last_used >= self.keep_alive_gpu + self.keep_alive_cpu:
                    await self.unload()
        except asyncio.CancelledError:
            pass


# ── Global state ─────────────────────────────────────────────────────────────

_mgr = XTTSModelManager()
_s3_client = None
_clone_cache: OrderedDict = OrderedDict()
_clone_cache_lock = threading.Lock()
_vad_model = None
_vad_lock = threading.Lock()


def _get_vad_model():
    """Lazy-load the bundled Silero VAD model (no network/torch.hub fetch)."""
    global _vad_model
    with _vad_lock:
        if _vad_model is None:
            from silero_vad import load_silero_vad

            _vad_model = load_silero_vad()
        return _vad_model


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        import boto3

        _s3_client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=S3_ACCESS_KEY_ID,
            aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        )
    return _s3_client


def _upload_to_s3(data: bytes, key: str, content_type: str = "audio/wav") -> str:
    client = _get_s3_client()
    client.put_object(Bucket=S3_BUCKET_NAME, Key=key, Body=data, ContentType=content_type)
    return f"{S3_PUBLIC_URL}/{key}"


def _device_tensor(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def _get_clone_latents(clone_id: str):
    """Load (gpt_cond_latent, speaker_embedding) for a stored clone, cached."""
    with _clone_cache_lock:
        if clone_id in _clone_cache:
            _clone_cache.move_to_end(clone_id)
            return _clone_cache[clone_id]

    import torch
    from botocore.exceptions import ClientError

    client = _get_s3_client()
    buf = io.BytesIO()
    try:
        client.download_fileobj(S3_BUCKET_NAME, clone_id, buf)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        logger.error("clone_latent_s3_error", clone_id=clone_id, error_code=error_code)
        raise ValueError(f"Voice clone not found: {clone_id}") from exc
    buf.seek(0)
    prompt = torch.load(buf, map_location="cpu", weights_only=False)

    with _clone_cache_lock:
        _clone_cache[clone_id] = prompt
        if len(_clone_cache) > _CLONE_CACHE_MAX:
            _clone_cache.popitem(last=False)

    return prompt


def _resolve_latents(req) -> tuple:
    """Return (gpt_cond_latent, speaker_embedding) on the model's device for a
    request, from a stored clone, a built-in speaker, or a default fallback."""
    model = _mgr.model
    device = next(model.parameters()).device

    if getattr(req, "voice_clone_id", None):
        prompt = _get_clone_latents(req.voice_clone_id)
        gpt = _device_tensor(prompt["gpt_cond_latent"], device)
        spk = _device_tensor(prompt["speaker_embedding"], device)
        return gpt, spk

    # Built-in studio speaker (or the first one as a sane default).
    speaker_name = getattr(req, "speaker", None)
    sm = getattr(model, "speaker_manager", None)
    if sm is not None and getattr(sm, "speakers", None):
        if not speaker_name:
            speaker_name = sorted(sm.speakers.keys())[0]
        if speaker_name not in sm.speakers:
            raise ValueError(f"Unknown speaker: {speaker_name!r}")
        entry = sm.speakers[speaker_name]
        gpt = _device_tensor(entry["gpt_cond_latent"], device)
        spk = _device_tensor(entry["speaker_embedding"], device)
        return gpt, spk

    raise ValueError("No voice_clone_id and no built-in speakers available")


def _audio_duration(path: str) -> float | None:
    """Best-effort clip duration in seconds (soundfile → torchaudio fallback).

    Returns None if neither backend can read the container, so the caller can
    defer to XTTS's own loader rather than rejecting a usable upload.
    """
    try:
        import soundfile as sf

        return float(sf.info(path).duration)
    except Exception:
        pass
    try:
        import torchaudio

        info = torchaudio.info(path)
        if info.sample_rate:
            return info.num_frames / info.sample_rate
    except Exception:
        pass
    return None


def _normalise_language(raw: str) -> str:
    lang = ISO_TO_XTTS.get((raw or "en").lower())
    if lang is None:
        raise ValueError(
            f"Unsupported language: {raw!r}. Supported: {sorted(set(ISO_TO_XTTS.values()))}"
        )
    return lang


def _gpu_memory_info() -> dict | None:
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


def _list_speakers() -> list[str]:
    model = _mgr.model
    sm = getattr(model, "speaker_manager", None) if model is not None else None
    if sm is not None and getattr(sm, "speakers", None):
        return sorted(sm.speakers.keys())
    return []


# ── App lifecycle ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    if PRELOAD:
        await _mgr.ensure_gpu()
    yield
    await _mgr.unload()


app = FastAPI(title="Voice Stack — XTTS-v2", version="1.0.0", lifespan=lifespan)


# ── Request models ────────────────────────────────────────────────────────────


class SynthesiseRequest(BaseModel):
    text: str
    voice_clone_id: Optional[str] = None
    speaker: Optional[str] = None
    voice_description: Optional[str] = None  # unused by XTTS; accepted for compat
    instruction: Optional[str] = None        # unused by XTTS; accepted for compat
    emotion: str = "neutral"                  # unused by XTTS; accepted for compat
    language: str = Field(default="en")
    speed: Optional[float] = None


# ── Synthesis helpers ──────────────────────────────────────────────────────────


def _run_inference(text: str, language: str, gpt, spk, speed: float):
    out = _mgr.model.inference(
        text,
        language,
        gpt,
        spk,
        temperature=TEMPERATURE,
        speed=speed,
        enable_text_splitting=True,
    )
    return out["wav"]


# ── POST /synthesise ────────────────────────────────────────────────────────


@app.post("/synthesise")
async def synthesise(req: SynthesiseRequest):
    """Generate speech, upload WAV to S3, return the public URL."""
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Model still loading"})

    t_req = time.monotonic()
    await _mgr.ensure_gpu()
    t_ready = time.monotonic()

    try:
        language = _normalise_language(req.language)
        gpt, spk = await asyncio.to_thread(_resolve_latents, req)
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})

    job_id = str(uuid.uuid4())
    speed = req.speed if req.speed and req.speed > 0 else 1.0

    t_gen0 = time.monotonic()
    try:
        wav = await asyncio.to_thread(_run_inference, req.text, language, gpt, spk, speed)
    except RuntimeError as exc:
        err_str = str(exc).lower()
        if "out of memory" in err_str or "cuda" in err_str:
            gpu_info = _gpu_memory_info()
            logger.error("xtts_gpu_oom", error=str(exc), gpu=gpu_info)
            return JSONResponse(status_code=503, content={"error": "GPU OOM", "detail": str(exc), "gpu": gpu_info})
        raise
    except Exception as exc:
        logger.error("xtts_generation_failed", error=str(exc), exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"TTS generation failed: {exc}"})
    t_gen1 = time.monotonic()

    import numpy as np

    audio_data = np.asarray(wav, dtype=np.float32)
    duration = len(audio_data) / SAMPLE_RATE

    import soundfile as sf

    buf = io.BytesIO()
    await asyncio.to_thread(sf.write, buf, audio_data, SAMPLE_RATE, format="WAV")
    wav_bytes = buf.getvalue()
    t_encoded = time.monotonic()

    s3_key = f"generated/voice/{job_id}.wav"
    audio_url = await asyncio.to_thread(_upload_to_s3, wav_bytes, s3_key)
    t_uploaded = time.monotonic()

    logger.info(
        "xtts_synthesised",
        job_id=job_id,
        duration=round(duration, 2),
        language=language,
        ready_s=round(t_ready - t_req, 2),
        generate_s=round(t_gen1 - t_gen0, 2),
        encode_s=round(t_encoded - t_gen1, 2),
        upload_s=round(t_uploaded - t_encoded, 2),
        total_s=round(t_uploaded - t_req, 2),
        rtf=round((t_gen1 - t_gen0) / duration, 2) if duration else None,
    )

    return {
        "audio_url": audio_url,
        "duration": duration,
        "duration_seconds": round(duration, 3),
        "format": "wav",
    }


# ── POST /synthesise/stream ───────────────────────────────────────────────────


@app.post("/synthesise/stream")
async def synthesise_stream(req: SynthesiseRequest):
    """Stream raw 24 kHz mono float32 PCM as it is generated (XTTS
    inference_stream). The client concatenates chunks; the trailing
    `X-Sample-Rate`/`X-Audio-Format` headers describe the byte layout."""
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Model still loading"})

    await _mgr.ensure_gpu()

    try:
        language = _normalise_language(req.language)
        gpt, spk = await asyncio.to_thread(_resolve_latents, req)
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})

    speed = req.speed if req.speed and req.speed > 0 else 1.0
    loop = asyncio.get_running_loop()

    def _produce(queue: asyncio.Queue):
        import torch

        try:
            stream = _mgr.model.inference_stream(
                req.text,
                language,
                gpt,
                spk,
                temperature=TEMPERATURE,
                speed=speed,
                stream_chunk_size=STREAM_CHUNK_SIZE,
                enable_text_splitting=True,
            )
            for chunk in stream:
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.detach().cpu().to(torch.float32).numpy()
                loop.call_soon_threadsafe(queue.put_nowait, chunk.tobytes())
        except Exception as exc:  # surface as stream termination
            logger.error("xtts_stream_failed", error=str(exc), exc_info=True)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        worker = asyncio.create_task(asyncio.to_thread(_produce, queue))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            await worker

    return StreamingResponse(
        event_stream(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Format": "pcm_f32le",
            "X-Channels": "1",
        },
    )


# ── Reference-clip preprocessing ──────────────────────────────────────────────


def _preprocess_clip(in_path: str) -> tuple[str, dict]:
    """Standardise a reference clip for cloning and report what changed.

    Chain: mono → resample → DC removal → cascaded high-pass (HVAC rumble) →
    Silero VAD split into speech + gap (noise) regions → SNR estimate → gated
    stationary denoise seeded from the gaps (HVAC hum under speech) → length cap
    → LUFS loudness normalisation (peak guarded). Writes a 16-bit WAV temp file.
    Each stage degrades gracefully and is recorded in the returned report so the
    UI can show per-clip results.
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    import torchaudio.functional as AF

    report: dict = {"applied": [], "warnings": []}

    wav, sr = torchaudio.load(in_path)  # (channels, samples) float32
    report["original_seconds"] = round(wav.shape[-1] / sr, 2)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        report["applied"].append("mono")

    if sr != PREPROCESS_SR:
        wav = AF.resample(wav, sr, PREPROCESS_SR)
        sr = PREPROCESS_SR

    wav = wav - wav.mean()  # DC offset removal

    if PREPROCESS_HIGHPASS_HZ > 0:
        passes = max(1, round(PREPROCESS_HIGHPASS_ORDER / 2))
        for _ in range(passes):
            wav = AF.highpass_biquad(wav, sr, PREPROCESS_HIGHPASS_HZ)
        report["applied"].append(f"highpass{int(PREPROCESS_HIGHPASS_HZ)}Hz/{passes * 12}dBoct")

    samples = wav.squeeze(0).contiguous()
    total_len = samples.shape[-1]

    # Split into speech vs non-speech (gap) regions. Speech feeds the clone; the
    # gaps are a clean, stationary noise profile — ideal for estimating HVAC hum.
    speech = samples
    noise = None
    if PREPROCESS_VAD:
        try:
            from silero_vad import get_speech_timestamps

            model = _get_vad_model()
            vad_wav = samples if sr == 16000 else AF.resample(samples.unsqueeze(0), sr, 16000).squeeze(0)
            ts = get_speech_timestamps(vad_wav, model, sampling_rate=16000, return_seconds=True)
            if ts:
                pad = PREPROCESS_VAD_PAD
                inset = int(0.05 * sr)  # shrink gaps so onsets/breaths don't pollute the noise profile
                min_gap = int(0.05 * sr)
                speech_segs, noise_segs = [], []
                prev_end = 0
                for t in ts:
                    a = max(0, int((t["start"] - pad) * sr))
                    b = min(total_len, int((t["end"] + pad) * sr))
                    if b > a:
                        speech_segs.append(samples[a:b])
                    g0, g1 = prev_end + inset, int(t["start"] * sr) - inset
                    if g1 - g0 > min_gap:
                        noise_segs.append(samples[g0:g1])
                    prev_end = int(t["end"] * sr)
                tail = prev_end + inset
                if total_len - tail > min_gap:
                    noise_segs.append(samples[tail:total_len])

                if speech_segs:
                    speech = torch.cat(speech_segs)
                    report["applied"].append("vad_trim")
                if noise_segs:
                    noise = torch.cat(noise_segs)
            else:
                report["warnings"].append("vad_no_speech")
        except Exception as exc:
            report["warnings"].append(f"vad_failed:{exc}")

    # SNR / noise-floor estimate (post high-pass, so it reflects residual hum).
    eps = 1e-9
    snr_db = None
    if noise is not None and noise.numel() > int(0.2 * sr):
        speech_rms = float(torch.sqrt(torch.mean(speech ** 2)).item())
        noise_rms = float(torch.sqrt(torch.mean(noise ** 2)).item())
        report["noise_floor_dbfs"] = round(20.0 * math.log10(noise_rms + eps), 1)
        if speech_rms > 0:
            snr_db = 20.0 * math.log10((speech_rms + eps) / (noise_rms + eps))
            report["snr_db"] = round(snr_db, 1)

    # Targeted stationary denoise (HVAC hum). Gated on SNR so clean clips skip it.
    if PREPROCESS_DENOISE and (snr_db is None or snr_db < PREPROCESS_DENOISE_SNR_DB):
        try:
            import noisereduce as nr

            sp = speech.numpy().astype(np.float32)
            if noise is not None and noise.numel() > int(0.2 * sr):
                den = nr.reduce_noise(
                    y=sp, sr=sr, y_noise=noise.numpy().astype(np.float32),
                    stationary=True, prop_decrease=PREPROCESS_DENOISE_PROP,
                )
                report["applied"].append(f"denoise(seeded,{PREPROCESS_DENOISE_PROP})")
            else:
                # No clean gap to seed from: self-estimate, but be more conservative.
                den = nr.reduce_noise(
                    y=sp, sr=sr, stationary=True,
                    prop_decrease=min(PREPROCESS_DENOISE_PROP, 0.6),
                )
                report["applied"].append("denoise(self,0.6)")
            speech = torch.from_numpy(np.ascontiguousarray(den, dtype=np.float32))
        except Exception as exc:
            report["warnings"].append(f"denoise_failed:{exc}")

    samples = speech

    max_samps = int(PREPROCESS_MAX_SECONDS * sr)
    if samples.shape[-1] > max_samps:
        samples = samples[:max_samps]
        report["applied"].append(f"capped{int(PREPROCESS_MAX_SECONDS)}s")

    report["kept_seconds"] = round(samples.shape[-1] / sr, 2)
    audio_np = samples.numpy().astype(np.float32)

    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loud_in = meter.integrated_loudness(audio_np)
        if np.isfinite(loud_in):
            report["lufs_in"] = round(float(loud_in), 1)
            normed = pyln.normalize.loudness(audio_np, loud_in, PREPROCESS_TARGET_LUFS)
            peak = float(np.max(np.abs(normed))) if normed.size else 0.0
            if peak > 0.99:
                normed = normed * (0.99 / peak)
            audio_np = normed.astype(np.float32)
            report["lufs_out"] = PREPROCESS_TARGET_LUFS
            report["gain_db"] = round(float(PREPROCESS_TARGET_LUFS - loud_in), 1)
            report["applied"].append("lufs_norm")
        else:
            report["warnings"].append("lufs_unmeasurable")
    except Exception as exc:
        report["warnings"].append(f"lufs_failed:{exc}")
        peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
        if peak > 0:
            audio_np = (audio_np * (0.97 / peak)).astype(np.float32)
            report["applied"].append("peak_norm")

    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    sf.write(out.name, audio_np, sr, subtype="PCM_16")
    report["sample_rate"] = sr
    return out.name, report


def _conditioning_latents(paths: list[str]):
    """Compute (gpt_cond_latent, speaker_embedding) with our tuned windows."""
    return _mgr.model.get_conditioning_latents(
        paths,
        gpt_cond_len=GPT_COND_LEN,
        gpt_cond_chunk_len=GPT_COND_CHUNK_LEN,
        max_ref_length=MAX_REF_LEN,
    )


# ── Clip scoring / ranking ────────────────────────────────────────────────────


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _quick_clip_metrics(path: str) -> dict:
    """Lightweight per-clip metrics for scoring — no denoise/LUFS/file writes.

    Measures clipping/peak on the raw audio and uses Silero VAD (at 16 kHz) to
    derive usable speech seconds + an SNR/noise-floor estimate from the gaps.
    SNR is intentionally measured on the *raw* signal (pre-denoise) so the score
    reflects how noisy the source actually is.
    """
    import torch
    import torchaudio
    import torchaudio.functional as AF

    m: dict = {"flags": []}
    wav, sr = torchaudio.load(path)
    mag = wav.abs()
    m["raw_seconds"] = round(wav.shape[-1] / sr, 2)
    m["peak"] = round(float(mag.max().item()) if mag.numel() else 0.0, 3)
    m["clip_ratio"] = round(float((mag > 0.99).float().mean().item()) if mag.numel() else 0.0, 4)

    mono = wav.mean(dim=0, keepdim=True) if wav.shape[0] > 1 else wav
    s16 = mono.squeeze(0) if sr == 16000 else AF.resample(mono, sr, 16000).squeeze(0)

    m["speech_seconds"] = m["raw_seconds"]
    m["snr_db"] = None
    m["noise_floor_dbfs"] = None
    try:
        from silero_vad import get_speech_timestamps

        ts = get_speech_timestamps(s16, _get_vad_model(), sampling_rate=16000, return_seconds=True)
        if not ts:
            m["flags"].append("vad_no_speech")
            return m

        total = s16.shape[-1]
        inset = int(0.05 * 16000)
        speech_segs, noise_segs = [], []
        prev_end = 0
        for t in ts:
            a, b = max(0, int(t["start"] * 16000)), min(total, int(t["end"] * 16000))
            if b > a:
                speech_segs.append(s16[a:b])
            g0, g1 = prev_end + inset, int(t["start"] * 16000) - inset
            if g1 - g0 > inset:
                noise_segs.append(s16[g0:g1])
            prev_end = int(t["end"] * 16000)
        tail = prev_end + inset
        if total - tail > inset:
            noise_segs.append(s16[tail:total])

        if speech_segs:
            speech = torch.cat(speech_segs)
            m["speech_seconds"] = round(speech.shape[-1] / 16000, 2)
            if noise_segs:
                noise = torch.cat(noise_segs)
                if noise.numel() > int(0.2 * 16000):
                    nrms = float(torch.sqrt(torch.mean(noise ** 2)).item())
                    srms = float(torch.sqrt(torch.mean(speech ** 2)).item())
                    m["noise_floor_dbfs"] = round(20.0 * math.log10(nrms + 1e-9), 1)
                    if srms > 0:
                        m["snr_db"] = round(20.0 * math.log10((srms + 1e-9) / (nrms + 1e-9)), 1)
    except Exception as exc:
        m["flags"].append(f"vad_failed:{exc}")
    return m


def _analyze_clips(raw_paths: list[str], names: list[str]) -> dict:
    """Score each candidate clip for cloning suitability and recommend a subset
    + an optimal ordering (best anchor first, for the GPT conditioning window).

    Per clip we measure lightweight quality metrics (clipping, SNR, usable
    speech seconds) and a speaker embedding. Clips are then scored on quality and
    on *consistency* — cosine similarity to the embedding centroid — so
    recordings from a different room/mic/voice surface as low-scoring outliers.
    The heavy denoise/LUFS pipeline is skipped here; it runs at clone time.
    """
    import torch

    model = _mgr.model
    items: list[dict] = []
    embeddings: list[Optional["torch.Tensor"]] = []

    for path, name in zip(raw_paths, names):
        item: dict = {"name": name, "flags": []}
        try:
            item.update(_quick_clip_metrics(path))
        except Exception as exc:
            item["flags"].append(f"unreadable:{exc}")
            items.append(item)
            embeddings.append(None)
            continue

        emb = None
        try:
            _gpt, spk = model.get_conditioning_latents([path])
            emb = spk.detach().cpu().flatten().float()
            emb = emb / (emb.norm() + 1e-9)
        except Exception as exc:
            item["flags"].append(f"embed_failed:{exc}")

        embeddings.append(emb)
        items.append(item)

    # Consistency: cosine similarity of each embedding to the centroid.
    valid = [e for e in embeddings if e is not None]
    centroid = None
    if valid:
        centroid = torch.stack(valid).mean(dim=0)
        centroid = centroid / (centroid.norm() + 1e-9)

    for item, emb in zip(items, embeddings):
        item["consistency"] = (
            round(float(torch.dot(emb, centroid).item()), 3)
            if (emb is not None and centroid is not None) else None
        )

    # Score each clip (0–100) and build flags.
    for item in items:
        snr = item.get("snr_db")
        q_snr = _clamp01((snr - 10.0) / 25.0) if snr is not None else 0.7
        speech = item.get("speech_seconds") or 0.0
        q_dur = _clamp01(speech / 6.0)
        q_clip = 1.0 - _clamp01((item.get("clip_ratio") or 0.0) / 0.02)
        quality = 0.4 * q_snr + 0.35 * q_dur + 0.25 * q_clip

        cons = item.get("consistency")
        cons_norm = _clamp01((cons - 0.5) / 0.45) if cons is not None else 0.5

        score = round((0.65 * quality + 0.35 * cons_norm) * 100)
        item["score"] = score
        item["_anchor"] = 0.45 * quality + 0.3 * cons_norm + 0.25 * _clamp01(speech / float(GPT_COND_LEN))

        if (item.get("clip_ratio") or 0.0) > 0.01:
            item["flags"].append("clipping")
        if snr is not None and snr < 15:
            item["flags"].append("noisy")
        if speech and speech < 2.0:
            item["flags"].append("very_short")
        if cons_norm < 0.3:
            item["flags"].append("outlier")
        item["recommended"] = score >= ANALYZE_SCORE_THRESHOLD
        item["flags"] = sorted(set(item["flags"]))

    # Optimal sequence: best anchor first (drives the GPT conditioning window).
    order = sorted(range(len(items)), key=lambda i: items[i]["_anchor"], reverse=True)
    for rank, idx in enumerate(order):
        items[idx]["rank"] = rank
    for item in items:
        item.pop("_anchor", None)

    return {
        "clips": items,
        "order": order,
        "threshold": ANALYZE_SCORE_THRESHOLD,
        "gpt_cond_len": GPT_COND_LEN,
    }


@app.post("/analyze-clips")
async def analyze_clips(audio: list[UploadFile] = File(...)):
    """Score candidate clips and recommend a subset + ordering (no clone made)."""
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Model still loading"})

    clips = audio if isinstance(audio, list) else [audio]
    if not clips:
        return JSONResponse(status_code=422, content={"error": "No clips provided"})

    await _mgr.ensure_gpu()

    raw_paths: list[str] = []
    names: list[str] = []
    try:
        for clip in clips:
            data = await clip.read()
            suffix = os.path.splitext(clip.filename or "")[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(data)
                raw_paths.append(f.name)
            names.append(clip.filename or f"clip-{len(names) + 1}")
        result = await asyncio.to_thread(_analyze_clips, raw_paths, names)
    except RuntimeError as exc:
        err_str = str(exc).lower()
        if "out of memory" in err_str or "cuda" in err_str:
            return JSONResponse(status_code=503, content={"error": "GPU OOM", "detail": str(exc)})
        raise
    finally:
        for path in raw_paths:
            try:
                os.unlink(path)
            except OSError:
                pass

    return result


# ── POST /clone-voice ─────────────────────────────────────────────────────────


@app.post("/clone-voice")
async def clone_voice(
    audio: list[UploadFile] = File(...),
    reference_text: str = Form(""),
    companion_id: str = Form(...),
    voice_name: str = Form(""),
):
    """Compute XTTS conditioning latents from one or more reference clips.

    XTTS's ``get_conditioning_latents`` accepts a *list* of audio files: it
    concatenates them for the GPT conditioning and averages the per-clip
    speaker embeddings. A single 6 s clip works, but several clips (varied
    prosody, clean takes) yield a more robust, less noise-sensitive clone — so
    this endpoint accepts the `audio` form field repeated N times.

    When `XTTS_PREPROCESS` is on (default), each clip is standardised first
    (mono → resample → high-pass → Silero VAD trim → LUFS normalise); the
    response includes a per-clip `clips` report of what changed.

    `reference_text` is accepted for API parity with the Qwen sidecar but XTTS
    does not require a transcript to clone.
    """
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Model still loading"})

    clips = audio if isinstance(audio, list) else [audio]
    if not clips:
        return JSONResponse(status_code=422, content={"error": "At least one reference clip is required"})

    await _mgr.ensure_gpu()

    raw_paths: list[str] = []
    latent_paths: list[str] = []  # what we feed to XTTS (preprocessed when enabled)
    reports: list[dict] = []
    try:
        total_duration = 0.0
        for clip in clips:
            clip_bytes = await clip.read()
            suffix = os.path.splitext(clip.filename or "")[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(clip_bytes)
                raw_paths.append(f.name)

            if PREPROCESS:
                try:
                    proc_path, report = await asyncio.to_thread(_preprocess_clip, raw_paths[-1])
                    report["name"] = clip.filename
                    latent_paths.append(proc_path)
                    reports.append(report)
                    total_duration += report.get("kept_seconds") or 0.0
                    continue
                except Exception as exc:
                    logger.warning("clip_preprocess_failed", name=clip.filename, error=str(exc))
                    reports.append({"name": clip.filename, "warnings": [f"preprocess_failed:{exc}"], "applied": []})
                    # fall through to use the raw clip

            latent_paths.append(raw_paths[-1])
            dur = await asyncio.to_thread(_audio_duration, raw_paths[-1])
            total_duration += dur or 0.0

        # total_duration stays 0 only when no clip could be probed (e.g. exotic
        # container); let get_conditioning_latents be the real arbiter then.
        if total_duration and total_duration < 3.0:
            return JSONResponse(
                status_code=422,
                content={"error": "Usable speech totals under 3 seconds after preprocessing (6+ recommended)"},
            )

        try:
            gpt_cond_latent, speaker_embedding = await asyncio.to_thread(
                _conditioning_latents, latent_paths
            )
        except RuntimeError as exc:
            err_str = str(exc).lower()
            if "out of memory" in err_str or "cuda" in err_str:
                gpu_info = _gpu_memory_info()
                logger.error("xtts_clone_gpu_oom", error=str(exc), gpu=gpu_info)
                return JSONResponse(status_code=503, content={"error": "GPU OOM", "detail": str(exc), "gpu": gpu_info})
            raise
    finally:
        for path in set(raw_paths + latent_paths):
            try:
                os.unlink(path)
            except OSError:
                pass

    import torch

    prompt = {
        "gpt_cond_latent": gpt_cond_latent.detach().cpu(),
        "speaker_embedding": speaker_embedding.detach().cpu(),
    }
    buf = io.BytesIO()
    torch.save(prompt, buf)
    prompt_bytes = buf.getvalue()

    s3_key = f"voice_clones/{companion_id}/prompt.pt"
    await asyncio.to_thread(_upload_to_s3, prompt_bytes, s3_key, "application/octet-stream")

    meta = {
        "voice_clone_id": s3_key,
        "companion_id": companion_id,
        "name": voice_name or companion_id,
        "num_clips": len(clips),
        "total_seconds": round(total_duration, 2),
        "preprocessed": PREPROCESS,
        "created_at": time.time(),
    }
    meta_key = f"voice_clones/{companion_id}/meta.json"
    await asyncio.to_thread(
        _upload_to_s3, json.dumps(meta).encode("utf-8"), meta_key, "application/json"
    )

    with _clone_cache_lock:
        _clone_cache[s3_key] = prompt
        if len(_clone_cache) > _CLONE_CACHE_MAX:
            _clone_cache.popitem(last=False)

    logger.info(
        "voice_cloned",
        companion_id=companion_id,
        clips=len(clips),
        duration=round(total_duration, 2),
        s3_key=s3_key,
    )
    return {**meta, "duration": total_duration, "clips": reports}


# ── GET /clones ───────────────────────────────────────────────────────────────


def _list_clones() -> list[dict]:
    """Enumerate stored clones from S3 by reading their meta.json sidecars."""
    from botocore.exceptions import ClientError

    client = _get_s3_client()
    clones: list[dict] = []
    paginator = client.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix="voice_clones/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("/meta.json"):
                    continue
                buf = io.BytesIO()
                try:
                    client.download_fileobj(S3_BUCKET_NAME, key, buf)
                    clones.append(json.loads(buf.getvalue().decode("utf-8")))
                except (ClientError, ValueError, json.JSONDecodeError) as exc:
                    logger.warning("clone_meta_unreadable", key=key, error=str(exc))
    except ClientError as exc:
        logger.error("clone_list_failed", error=str(exc))
        raise
    clones.sort(key=lambda c: c.get("created_at", 0), reverse=True)
    return clones


@app.get("/clones")
async def list_clones():
    """List stored voice clones (id, name, clip count, duration)."""
    try:
        clones = await asyncio.to_thread(_list_clones)
    except Exception as exc:
        logger.error("clone_list_error", error=str(exc))
        return JSONResponse(status_code=500, content={"error": f"Failed to list clones: {exc}"})
    return {"clones": clones}


def _build_clone_zip(companion_id: str) -> bytes:
    """Bundle a clone's prompt.pt (+ meta.json if present) into a zip."""
    import zipfile
    from botocore.exceptions import ClientError

    client = _get_s3_client()
    prompt_key = f"voice_clones/{companion_id}/prompt.pt"
    meta_key = f"voice_clones/{companion_id}/meta.json"

    prompt_buf = io.BytesIO()
    try:
        client.download_fileobj(S3_BUCKET_NAME, prompt_key, prompt_buf)
    except ClientError as exc:
        raise FileNotFoundError(f"Voice clone not found: {companion_id}") from exc

    meta_buf = io.BytesIO()
    have_meta = True
    try:
        client.download_fileobj(S3_BUCKET_NAME, meta_key, meta_buf)
    except ClientError:
        have_meta = False

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prompt.pt", prompt_buf.getvalue())
        if have_meta:
            zf.writestr("meta.json", meta_buf.getvalue())
    return zip_buf.getvalue()


@app.get("/clones/{companion_id}/download")
async def download_clone(companion_id: str):
    """Download a clone's latents + metadata sidecar as a zip."""
    try:
        data = await asyncio.to_thread(_build_clone_zip, companion_id)
    except FileNotFoundError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    except Exception as exc:
        logger.error("clone_download_error", companion_id=companion_id, error=str(exc))
        return JSONResponse(status_code=500, content={"error": f"Failed to bundle clone: {exc}"})

    headers = {"Content-Disposition": f'attachment; filename="{companion_id}.zip"'}
    return Response(content=data, media_type="application/zip", headers=headers)


def _slug_id(value: str) -> str:
    s = (value or "").strip().lower()
    out = "".join(ch if ch.isalnum() else "-" for ch in s).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out or f"imported-{int(time.time())}"


_MAX_PROMPT_BYTES = 64 * 1024 * 1024


def _import_clone(zip_bytes: bytes, companion_override: str, name_override: str) -> dict:
    """Validate and store a clone archive (prompt.pt [+ meta.json]).

    The uploaded ``prompt.pt`` is untrusted, so it's loaded with
    ``weights_only=True`` (tensors only — no arbitrary pickle execution) and
    then re-serialised from the validated tensors, guaranteeing the stored file
    is clean regardless of what the archive contained.
    """
    import zipfile

    import torch

    try:
        z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Not a valid zip archive: {exc}") from exc

    names = z.namelist()
    prompt_name = next((n for n in names if n.rsplit("/", 1)[-1] == "prompt.pt"), None)
    if not prompt_name:
        raise ValueError("Archive does not contain prompt.pt")
    if z.getinfo(prompt_name).file_size > _MAX_PROMPT_BYTES:
        raise ValueError("prompt.pt is unexpectedly large")

    try:
        loaded = torch.load(io.BytesIO(z.read(prompt_name)), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Invalid or untrusted prompt.pt (not a tensor archive): {exc}") from exc

    if not (isinstance(loaded, dict) and "gpt_cond_latent" in loaded and "speaker_embedding" in loaded):
        raise ValueError("prompt.pt must contain gpt_cond_latent and speaker_embedding")
    gpt, spk = loaded["gpt_cond_latent"], loaded["speaker_embedding"]
    if not (torch.is_tensor(gpt) and torch.is_tensor(spk)):
        raise ValueError("prompt.pt latents are not tensors")

    src_meta: dict = {}
    meta_name = next((n for n in names if n.rsplit("/", 1)[-1] == "meta.json"), None)
    if meta_name:
        try:
            src_meta = json.loads(z.read(meta_name).decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            src_meta = {}

    companion_id = (companion_override or src_meta.get("companion_id") or "").strip()
    companion_id = _slug_id(companion_id or name_override or src_meta.get("name") or "imported-voice")

    prompt = {"gpt_cond_latent": gpt.detach().cpu(), "speaker_embedding": spk.detach().cpu()}
    buf = io.BytesIO()
    torch.save(prompt, buf)
    s3_key = f"voice_clones/{companion_id}/prompt.pt"
    _upload_to_s3(buf.getvalue(), s3_key, "application/octet-stream")

    meta = {
        "voice_clone_id": s3_key,
        "companion_id": companion_id,
        "name": name_override or src_meta.get("name") or companion_id,
        "num_clips": src_meta.get("num_clips", 0),
        "total_seconds": src_meta.get("total_seconds", 0),
        "imported": True,
        "created_at": time.time(),
    }
    _upload_to_s3(json.dumps(meta).encode("utf-8"), f"voice_clones/{companion_id}/meta.json", "application/json")

    with _clone_cache_lock:
        _clone_cache[s3_key] = prompt
        if len(_clone_cache) > _CLONE_CACHE_MAX:
            _clone_cache.popitem(last=False)

    logger.info("voice_imported", companion_id=companion_id, s3_key=s3_key)
    return {**meta, "duration": meta["total_seconds"]}


@app.post("/clones/import")
async def import_clone(
    archive: UploadFile = File(...),
    voice_name: str = Form(""),
    companion_id: str = Form(""),
):
    """Import a previously downloaded voice archive (prompt.pt + meta.json zip)."""
    data = await archive.read()
    try:
        meta = await asyncio.to_thread(_import_clone, data, companion_id, voice_name)
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})
    except Exception as exc:
        logger.error("clone_import_error", error=str(exc))
        return JSONResponse(status_code=500, content={"error": f"Failed to import voice: {exc}"})
    return meta


# ── GET /speakers ─────────────────────────────────────────────────────────────


@app.get("/speakers")
async def list_speakers():
    """List built-in XTTS studio speakers."""
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Model still loading"})

    await _mgr.ensure_gpu()

    names = _list_speakers()
    supported_langs = sorted(set(ISO_TO_XTTS.values()))
    speakers = [
        {"id": name, "name": name, "language": "en", "languages": supported_langs}
        for name in names
    ]
    return {"speakers": speakers}


# ── GET /health ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {
        "status": "ok",
        "engine": "xtts-v2",
        "model": XTTS_MODEL,
        "device": DEVICE,
        "model_state": _mgr.state.value,
        "keep_alive_gpu": KEEP_ALIVE_GPU,
        "keep_alive_cpu": KEEP_ALIVE_CPU,
        "gpu": _gpu_memory_info(),
    }


@app.post("/admin/evict")
async def admin_evict():
    """Force-demote the model from GPU to CPU, freeing VRAM immediately."""
    await _mgr.demote_to_cpu()
    return {"status": "evicted", "model_state": _mgr.state.value}


@app.post("/admin/unload")
async def admin_unload():
    """Fully unload the model from memory (GPU + CPU)."""
    await _mgr.unload()
    return {"status": "unloaded", "model_state": _mgr.state.value}


@app.post("/admin/wake")
async def admin_wake():
    """Load the model and promote to GPU."""
    await _mgr.ensure_gpu()
    return {"status": "ready", "model_state": _mgr.state.value}

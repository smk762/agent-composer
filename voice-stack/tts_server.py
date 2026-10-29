"""
Qwen3-TTS server — FastAPI, port 8031.

Voice-stack production version. Based on playact-engine's tts_server.py with
enhanced API surface for multiple consumers (playact-engine, rag-chat,
mithrandir, kimini-api).

Loads two model variants:
  - Base: voice cloning from reference audio
  - CustomVoice: predefined speakers + instruction-based emotion

GPU memory lifecycle (three-state, like ModernBERT):
  unloaded → cpu → gpu

  On request the models are promoted to GPU; after TTS_KEEP_ALIVE_GPU seconds
  of idle time they are demoted back to CPU, freeing VRAM for other services
  (Ollama, Whisper, Infinity).  After a further TTS_KEEP_ALIVE_CPU seconds
  on CPU they are fully unloaded.

Env vars:
  TTS_MODEL_SIZE       — "0.6b" (default) or "1.7b"
  TTS_DEVICE           — "cuda:0" (default) or "cpu"
  TTS_LOAD_MODELS      — "both" (default), "custom_voice", or "base"
  TTS_PRELOAD          — "0" (default) lazy-load on first request;
                          "1" to load into GPU at startup
  TTS_KEEP_ALIVE_GPU   — seconds on GPU after last request before demoting
                          to CPU.  -1 = forever, 0 = immediate.  Default 120.
  TTS_KEEP_ALIVE_CPU   — seconds on CPU after GPU eviction before full
                          unload.  -1 = forever, 0 = immediate.  Default 600.
  S3_ENDPOINT_URL      — MinIO/S3 endpoint
  S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY / S3_BUCKET_NAME / S3_PUBLIC_URL
"""

import asyncio
import gc
import io
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
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────


def _normalise_model_size(raw_size: str) -> str:
    normalised = raw_size.strip().lower()
    if normalised not in {"1.7b", "0.6b"}:
        raise ValueError("TTS_MODEL_SIZE must be one of: 1.7b, 1.7B, 0.6b, 0.6B")
    return normalised


MODEL_SIZE = _normalise_model_size(os.getenv("TTS_MODEL_SIZE", "0.6b"))
DEVICE = os.getenv("TTS_DEVICE", "cuda:0")
LOAD_MODELS = os.getenv("TTS_LOAD_MODELS", "both")
PRELOAD = os.getenv("TTS_PRELOAD", "0") == "1"
KEEP_ALIVE_GPU: float = float(os.getenv("TTS_KEEP_ALIVE_GPU", "120"))
KEEP_ALIVE_CPU: float = float(os.getenv("TTS_KEEP_ALIVE_CPU", "600"))
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "minioadmin")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "minioadmin")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "voice")
S3_PUBLIC_URL = os.getenv("S3_PUBLIC_URL", "http://localhost:9000/voice")

# Cap autoregressive codec-token generation. The talker default
# (max_new_tokens=4096 ≈ 340s @12Hz) with do_sample=True can occasionally fail
# to emit EOS and run away, producing minute-long synthesis for a short clip.
# 0 = auto-size from text length; >0 = fixed cap.
TTS_MAX_NEW_TOKENS = int(os.getenv("TTS_MAX_NEW_TOKENS", "0"))
TTS_MAX_NEW_TOKENS_CEILING = int(os.getenv("TTS_MAX_NEW_TOKENS_CEILING", "1024"))

MODEL_IDS = {
    "1.7b": {
        "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    },
    "0.6b": {
        "base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "custom_voice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    },
}

EMOTION_TO_INSTRUCTION: dict[str, str] = {
    "happy": "speak happily and enthusiastically",
    "sad": "speak softly with a melancholic tone",
    "excited": "speak with high energy and excitement",
    "flirty": "speak warmly and playfully",
    "calm": "speak calmly and soothingly",
    "neutral": "",
}

ISO_TO_LANGUAGE: dict[str, str] = {
    "auto": "auto",
    "zh": "chinese", "en": "english", "fr": "french", "de": "german",
    "it": "italian", "ja": "japanese", "ko": "korean", "pt": "portuguese",
    "ru": "russian", "es": "spanish",
    "chinese": "chinese", "english": "english", "french": "french",
    "german": "german", "italian": "italian", "japanese": "japanese",
    "korean": "korean", "portuguese": "portuguese", "russian": "russian",
    "spanish": "spanish",
}

# Reverse mapping: full language name → ISO 639-1 code
_LANGUAGE_TO_ISO: dict[str, str] = {
    "auto": "auto", "chinese": "zh", "english": "en", "french": "fr",
    "german": "de", "italian": "it", "japanese": "ja", "korean": "ko",
    "portuguese": "pt", "russian": "ru", "spanish": "es",
}

_CLONE_CACHE_MAX = 50

# ── GPU memory lifecycle ─────────────────────────────────────────────────────


class TTSState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    CPU = "cpu"
    GPU = "gpu"
    ERROR = "error"


class TTSModelManager:
    """Three-state lifecycle: unloaded → cpu → gpu.

    Models are loaded to CPU first, then promoted to GPU on demand.
    After keep_alive_gpu seconds of idle the models demote to CPU.
    After keep_alive_cpu seconds on CPU the models fully unload.
    """

    def __init__(self) -> None:
        self.keep_alive_gpu = KEEP_ALIVE_GPU
        self.keep_alive_cpu = KEEP_ALIVE_CPU
        self._state = TTSState.UNLOADED
        self._error: Optional[str] = None
        self._base_model = None
        self._custom_model = None
        self._last_used: float = 0.0
        self._lock = asyncio.Lock()
        self._evict_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> TTSState:
        return self._state

    @property
    def base_model(self):
        return self._base_model

    @property
    def custom_model(self):
        return self._custom_model

    @property
    def is_ready(self) -> bool:
        return self._state in (TTSState.CPU, TTSState.GPU)

    def _load_to_cpu(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        dtype = torch.bfloat16
        ids = MODEL_IDS[MODEL_SIZE]

        def _load(model_id: str):
            # Prefer torch's built-in SDPA attention over the slow "manual PyTorch"
            # path used when flash-attn is absent. Fall back if the model rejects
            # the kwarg (custom model may not honor HF's attn dispatch).
            try:
                return Qwen3TTSModel.from_pretrained(
                    model_id, device_map="cpu", dtype=dtype,
                    attn_implementation="sdpa",
                )
            except (TypeError, ValueError) as exc:
                logger.warning("tts_sdpa_unsupported", error=str(exc))
                return Qwen3TTSModel.from_pretrained(
                    model_id, device_map="cpu", dtype=dtype,
                )

        if LOAD_MODELS in ("both", "base"):
            logger.info("loading_qwen3_tts_base", model=ids["base"], target="cpu")
            self._base_model = _load(ids["base"])
            logger.info("qwen3_tts_base_loaded_cpu")

        if LOAD_MODELS in ("both", "custom_voice"):
            logger.info("loading_qwen3_tts_custom_voice", model=ids["custom_voice"], target="cpu")
            self._custom_model = _load(ids["custom_voice"])
            logger.info("qwen3_tts_custom_voice_loaded_cpu")

    def _move_to_device(self, device: str) -> None:
        import torch

        target = torch.device(device)
        # Qwen3TTSModel is a plain wrapper with no .to(); the real nn.Module lives
        # on `.model`. Move that, then sync the wrapper's cached `.device` so
        # generate() places inputs on the same device as the weights.
        for wrapper in (self._base_model, self._custom_model):
            if wrapper is None:
                continue
            inner = getattr(wrapper, "model", None)
            if inner is not None and hasattr(inner, "to"):
                inner.to(target)
                wrapper.device = getattr(inner, "device", target)
            elif hasattr(wrapper, "to") and callable(wrapper.to):
                wrapper.to(target)

        if device == "cpu":
            torch.cuda.empty_cache()

    async def ensure_gpu(self) -> None:
        """Guarantee models are on GPU and reset the idle timer."""
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
                    logger.error("tts_load_failed", error=str(exc))
                    raise

            if self._state == TTSState.CPU:
                if DEVICE.startswith("cuda"):
                    logger.info("tts_promoting_to_gpu", device=DEVICE)
                    await asyncio.to_thread(self._move_to_device, DEVICE)
                    self._state = TTSState.GPU
                    logger.info("tts_on_gpu")
                else:
                    self._state = TTSState.GPU  # CPU-only: treat as "ready"

            self._last_used = time.time()
            self._arm_evict()

    async def demote_to_cpu(self) -> None:
        async with self._lock:
            if self._state != TTSState.GPU:
                return
            if DEVICE == "cpu":
                return
            logger.info("tts_demoting_to_cpu")
            await asyncio.to_thread(self._move_to_device, "cpu")
            self._state = TTSState.CPU
            logger.info("tts_on_cpu")

    async def unload(self) -> None:
        async with self._lock:
            if self._evict_task and not self._evict_task.done():
                self._evict_task.cancel()
            for attr in ("_base_model", "_custom_model"):
                model = getattr(self, attr)
                if model is not None:
                    del model
                    setattr(self, attr, None)
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._state = TTSState.UNLOADED
            logger.info("tts_fully_unloaded")

    def _arm_evict(self) -> None:
        if self._evict_task and not self._evict_task.done():
            self._evict_task.cancel()
        self._evict_task = asyncio.create_task(self._eviction_loop())

    async def _eviction_loop(self) -> None:
        try:
            # GPU keep-alive
            if self.keep_alive_gpu == 0:
                await self.demote_to_cpu()
            elif self.keep_alive_gpu > 0:
                await asyncio.sleep(self.keep_alive_gpu)
                if time.time() - self._last_used >= self.keep_alive_gpu:
                    await self.demote_to_cpu()

            # CPU keep-alive
            if self.keep_alive_cpu == 0:
                await self.unload()
            elif self.keep_alive_cpu > 0:
                await asyncio.sleep(self.keep_alive_cpu)
                if time.time() - self._last_used >= self.keep_alive_gpu + self.keep_alive_cpu:
                    await self.unload()
        except asyncio.CancelledError:
            pass


# ── Global state ─────────────────────────────────────────────────────────────

_mgr = TTSModelManager()
_s3_client = None
_clone_cache: OrderedDict = OrderedDict()
_clone_cache_lock = threading.Lock()


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


def _get_clone_prompt(clone_id: str):
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
        logger.error("clone_prompt_s3_error", clone_id=clone_id, error_code=error_code)
        raise ValueError(f"Voice clone prompt not found: {clone_id}") from exc
    buf.seek(0)
    prompt = torch.load(buf, map_location="cpu", weights_only=True)

    with _clone_cache_lock:
        _clone_cache[clone_id] = prompt
        if len(_clone_cache) > _CLONE_CACHE_MAX:
            _clone_cache.popitem(last=False)

    return prompt


def _gpu_memory_info() -> dict | None:
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


def _get_supported_languages() -> list[str]:
    """Return ISO codes for languages the model supports."""
    return sorted(set(_LANGUAGE_TO_ISO.values()) - {"auto"})


# ── App lifecycle ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    if PRELOAD:
        await _mgr.ensure_gpu()
    yield
    await _mgr.unload()


app = FastAPI(title="Voice Stack — Qwen3-TTS", version="1.1.0", lifespan=lifespan)


# ── Request / Response models ─────────────────────────────────────────────────


class SynthesiseRequest(BaseModel):
    text: str
    voice_clone_id: Optional[str] = None
    speaker: Optional[str] = None
    voice_description: Optional[str] = None
    instruction: Optional[str] = None
    emotion: str = "neutral"
    language: str = Field(default="en")
    speed: Optional[float] = None  # reserved for future use


# ── POST /synthesise ──────────────────────────────────────────────────────────


def _max_new_tokens_for(text: str) -> int:
    """Bound the talker's codec-token budget so a runaway sample can't generate
    minutes of audio for a short clip. The talker runs at ~12 Hz; English needs
    well under one frame per character, so 4x + a floor is generous headroom."""
    if TTS_MAX_NEW_TOKENS > 0:
        return TTS_MAX_NEW_TOKENS
    est = int(len(text or "") * 4) + 64
    return max(192, min(est, TTS_MAX_NEW_TOKENS_CEILING))


@app.post("/synthesise")
async def synthesise(req: SynthesiseRequest):
    """Generate speech from text.

    Uses Base model if voice_clone_id is provided (voice cloning path).
    Uses CustomVoice model otherwise (predefined speaker + instruction path).
    """
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Models still loading"})

    t_req = time.monotonic()
    await _mgr.ensure_gpu()
    t_ready = time.monotonic()

    instruction = req.instruction
    if instruction is None:
        instruction = EMOTION_TO_INSTRUCTION.get(req.emotion, "")

    language = ISO_TO_LANGUAGE.get(req.language.lower(), req.language.lower())
    if language not in ISO_TO_LANGUAGE.values():
        return JSONResponse(
            status_code=422,
            content={
                "error": f"Unsupported language: {req.language!r}. "
                f"Supported: {sorted(set(ISO_TO_LANGUAGE.values()))}"
            },
        )

    job_id = str(uuid.uuid4())
    max_new_tokens = _max_new_tokens_for(req.text)

    t_gen0 = time.monotonic()
    try:
        if req.voice_clone_id:
            if _mgr.base_model is None:
                return JSONResponse(status_code=503, content={"error": "Base model not loaded (TTS_LOAD_MODELS=custom_voice)"})
            clone_prompt = await asyncio.to_thread(_get_clone_prompt, req.voice_clone_id)
            audio_arrays, sample_rate = await asyncio.to_thread(
                _mgr.base_model.generate_voice_clone,
                req.text, voice_clone_prompt=clone_prompt, language=language,
                max_new_tokens=max_new_tokens,
            )

        elif req.voice_description:
            design_instruct = req.voice_description
            if instruction:
                design_instruct = f"{req.voice_description}. {instruction}"

            if _mgr.base_model is not None:
                audio_arrays, sample_rate = await asyncio.to_thread(
                    _mgr.base_model.generate_voice_design,
                    req.text, instruct=design_instruct, language=language,
                    max_new_tokens=max_new_tokens,
                )
            elif _mgr.custom_model is not None:
                available = _mgr.custom_model.get_supported_speakers()
                fallback_speaker = req.speaker or sorted(available)[0]
                logger.info("voice_design_fallback_to_speaker", speaker=fallback_speaker)
                audio_arrays, sample_rate = await asyncio.to_thread(
                    _mgr.custom_model.generate_custom_voice,
                    text=req.text, speaker=fallback_speaker,
                    instruct=design_instruct, language=language,
                    max_new_tokens=max_new_tokens,
                )
            else:
                return JSONResponse(status_code=503, content={"error": "No model loaded"})

        else:
            if _mgr.custom_model is None:
                return JSONResponse(status_code=503, content={"error": "CustomVoice model not loaded (TTS_LOAD_MODELS=base)"})
            speaker = req.speaker
            if not speaker:
                available = _mgr.custom_model.get_supported_speakers()
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": "Either 'speaker', 'voice_description', or 'voice_clone_id' is required.",
                        "available_speakers": list(available),
                    },
                )
            generate_kwargs: dict = {
                "text": req.text,
                "speaker": speaker,
                "language": language,
                "max_new_tokens": max_new_tokens,
            }
            if instruction:
                generate_kwargs["instruct"] = instruction
            audio_arrays, sample_rate = await asyncio.to_thread(
                _mgr.custom_model.generate_custom_voice, **generate_kwargs,
            )

    except RuntimeError as exc:
        err_str = str(exc).lower()
        if "out of memory" in err_str or "cuda" in err_str:
            gpu_info = _gpu_memory_info()
            logger.error("tts_gpu_oom", error=str(exc), gpu=gpu_info)
            return JSONResponse(status_code=503, content={"error": "GPU OOM", "detail": str(exc), "gpu": gpu_info})
        raise
    except ValueError as exc:
        logger.warning("tts_validation_error", error=str(exc))
        return JSONResponse(status_code=422, content={"error": str(exc)})
    except Exception as exc:
        logger.error("tts_generation_failed", error=str(exc), exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"TTS generation failed: {exc}"})

    t_gen1 = time.monotonic()

    import numpy as np

    audio_data = np.concatenate(audio_arrays) if len(audio_arrays) > 1 else audio_arrays[0]

    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        await asyncio.to_thread(sf.write, tmp_path, audio_data, sample_rate)
        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()
        info = await asyncio.to_thread(sf.info, tmp_path)
        duration = info.duration
    finally:
        os.unlink(tmp_path)

    t_encoded = time.monotonic()

    s3_key = f"generated/voice/{job_id}.wav"
    audio_url = await asyncio.to_thread(_upload_to_s3, wav_bytes, s3_key)

    t_uploaded = time.monotonic()

    logger.info(
        "tts_synthesised",
        job_id=job_id,
        duration=round(duration, 2),
        language=req.language,
        max_new_tokens=max_new_tokens,
        ready_s=round(t_ready - t_req, 2),
        generate_s=round(t_gen1 - t_gen0, 2),
        encode_s=round(t_encoded - t_gen1, 2),
        upload_s=round(t_uploaded - t_encoded, 2),
        total_s=round(t_uploaded - t_req, 2),
        rtf=round((t_gen1 - t_gen0) / duration, 1) if duration else None,
    )

    return {
        "audio_url": audio_url,
        "duration": duration,
        "duration_seconds": round(duration, 3),
        "format": "wav",
    }


# ── POST /clone-voice ─────────────────────────────────────────────────────────


@app.post("/clone-voice")
async def clone_voice(
    audio: UploadFile = File(...),
    reference_text: str = Form(""),
    companion_id: str = Form(...),
):
    """Extract voice clone prompt from reference audio and store in S3."""
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Models still loading"})

    await _mgr.ensure_gpu()

    if _mgr.base_model is None:
        return JSONResponse(status_code=503, content={"error": "Base model not loaded (TTS_LOAD_MODELS=custom_voice)"})

    import soundfile as sf

    audio_bytes = await audio.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        info = await asyncio.to_thread(sf.info, tmp_path)
        duration = info.duration

        if duration < 3.0:
            return JSONResponse(status_code=422, content={"error": "Reference audio must be at least 3 seconds long"})

        use_xvector_only = not bool(reference_text)
        clone_prompt = await asyncio.to_thread(
            _mgr.base_model.create_voice_clone_prompt,
            tmp_path,
            ref_text=reference_text if reference_text else None,
            x_vector_only_mode=use_xvector_only,
        )
    except RuntimeError as exc:
        err_str = str(exc).lower()
        if "out of memory" in err_str or "cuda" in err_str:
            gpu_info = _gpu_memory_info()
            logger.error("tts_clone_gpu_oom", error=str(exc), gpu=gpu_info)
            return JSONResponse(status_code=503, content={"error": "GPU OOM", "detail": str(exc), "gpu": gpu_info})
        raise
    finally:
        os.unlink(tmp_path)

    import torch

    buf = io.BytesIO()
    torch.save(clone_prompt, buf)
    prompt_bytes = buf.getvalue()

    s3_key = f"voice_clones/{companion_id}/prompt.pt"
    await asyncio.to_thread(_upload_to_s3, prompt_bytes, s3_key, "application/octet-stream")

    with _clone_cache_lock:
        _clone_cache[s3_key] = clone_prompt
        if len(_clone_cache) > _CLONE_CACHE_MAX:
            _clone_cache.popitem(last=False)

    logger.info("voice_cloned", companion_id=companion_id, duration=duration, s3_key=s3_key)
    return {"voice_clone_id": s3_key, "companion_id": companion_id, "duration": duration}


# ── GET /speakers ─────────────────────────────────────────────────────────────


@app.get("/speakers")
async def list_speakers():
    """List available predefined CustomVoice speakers.

    Returns structured objects per speaker with id, name, and language.
    Consumers: mithrandir (voice registry validation), rag-chat (UI dropdown),
    playact-engine (speaker selection).
    """
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Models still loading"})

    await _mgr.ensure_gpu()

    if _mgr.custom_model is None:
        return JSONResponse(status_code=503, content={"error": "CustomVoice model not loaded"})

    raw_speakers = await asyncio.to_thread(_mgr.custom_model.get_supported_speakers)
    supported_langs = _get_supported_languages()

    speakers = [
        {
            "id": name,
            "name": name,
            "language": "en",
            "languages": supported_langs,
        }
        for name in sorted(raw_speakers)
    ]

    return {"speakers": speakers}


# ── GET /health ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"status": "loading"})
    gpu_info = _gpu_memory_info()
    return {
        "status": "ok",
        "engine": "qwen-tts",
        "model": MODEL_SIZE,
        "model_size": MODEL_SIZE,
        "device": DEVICE,
        "load_models": LOAD_MODELS,
        "model_state": _mgr.state.value,
        "keep_alive_gpu": KEEP_ALIVE_GPU,
        "keep_alive_cpu": KEEP_ALIVE_CPU,
        "gpu": gpu_info,
    }


@app.post("/admin/evict")
async def admin_evict():
    """Force-demote models from GPU to CPU, freeing VRAM immediately."""
    await _mgr.demote_to_cpu()
    return {"status": "evicted", "model_state": _mgr.state.value}


@app.post("/admin/unload")
async def admin_unload():
    """Fully unload models from memory (GPU + CPU)."""
    await _mgr.unload()
    return {"status": "unloaded", "model_state": _mgr.state.value}


@app.post("/admin/wake")
async def admin_wake():
    """Load models and promote to GPU."""
    await _mgr.ensure_gpu()
    return {"status": "ready", "model_state": _mgr.state.value}

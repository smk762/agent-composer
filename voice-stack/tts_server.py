"""
Qwen3-TTS server — FastAPI, port 8031.

Voice-stack production version. Based on playact-engine's tts_server.py with
enhanced API surface for multiple consumers (playact-engine, rag-chat,
mithrandir, kimini-api).

Loads two model variants:
  - Base: voice cloning from reference audio
  - CustomVoice: predefined speakers + instruction-based emotion

GPU memory lifecycle (three-state, like ModernBERT):
  unloaded → gpu → cpu

  The first request loads with device_map set to TTS_DEVICE. Loading onto
  CPU and then calling Module.to("cuda") leaves Accelerate dispatch hooks
  in place: weights report cuda but generation runs tens of times slower
  than real time. The CPU copy is only produced later, by .to("cpu") of a
  model that was loaded on CUDA, which stays on the fast path when promoted
  again. After TTS_KEEP_ALIVE_GPU seconds of idle the models demote to CPU,
  freeing VRAM for other services. After a further TTS_KEEP_ALIVE_CPU
  seconds on CPU they are fully unloaded.

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
import re
import tempfile
import threading
import time
import uuid
import weakref
from collections import OrderedDict
from contextlib import asynccontextmanager
from enum import Enum
from typing import AsyncIterator, Optional

import structlog
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
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
# Used when a request names no speaker, clone, or voice description.
# A request `speaker` always wins.
DEFAULT_SPEAKER = os.getenv("TTS_DEFAULT_SPEAKER", "").strip()
PRELOAD = os.getenv("TTS_PRELOAD", "0") == "1"
KEEP_ALIVE_GPU: float = float(os.getenv("TTS_KEEP_ALIVE_GPU", "120"))
KEEP_ALIVE_CPU: float = float(os.getenv("TTS_KEEP_ALIVE_CPU", "600"))
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
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
    """Three-state lifecycle: unloaded → gpu → cpu.

    The first load uses device_map=<TTS_DEVICE>. A later idle demotes with
    .to("cpu"); the next request moves that same module back. Do not reload
    with device_map="cpu" and then .to("cuda") — Accelerate hooks survive
    that move and generation becomes ~50× real time.
    After keep_alive_gpu seconds of idle the models demote to CPU.
    After keep_alive_cpu seconds on CPU the models fully unload.

    Generation must hold a lease (``async with _mgr.use(): ...``) for as long as
    it touches the models. Demotion and unload wait until no lease is held, and
    the idle timer only starts when the last lease is released.
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
        self._in_use = 0
        self._idle = asyncio.Event()
        self._idle.set()

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

    def _load(self, device: str) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        dtype = torch.bfloat16
        ids = MODEL_IDS[MODEL_SIZE]
        # Single-device map on the real target. See the class docstring.
        device_map = device if str(device).startswith("cuda") else "cpu"

        def _load_one(model_id: str):
            # Prefer torch's built-in SDPA attention over the slow "manual PyTorch"
            # path used when flash-attn is absent. Fall back if the model rejects
            # the kwarg (custom model may not honor HF's attn dispatch).
            try:
                return Qwen3TTSModel.from_pretrained(
                    model_id, device_map=device_map, dtype=dtype,
                    attn_implementation="sdpa",
                )
            except (TypeError, ValueError) as exc:
                logger.warning("tts_sdpa_unsupported", error=str(exc))
                return Qwen3TTSModel.from_pretrained(
                    model_id, device_map=device_map, dtype=dtype,
                )

        if LOAD_MODELS in ("both", "base"):
            logger.info("loading_qwen3_tts_base", model=ids["base"], target=device_map)
            self._base_model = _load_one(ids["base"])
            logger.info("qwen3_tts_base_loaded", device=device_map)

        if LOAD_MODELS in ("both", "custom_voice"):
            logger.info("loading_qwen3_tts_custom_voice", model=ids["custom_voice"], target=device_map)
            self._custom_model = _load_one(ids["custom_voice"])
            logger.info("qwen3_tts_custom_voice_loaded", device=device_map)

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

    async def _make_ready(self) -> None:
        # Caller holds self._lock.
        if self._state == TTSState.GPU:
            return

        if self._state in (TTSState.UNLOADED, TTSState.ERROR):
            self._state = TTSState.LOADING
            try:
                await asyncio.to_thread(self._load, DEVICE)
                self._state = TTSState.GPU
                self._error = None
            except Exception as exc:
                self._state = TTSState.ERROR
                self._error = str(exc)
                logger.error("tts_load_failed", error=str(exc))
                raise
            return

        if self._state == TTSState.CPU:
            if DEVICE.startswith("cuda"):
                logger.info("tts_promoting_to_gpu", device=DEVICE)
                await asyncio.to_thread(self._move_to_device, DEVICE)
                self._state = TTSState.GPU
                logger.info("tts_on_gpu")
            else:
                self._state = TTSState.GPU  # CPU-only: treat as "ready"

    async def ensure_gpu(self) -> None:
        """Guarantee models are on GPU and reset the idle timer.

        Does not protect a following generation; use ``use()`` for that.
        """
        async with self._lock:
            await self._make_ready()
            self._last_used = time.time()
            if self._in_use == 0:
                self._arm_evict()

    @asynccontextmanager
    async def use(self) -> AsyncIterator["TTSModelManager"]:
        """Put the models on GPU and hold them there for the whole block.

        Don't call ensure_gpu/demote_to_cpu/unload inside the block: they take
        the lock a waiting demote/unload holds, which would deadlock.
        """
        async with self._lock:
            await self._make_ready()
            self._in_use += 1
            self._idle.clear()
            if self._evict_task and not self._evict_task.done():
                self._evict_task.cancel()
        try:
            yield self
        finally:
            self._in_use -= 1
            self._last_used = time.time()
            if self._in_use == 0:
                self._idle.set()
                self._arm_evict()

    async def _demote_locked(self) -> None:
        if self._state != TTSState.GPU:
            return
        if DEVICE == "cpu":
            self._state = TTSState.CPU  # nothing to move; lets the CPU stage run
            return
        logger.info("tts_demoting_to_cpu")
        await asyncio.to_thread(self._move_to_device, "cpu")
        self._state = TTSState.CPU
        logger.info("tts_on_cpu")

    async def demote_to_cpu(self) -> None:
        """Move GPU → CPU (waits for in-flight generations)."""
        async with self._lock:
            await self._idle.wait()
            await self._demote_locked()

    def _unload_locked(self) -> None:
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

    async def unload(self) -> None:
        """Full unload (waits for in-flight generations)."""
        async with self._lock:
            await self._idle.wait()
            if (self._evict_task and not self._evict_task.done()
                    and self._evict_task is not asyncio.current_task()):
                self._evict_task.cancel()
            self._unload_locked()

    def _arm_evict(self) -> None:
        if self._evict_task and not self._evict_task.done():
            self._evict_task.cancel()
        self._evict_task = asyncio.create_task(self._eviction_loop())

    async def _eviction_loop(self) -> None:
        # Armed when the models go idle; any new lease cancels it. A stage that
        # finds a lease held just stops: the lease's release re-arms the timer.
        try:
            # GPU keep-alive
            if self.keep_alive_gpu < 0:
                return  # stay on GPU forever
            if self._state == TTSState.GPU:
                if self.keep_alive_gpu > 0:
                    await asyncio.sleep(self.keep_alive_gpu)
                async with self._lock:
                    if self._state != TTSState.GPU or self._in_use:
                        return
                    await self._demote_locked()
            elif self._state != TTSState.CPU:
                return

            # CPU keep-alive (only once the models are actually on CPU)
            if self.keep_alive_cpu < 0:
                return
            if self.keep_alive_cpu > 0:
                await asyncio.sleep(self.keep_alive_cpu)
            async with self._lock:
                if self._state != TTSState.CPU or self._in_use:
                    return
                self._unload_locked()
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
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            return {
                "allocated_mb": round(allocated, 1),
                "reserved_mb": round(reserved, 1),
                "total_mb": round(total, 1),
            }
    except Exception as exc:
        logger.warning("gpu_memory_info_failed", error=str(exc))
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


async def _generate_audio(req: SynthesiseRequest):
    """Render speech. Returns ``(float32 samples, sample_rate)`` or a JSONResponse error.

    Uses Base model if voice_clone_id is provided (voice cloning path).
    Uses CustomVoice model otherwise (predefined speaker + instruction path).
    """
    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Models still loading"})

    async with _mgr.use():
        return await _generate_audio_leased(req)


async def _generate_audio_leased(req: SynthesiseRequest):
    """Body of ``_generate_audio``; the caller holds a ``_mgr.use()`` lease."""
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

    max_new_tokens = _max_new_tokens_for(req.text)

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
                fallback_speaker = (req.speaker or "").strip() or DEFAULT_SPEAKER or sorted(available)[0]
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
            speaker = (req.speaker or "").strip() or DEFAULT_SPEAKER
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

    import numpy as np

    audio = np.concatenate(audio_arrays) if len(audio_arrays) > 1 else audio_arrays[0]
    audio = np.ascontiguousarray(np.asarray(audio), dtype="<f4").reshape(-1)
    return audio, int(sample_rate)


@app.post("/synthesise")
async def synthesise(req: SynthesiseRequest):
    """Generate speech from text and store a WAV. Returns ``{audio_url, duration, ...}``."""
    t_req = time.monotonic()
    rendered = await _generate_audio(req)
    if isinstance(rendered, JSONResponse):
        return rendered
    audio, sample_rate = rendered

    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        await asyncio.to_thread(sf.write, tmp_path, audio, sample_rate)
        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()
        info = await asyncio.to_thread(sf.info, tmp_path)
        duration = info.duration
    finally:
        os.unlink(tmp_path)

    job_id = str(uuid.uuid4())
    s3_key = f"generated/voice/{job_id}.wav"
    audio_url = await asyncio.to_thread(_upload_to_s3, wav_bytes, s3_key)

    logger.info(
        "tts_synthesised",
        job_id=job_id,
        duration=round(duration, 2),
        language=req.language,
        total_s=round(time.monotonic() - t_req, 2),
        rtf=round((time.monotonic() - t_req) / duration, 1) if duration else None,
    )

    return {
        "audio_url": audio_url,
        "duration": duration,
        "duration_seconds": round(duration, 3),
        "format": "wav",
    }


@app.post("/synthesise/stream")
async def synthesise_stream(req: SynthesiseRequest):
    """Return the utterance as raw mono float32 PCM.

    Argus-mask ``POST /speak`` reads ``X-Sample-Rate``, ``X-Audio-Format`` and
    ``X-Channels`` and then the body. Qwen renders the whole clip before the
    first byte; mask buffers that by default.
    """
    rendered = await _generate_audio(req)
    if isinstance(rendered, JSONResponse):
        return rendered
    audio, sample_rate = rendered
    return Response(
        content=audio.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Audio-Format": "pcm_f32le",
            "X-Channels": "1",
            "Connection": "close",
        },
    )


# ── POST /clone-voice ─────────────────────────────────────────────────────────


_COMPANION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _clone_key(companion_id: str) -> str:
    """S3 key of a companion's clone prompt; ``companion_id`` must be validated."""
    return f"voice_clones/{companion_id}/prompt.pt"


def _s3_object_exists(key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        _get_s3_client().head_object(Bucket=S3_BUCKET_NAME, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise
    return True


# One lock per companion_id spans the existence check through the upload, so two
# concurrent replace=false clones of the same id can't both pass the check. The
# sidecar runs a single uvicorn worker, so an in-process lock is enough.
_clone_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = weakref.WeakValueDictionary()


def _clone_lock(companion_id: str) -> asyncio.Lock:
    lock = _clone_locks.get(companion_id)
    if lock is None:
        lock = asyncio.Lock()
        _clone_locks[companion_id] = lock
    return lock


@app.post("/clone-voice")
async def clone_voice(
    audio: UploadFile = File(...),
    reference_text: str = Form(""),
    companion_id: str = Form(...),
    replace: bool = Form(False),
):
    """Extract voice clone prompt from reference audio and store in S3.

    ``companion_id`` must match ``^[A-Za-z0-9_-]{1,64}$`` (422 otherwise). An
    existing clone is only overwritten when ``replace`` is true (409 otherwise).
    This sidecar has no caller identity; ownership is enforced by the gateway.
    """
    if not _COMPANION_ID_RE.fullmatch(companion_id):
        return JSONResponse(
            status_code=422,
            content={"error": "companion_id must be 1-64 characters of A-Z, a-z, 0-9, '_' or '-'"},
        )
    s3_key = _clone_key(companion_id)

    if _mgr.state == TTSState.LOADING:
        return JSONResponse(status_code=503, content={"error": "Models still loading"})

    lock = _clone_lock(companion_id)
    async with lock:
        if not replace and await asyncio.to_thread(_s3_object_exists, s3_key):
            return JSONResponse(
                status_code=409,
                content={"error": f"Voice clone {companion_id!r} already exists; send replace=true to overwrite"},
            )

        async with _mgr.use():
            return await _clone_voice_leased(audio, reference_text, companion_id, s3_key)


async def _clone_voice_leased(audio: UploadFile, reference_text: str, companion_id: str, s3_key: str):
    """Body of ``clone_voice``; the caller holds a ``_mgr.use()`` lease."""
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

    async with _mgr.use():
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
        "default_speaker": DEFAULT_SPEAKER or None,
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

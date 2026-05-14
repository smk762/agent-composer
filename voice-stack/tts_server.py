"""
Qwen3-TTS server — FastAPI, port 8031.

Voice-stack production version. Based on playact-engine's tts_server.py with
enhanced API surface for multiple consumers (playact-engine, rag-chat,
mithrandir, kimini-api).

Loads two model variants:
  - Base: voice cloning from reference audio
  - CustomVoice: predefined speakers + instruction-based emotion

Env vars:
  TTS_MODEL_SIZE   — "0.6b" (default) or "1.7b"
  TTS_DEVICE       — "cuda:0" (default) or "cpu"
  TTS_LOAD_MODELS  — "both" (default), "custom_voice", or "base"
  S3_ENDPOINT_URL  — MinIO/S3 endpoint
  S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY / S3_BUCKET_NAME / S3_PUBLIC_URL
"""

import asyncio
import io
import os
import tempfile
import threading
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
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
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "minioadmin")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "minioadmin")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "compance-media")
S3_PUBLIC_URL = os.getenv("S3_PUBLIC_URL", "http://localhost:9000/compance-media")

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

# ── Global state ─────────────────────────────────────────────────────────────

_base_model = None
_custom_model = None
_s3_client = None
_clone_cache: OrderedDict = OrderedDict()
_clone_cache_lock = threading.Lock()
_models_ready = False


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


def _load_models():
    global _base_model, _custom_model, _models_ready

    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
    ids = MODEL_IDS[MODEL_SIZE]

    if LOAD_MODELS in ("both", "base"):
        logger.info("loading_qwen3_tts_base", model=ids["base"], device=DEVICE)
        _base_model = Qwen3TTSModel.from_pretrained(ids["base"], device_map=DEVICE, dtype=dtype)
        logger.info("qwen3_tts_base_loaded")

    if LOAD_MODELS in ("both", "custom_voice"):
        logger.info("loading_qwen3_tts_custom_voice", model=ids["custom_voice"], device=DEVICE)
        _custom_model = Qwen3TTSModel.from_pretrained(ids["custom_voice"], device_map=DEVICE, dtype=dtype)
        logger.info("qwen3_tts_custom_voice_loaded")

    logger.info("tts_models_ready", load_mode=LOAD_MODELS)
    _models_ready = True


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
    prompt = torch.load(buf, map_location=DEVICE, weights_only=True)

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
    await asyncio.to_thread(_load_models)
    yield


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


@app.post("/synthesise")
async def synthesise(req: SynthesiseRequest):
    """Generate speech from text.

    Uses Base model if voice_clone_id is provided (voice cloning path).
    Uses CustomVoice model otherwise (predefined speaker + instruction path).
    """
    if not _models_ready:
        return JSONResponse(status_code=503, content={"error": "Models still loading"})

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

    try:
        if req.voice_clone_id:
            if _base_model is None:
                return JSONResponse(status_code=503, content={"error": "Base model not loaded (TTS_LOAD_MODELS=custom_voice)"})
            clone_prompt = await asyncio.to_thread(_get_clone_prompt, req.voice_clone_id)
            audio_arrays, sample_rate = await asyncio.to_thread(
                _base_model.generate_voice_clone,
                req.text, voice_clone_prompt=clone_prompt, language=language,
            )

        elif req.voice_description:
            if _custom_model is None:
                return JSONResponse(status_code=503, content={"error": "CustomVoice model not loaded (TTS_LOAD_MODELS=base)"})
            design_instruct = req.voice_description
            if instruction:
                design_instruct = f"{req.voice_description}. {instruction}"
            audio_arrays, sample_rate = await asyncio.to_thread(
                _custom_model.generate_voice_design,
                req.text, instruct=design_instruct, language=language,
            )

        else:
            if _custom_model is None:
                return JSONResponse(status_code=503, content={"error": "CustomVoice model not loaded (TTS_LOAD_MODELS=base)"})
            speaker = req.speaker
            if not speaker:
                available = _custom_model.get_supported_speakers()
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": "Either 'speaker', 'voice_description', or 'voice_clone_id' is required.",
                        "available_speakers": list(available),
                    },
                )
            generate_kwargs: dict = {"text": req.text, "speaker": speaker, "language": language}
            if instruction:
                generate_kwargs["instruct"] = instruction
            audio_arrays, sample_rate = await asyncio.to_thread(
                _custom_model.generate_custom_voice, **generate_kwargs,
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

    s3_key = f"generated/voice/{job_id}.wav"
    audio_url = await asyncio.to_thread(_upload_to_s3, wav_bytes, s3_key)

    logger.info("tts_synthesised", job_id=job_id, duration=duration, language=req.language)

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
    if not _models_ready:
        return JSONResponse(status_code=503, content={"error": "Models still loading"})
    if _base_model is None:
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
            _base_model.create_voice_clone_prompt,
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
    if not _models_ready:
        return JSONResponse(status_code=503, content={"error": "Models still loading"})
    if _custom_model is None:
        return JSONResponse(status_code=503, content={"error": "CustomVoice model not loaded"})

    raw_speakers = await asyncio.to_thread(_custom_model.get_supported_speakers)
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
    if not _models_ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {
        "status": "ok",
        "engine": "qwen-tts",
        "model": MODEL_SIZE,
        "model_size": MODEL_SIZE,
        "device": DEVICE,
        "load_models": LOAD_MODELS,
    }

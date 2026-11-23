"""OpenAI provider — DALL-E 3 / GPT-Image-1 (images) and Sora (video)."""

import asyncio
import base64
import logging
from typing import List, Optional
from uuid import uuid4

import httpx

from app.providers.base import (
    Capability, GeneratedMedia, GenerationResult, ImageProvider, VideoProvider,
)
from app.providers.registry import register_provider

log = logging.getLogger("rag-chat.provider.openai")

_API_BASE = "https://api.openai.com/v1"

_IMAGE_MODELS = ["gpt-image-1", "dall-e-3", "dall-e-2"]
_VIDEO_MODELS = ["sora-2", "sora-2-pro"]
# The /videos job API takes ``seconds`` (a string: 4, 8 or 12) and ``size``, not duration/aspect_ratio.
_VIDEO_SECONDS = (4, 8, 12)
_VIDEO_ASPECTS = {"16:9": "1280x720", "9:16": "720x1280"}
_VIDEO_SIZES = ("1280x720", "720x1280", "1792x1024", "1024x1792")
_VIDEO_POLL_INTERVAL = 5
_VIDEO_MAX_POLLS = 120  # 10 min at 5s


class OpenAIProvider(ImageProvider, VideoProvider):
    name = "openai"
    capabilities = [
        Capability.IMAGE_GENERATE,
        Capability.IMAGE_EDIT,
        Capability.VIDEO_GENERATE,
    ]

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key or ""

    async def validate_config(self) -> bool:
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get(
                    f"{_API_BASE}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            return r.status_code == 200
        except httpx.HTTPError as exc:  # unreachable or timed out, not a rejected key: say which in the log
            log.warning("OpenAI config check failed: %s: %s", type(exc).__name__, exc)
            return False

    def available_models(self, capability: Capability) -> List[str]:
        if capability in (Capability.IMAGE_GENERATE, Capability.IMAGE_EDIT):
            return list(_IMAGE_MODELS)
        if capability == Capability.VIDEO_GENERATE:
            return list(_VIDEO_MODELS)
        return []

    # ---- Image generation ----

    async def generate_image(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        model: Optional[str] = None,
        size: str = "1024x1024",
        n: int = 1,
        style: Optional[str] = None,
    ) -> GenerationResult:
        model = model or "gpt-image-1"
        if n < 1:
            raise LookupError(f"OpenAI image generation needs n >= 1, got n={n}")
        if n > 10:
            # The API caps a request at 10 images; refuse rather than silently return fewer.
            raise LookupError(f"OpenAI image generation makes at most 10 images per request, got n={n}")
        if model == "dall-e-3" and n > 1:
            # The API answers n>1 for dall-e-3 with a 400; say so before sending.
            raise LookupError(f"dall-e-3 generates one image per request, got n={n}")
        body: dict = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
        }
        if style and model == "dall-e-3":
            # Only dall-e-3 takes `style`; dall-e-2 and gpt-image-1 answer it with a 400.
            body["style"] = style

        if model == "gpt-image-1":
            body["output_format"] = "png"
            return await self._generate_image_gpt(body)

        body["response_format"] = "b64_json"
        return await self._generate_image_dalle(body, model)

    async def _generate_image_gpt(self, body: dict) -> GenerationResult:
        """GPT-Image-1 returns base64 images directly."""
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(
                f"{_API_BASE}/images/generations",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
        r.raise_for_status()
        data = r.json()
        media = []
        for i, item in enumerate(data.get("data", [])):
            img_bytes = _b64_image(item)
            media.append(GeneratedMedia(
                data=img_bytes,
                mime_type="image/png",
                filename=f"image_{i}_{uuid4().hex[:8]}.png",
                width=_parse_dim(body.get("size", ""), 0),
                height=_parse_dim(body.get("size", ""), 1),
            ))
        return GenerationResult(media=media, model=body.get("model", "gpt-image-1"), provider=self.name)

    async def _generate_image_dalle(self, body: dict, model: str) -> GenerationResult:
        """DALL-E 2/3 path."""
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(
                f"{_API_BASE}/images/generations",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
        r.raise_for_status()
        data = r.json()
        media = []
        for i, item in enumerate(data.get("data", [])):
            img_bytes = _b64_image(item)
            media.append(GeneratedMedia(
                data=img_bytes,
                mime_type="image/png",
                filename=f"image_{i}_{uuid4().hex[:8]}.png",
                width=_parse_dim(body.get("size", ""), 0),
                height=_parse_dim(body.get("size", ""), 1),
            ))
        return GenerationResult(media=media, model=model, provider=self.name)

    # ---- Image editing ----

    async def edit_image(
        self,
        prompt: str,
        image: bytes,
        *,
        mask: Optional[bytes] = None,
        model: Optional[str] = None,
        size: Optional[str] = None,
    ) -> GenerationResult:
        model = model or "gpt-image-1"

        files: dict = {
            "image": ("input.png", image, "image/png"),
            "prompt": (None, prompt),
            "model": (None, model),
        }
        if mask:
            files["mask"] = ("mask.png", mask, "image/png")
        if size:
            files["size"] = (None, size)
        if not model.startswith("gpt-image"):
            # DALL-E answers with URLs unless asked for base64; gpt-image-1 always sends base64.
            files["response_format"] = (None, "b64_json")

        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(
                f"{_API_BASE}/images/edits",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
            )
        r.raise_for_status()
        data = r.json()
        media = []
        for i, item in enumerate(data.get("data", [])):
            img_bytes = _b64_image(item)
            media.append(GeneratedMedia(
                data=img_bytes,
                mime_type="image/png",
                filename=f"edit_{i}_{uuid4().hex[:8]}.png",
                width=_parse_dim(size or "", 0),
                height=_parse_dim(size or "", 1),
            ))
        return GenerationResult(media=media, model=model, provider=self.name)

    # ---- Video generation ----

    async def generate_video(
        self,
        prompt: str,
        *,
        image: Optional[bytes] = None,
        model: Optional[str] = None,
        duration: Optional[float] = None,
        aspect_ratio: Optional[str] = None,
    ) -> GenerationResult:
        """Sora is a job API: create ``POST /videos``, poll ``GET /videos/{id}`` until it is
        ``completed`` (or ``failed``), then download ``GET /videos/{id}/content``."""
        if image is not None:
            # This call sends no reference image; refuse rather than bill a text-only video.
            raise LookupError("OpenAI video generation here does not take an input image; send the prompt alone")
        model = model or _VIDEO_MODELS[0]
        if model not in _VIDEO_MODELS:
            raise LookupError(f"Unknown OpenAI video model {model!r}: use one of {_VIDEO_MODELS}")
        form: dict = {"model": (None, model), "prompt": (None, prompt)}
        seconds = None
        if duration:
            seconds = int(duration)
            if seconds != duration or seconds not in _VIDEO_SECONDS:
                raise LookupError(f"OpenAI video duration must be one of {list(_VIDEO_SECONDS)} seconds, got {duration}")
            form["seconds"] = (None, str(seconds))
        if aspect_ratio:
            size = _VIDEO_ASPECTS.get(aspect_ratio, aspect_ratio if aspect_ratio in _VIDEO_SIZES else None)
            if size is None:
                raise LookupError(
                    f"OpenAI video aspect ratio must be one of {sorted(_VIDEO_ASPECTS)} "
                    f"or a size {list(_VIDEO_SIZES)}, got {aspect_ratio!r}"
                )
            form["size"] = (None, size)

        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{_API_BASE}/videos", headers=headers, files=form)
            r.raise_for_status()
            job = r.json()
            video_id = job.get("id")
            if not video_id:
                raise RuntimeError("OpenAI video job was created without an id")
            polls = 0
            while True:
                # Checked after every fetch, the last poll's included: a job that finishes (or
                # fails) on the final poll is reported as such, not as a timeout.
                status = job.get("status")
                if status == "completed":
                    break
                if status == "failed":
                    raise RuntimeError(f"OpenAI video job {video_id} failed: {job.get('error')}")
                if polls >= _VIDEO_MAX_POLLS:
                    raise TimeoutError(f"OpenAI video job {video_id} did not finish in time")
                polls += 1
                await asyncio.sleep(_VIDEO_POLL_INTERVAL)
                r = await c.get(f"{_API_BASE}/videos/{video_id}", headers=headers)
                r.raise_for_status()
                job = r.json()

            vr = await c.get(f"{_API_BASE}/videos/{video_id}/content", headers=headers, timeout=300)
            vr.raise_for_status()
        if not vr.content:
            raise RuntimeError(f"OpenAI video job {video_id} completed but returned no content")
        job_seconds = job.get("seconds")
        try:
            duration_seconds = float(job_seconds) if job_seconds is not None else seconds
        except (TypeError, ValueError):
            duration_seconds = seconds
        media = [GeneratedMedia(
            data=vr.content,
            mime_type="video/mp4",
            filename=f"video_0_{uuid4().hex[:8]}.mp4",
            width=_parse_dim(str(job.get("size") or ""), 0),
            height=_parse_dim(str(job.get("size") or ""), 1),
            duration_seconds=duration_seconds,
        )]
        return GenerationResult(media=media, model=model, provider=self.name)


def _b64_image(item: dict) -> bytes:
    """An image item's bytes; an item without ``b64_json`` (e.g. a ``url`` item) is an error,
    never an empty "PNG" that looks like a success."""
    b64 = item.get("b64_json")
    if not b64:
        raise RuntimeError("OpenAI returned an image without b64_json data")
    return base64.b64decode(b64)


def _parse_dim(size: str, idx: int) -> Optional[int]:
    parts = size.lower().split("x")
    if len(parts) == 2:
        try:
            return int(parts[idx])
        except ValueError:
            pass
    return None


register_provider("openai", OpenAIProvider)

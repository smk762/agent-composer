"""OpenAI provider — DALL-E 3 / GPT-Image-1 (images) and Sora (video)."""

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
_VIDEO_MODELS = ["sora"]


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
        except Exception:
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
        body: dict = {
            "model": model,
            "prompt": prompt,
            "n": min(n, 10),
            "size": size,
        }
        if style:
            body["style"] = style

        if model == "gpt-image-1":
            body["output_format"] = "png"
            body.pop("style", None)
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
            b64 = item.get("b64_json", "")
            img_bytes = base64.b64decode(b64)
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
            b64 = item.get("b64_json", "")
            img_bytes = base64.b64decode(b64)
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
            b64 = item.get("b64_json", "")
            img_bytes = base64.b64decode(b64)
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
        model = model or "sora"
        body: dict = {
            "model": model,
            "prompt": prompt,
        }
        if duration:
            body["duration"] = int(duration)
        if aspect_ratio:
            body["aspect_ratio"] = aspect_ratio

        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.post(
                f"{_API_BASE}/videos/generations",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
        r.raise_for_status()
        data = r.json()

        media = []
        for i, item in enumerate(data.get("data", [])):
            url = item.get("url", "")
            if url:
                async with httpx.AsyncClient(timeout=120) as dl:
                    vr = await dl.get(url)
                    vr.raise_for_status()
                media.append(GeneratedMedia(
                    data=vr.content,
                    mime_type="video/mp4",
                    filename=f"video_{i}_{uuid4().hex[:8]}.mp4",
                    duration_seconds=duration,
                ))
        return GenerationResult(media=media, model=model, provider=self.name)


def _parse_dim(size: str, idx: int) -> Optional[int]:
    parts = size.lower().split("x")
    if len(parts) == 2:
        try:
            return int(parts[idx])
        except ValueError:
            pass
    return None


register_provider("openai", OpenAIProvider)

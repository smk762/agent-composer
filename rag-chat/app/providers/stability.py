"""Stability AI provider — SDXL / SD3 image generation and editing."""

import logging
from typing import List, Optional
from uuid import uuid4

import httpx

from app.providers.base import (
    Capability, GeneratedMedia, GenerationResult, ImageProvider,
)
from app.providers.registry import register_provider

log = logging.getLogger("rag-chat.provider.stability")

_API_BASE = "https://api.stability.ai/v2beta"

_IMAGE_MODELS = ["sd3-large", "sd3-large-turbo", "sd3-medium", "core"]


class StabilityAIProvider(ImageProvider):
    name = "stability"
    capabilities = [
        Capability.IMAGE_GENERATE,
        Capability.IMAGE_EDIT,
        Capability.IMAGE_INPAINT,
    ]

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key or ""

    async def validate_config(self) -> bool:
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get(
                    "https://api.stability.ai/v1/user/account",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            return r.status_code == 200
        except Exception:
            return False

    def available_models(self, capability: Capability) -> List[str]:
        if capability in (Capability.IMAGE_GENERATE, Capability.IMAGE_EDIT, Capability.IMAGE_INPAINT):
            return list(_IMAGE_MODELS)
        return []

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
        model = model or "sd3-large"
        parts = size.split("x")
        w = int(parts[0]) if len(parts) >= 1 else 1024
        h = int(parts[1]) if len(parts) >= 2 else w
        aspect = _closest_aspect(w, h)

        endpoint = f"{_API_BASE}/stable-image/generate/{model}" if model != "core" else f"{_API_BASE}/stable-image/generate/core"

        all_media: list[GeneratedMedia] = []
        for _ in range(n):
            form: dict = {
                "prompt": (None, prompt),
                "output_format": (None, "png"),
                "aspect_ratio": (None, aspect),
            }
            if negative_prompt:
                form["negative_prompt"] = (None, negative_prompt)

            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Accept": "image/*",
                    },
                    files=form,
                )
            r.raise_for_status()
            all_media.append(GeneratedMedia(
                data=r.content,
                mime_type="image/png",
                filename=f"stability_{uuid4().hex[:8]}.png",
                width=w, height=h,
            ))

        return GenerationResult(media=all_media, model=model, provider=self.name)

    async def edit_image(
        self,
        prompt: str,
        image: bytes,
        *,
        mask: Optional[bytes] = None,
        model: Optional[str] = None,
        size: Optional[str] = None,
    ) -> GenerationResult:
        endpoint = f"{_API_BASE}/stable-image/edit/inpaint" if mask else f"{_API_BASE}/stable-image/edit/search-and-replace"

        form: dict = {
            "image": ("input.png", image, "image/png"),
            "prompt": (None, prompt),
            "output_format": (None, "png"),
        }
        if mask:
            form["mask"] = ("mask.png", mask, "image/png")

        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "image/*",
                },
                files=form,
            )
        r.raise_for_status()
        return GenerationResult(
            media=[GeneratedMedia(
                data=r.content,
                mime_type="image/png",
                filename=f"stability_edit_{uuid4().hex[:8]}.png",
            )],
            model=model or "core",
            provider=self.name,
        )


def _closest_aspect(w: int, h: int) -> str:
    """Map WxH to one of Stability's supported aspect ratios."""
    ratio = w / h if h else 1.0
    aspects = [
        (1.0, "1:1"), (16 / 9, "16:9"), (9 / 16, "9:16"),
        (21 / 9, "21:9"), (9 / 21, "9:21"),
        (4 / 3, "4:3"), (3 / 4, "3:4"),
        (3 / 2, "3:2"), (2 / 3, "2:3"),
        (5 / 4, "5:4"), (4 / 5, "4:5"),
    ]
    return min(aspects, key=lambda x: abs(x[0] - ratio))[1]


register_provider("stability", StabilityAIProvider)

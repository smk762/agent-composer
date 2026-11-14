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

_IMAGE_MODELS = ["sd3-large", "sd3-large-turbo", "sd3-medium", "core", "ultra"]
_SD3_MODELS = {m for m in _IMAGE_MODELS if m.startswith("sd3")}
_DEFAULT_SD3 = "sd3-large"
# Mask-less edits run SD3 image-to-image; how far the result may move from the input.
_EDIT_STRENGTH = "0.7"


def _generate_target(model: str) -> tuple[str, dict]:
    """Endpoint and extra form fields for a generation model.

    v2beta serves every SD3 variant from ``/generate/sd3`` with the variant in a
    ``model`` field; ``/generate/sd3-large`` does not exist.
    """
    if model in _SD3_MODELS:
        return f"{_API_BASE}/stable-image/generate/sd3", {"model": (None, model)}
    if model in ("core", "ultra"):
        return f"{_API_BASE}/stable-image/generate/{model}", {}
    # LookupError: the generation router answers it with a 400.
    raise LookupError(f"Unknown Stability model {model!r}; expected one of {', '.join(_IMAGE_MODELS)}")


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
        model = model or _DEFAULT_SD3
        endpoint, extra = _generate_target(model)
        parts = size.split("x")
        w = int(parts[0]) if len(parts) >= 1 else 1024
        h = int(parts[1]) if len(parts) >= 2 else w
        aspect = _closest_aspect(w, h)

        all_media: list[GeneratedMedia] = []
        for _ in range(n):
            form: dict = {
                "prompt": (None, prompt),
                "output_format": (None, "png"),
                "aspect_ratio": (None, aspect),
                **extra,
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
        form: dict = {
            "image": ("input.png", image, "image/png"),
            "prompt": (None, prompt),
            "output_format": (None, "png"),
        }
        if mask:
            endpoint = f"{_API_BASE}/stable-image/edit/inpaint"
            form["mask"] = ("mask.png", mask, "image/png")
            model = model or "core"
        else:
            # search-and-replace needs a search_prompt this interface doesn't carry,
            # so a mask-less edit is SD3 image-to-image (core/ultra have no such mode).
            model = model if model in _SD3_MODELS else _DEFAULT_SD3
            endpoint, extra = _generate_target(model)
            form.update(extra)
            form["mode"] = (None, "image-to-image")
            form["strength"] = (None, _EDIT_STRENGTH)

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
            model=model,
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

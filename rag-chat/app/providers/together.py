"""Together AI provider — Flux image generation."""

import base64
import logging
from typing import List, Optional
from uuid import uuid4

import httpx

from app.providers.base import (
    Capability, GeneratedMedia, GenerationResult, ImageProvider,
)
from app.providers.registry import register_provider

log = logging.getLogger("rag-chat.provider.together")

_API_BASE = "https://api.together.xyz/v1"

_IMAGE_MODELS = ["black-forest-labs/FLUX.1-schnell-Free", "black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1.1-pro"]


class TogetherAIProvider(ImageProvider):
    name = "together"
    capabilities = [Capability.IMAGE_GENERATE]

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
        if capability == Capability.IMAGE_GENERATE:
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
        model = model or "black-forest-labs/FLUX.1-schnell-Free"
        parts = size.split("x")
        w = int(parts[0]) if len(parts) >= 1 else 1024
        h = int(parts[1]) if len(parts) >= 2 else w

        body = {
            "model": model,
            "prompt": prompt,
            "width": w,
            "height": h,
            "n": min(n, 4),
            "response_format": "b64_json",
            "steps": 4 if "schnell" in model.lower() else 28,
        }
        if negative_prompt:
            body["negative_prompt"] = negative_prompt

        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(
                f"{_API_BASE}/images/generations",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
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
                filename=f"together_{i}_{uuid4().hex[:8]}.png",
                width=w, height=h,
            ))

        return GenerationResult(media=media, model=model, provider=self.name)


register_provider("together", TogetherAIProvider)

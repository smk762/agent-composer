"""Replicate provider — cloud-hosted image & video models (Flux, SD, etc.).

Uses the Replicate HTTP API (predictions endpoint) with polling.
"""

import asyncio
import logging
import re
from typing import List, Optional
from uuid import uuid4

import httpx

from app.providers.base import (
    Capability, GeneratedMedia, GenerationResult, ImageProvider, VideoProvider, parse_size,
)
from app.providers.registry import register_provider

log = logging.getLogger("rag-chat.provider.replicate")

_API_BASE = "https://api.replicate.com/v1"
_POLL_INTERVAL = 3
_MAX_POLLS = 100
_MAX_IMAGES = 4

_IMAGE_MODELS = {
    "flux-1.1-pro": "black-forest-labs/flux-1.1-pro",
    "flux-schnell": "black-forest-labs/flux-schnell",
    "sdxl": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
}

_VIDEO_MODELS = {
    "minimax-video": "minimax/video-01",
    "luma-ray2": "luma/ray",
}

# A model named outright: ``owner/name`` or ``owner/name:<64-hex version>``. Anything else
# (extra path segments, query characters) never reaches the authenticated API URL.
_MODEL_REF_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,99}/[a-z0-9][a-z0-9_.-]{0,99}(:[0-9a-f]{64})?", re.IGNORECASE)


def _resolve_model(model_key: str, known: dict) -> str:
    """A known alias, or a well-formed ``owner/name[:version]``; LookupError (400) otherwise."""
    if model_key in known:
        return known[model_key]
    if _MODEL_REF_RE.fullmatch(model_key):
        return model_key
    raise LookupError(
        f"Unknown Replicate model {model_key!r}: use one of {sorted(known)} or owner/name[:version]"
    )


class ReplicateProvider(ImageProvider, VideoProvider):
    name = "replicate"
    capabilities = [
        Capability.IMAGE_GENERATE,
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
            log.warning("Replicate config check failed: %s: %s", type(exc).__name__, exc)
            return False

    def available_models(self, capability: Capability) -> List[str]:
        if capability == Capability.IMAGE_GENERATE:
            return list(_IMAGE_MODELS.keys())
        if capability == Capability.VIDEO_GENERATE:
            return list(_VIDEO_MODELS.keys())
        return []

    async def _create_prediction(self, model_version: str, input_data: dict) -> str:
        """Create a prediction and return its ID."""
        if ":" in model_version:
            body = {"version": model_version.split(":")[-1], "input": input_data}
            url = f"{_API_BASE}/predictions"
        else:
            body = {"input": input_data}
            url = f"{_API_BASE}/models/{model_version}/predictions"

        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Prefer": "respond-async",
                },
                json=body,
            )
        r.raise_for_status()
        return r.json()["id"]

    async def _cancel_prediction(self, pred_id: str) -> None:
        """Best-effort: stop a prediction we gave up on so it doesn't keep billing compute."""
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                await c.post(
                    f"{_API_BASE}/predictions/{pred_id}/cancel",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
        except Exception as exc:
            log.warning("Could not cancel Replicate prediction %s: %s", pred_id, exc)

    async def _poll_prediction(self, pred_id: str) -> dict:
        try:
            return await self._poll_until_done(pred_id)
        except (TimeoutError, asyncio.CancelledError):
            await self._cancel_prediction(pred_id)
            raise

    async def _poll_until_done(self, pred_id: str) -> dict:
        for _ in range(_MAX_POLLS):
            await asyncio.sleep(_POLL_INTERVAL)
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get(
                    f"{_API_BASE}/predictions/{pred_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            r.raise_for_status()
            data = r.json()
            status = data.get("status")
            if status == "succeeded":
                return data
            if status in ("failed", "canceled"):
                raise RuntimeError(f"Replicate prediction {status}: {data.get('error')}")
        raise TimeoutError("Replicate prediction timed out")

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
        model_key = model or "flux-schnell"
        version = _resolve_model(model_key, _IMAGE_MODELS)

        w, h = parse_size(size)
        if not 1 <= n <= _MAX_IMAGES:
            raise LookupError(f"Replicate generates 1-{_MAX_IMAGES} images per request, got n={n}")

        input_data = {"prompt": prompt, "width": w, "height": h, "num_outputs": n}
        if negative_prompt:
            input_data["negative_prompt"] = negative_prompt

        pred_id = await self._create_prediction(version, input_data)
        result = await self._poll_prediction(pred_id)

        output = result.get("output")
        if isinstance(output, str):
            output = [output]

        media = []
        for i, url in enumerate(output or []):
            async with httpx.AsyncClient(timeout=120) as dl:
                r = await dl.get(url)
                r.raise_for_status()
            ct = r.headers.get("content-type", "image/png")
            ext = ".webp" if "webp" in ct else ".png" if "png" in ct else ".jpg"
            media.append(GeneratedMedia(
                data=r.content,
                mime_type=ct.split(";")[0],
                filename=f"replicate_{i}_{uuid4().hex[:8]}{ext}",
                width=w, height=h,
            ))

        return GenerationResult(media=media, model=model_key, provider=self.name)

    async def generate_video(
        self,
        prompt: str,
        *,
        image: Optional[bytes] = None,
        model: Optional[str] = None,
        duration: Optional[float] = None,
        aspect_ratio: Optional[str] = None,
    ) -> GenerationResult:
        if image is not None:
            # No input-image mapping per model yet; refuse rather than bill a text-only video.
            raise LookupError("Replicate video generation here does not take an input image; send the prompt alone")
        model_key = model or "minimax-video"
        version = _resolve_model(model_key, _VIDEO_MODELS)

        input_data: dict = {"prompt": prompt}
        if aspect_ratio:
            input_data["aspect_ratio"] = aspect_ratio

        pred_id = await self._create_prediction(version, input_data)
        result = await self._poll_prediction(pred_id)

        output = result.get("output")
        if isinstance(output, str):
            output = [output]

        media = []
        for i, url in enumerate(output or []):
            async with httpx.AsyncClient(timeout=120) as dl:
                r = await dl.get(url)
                r.raise_for_status()
            media.append(GeneratedMedia(
                data=r.content,
                mime_type="video/mp4",
                filename=f"replicate_vid_{i}_{uuid4().hex[:8]}.mp4",
                # The requested duration isn't sent (the models' inputs differ), so the clip's
                # length is unknown here; don't report the request as if it were the output.
                duration_seconds=None,
            ))

        return GenerationResult(media=media, model=model_key, provider=self.name)


register_provider("replicate", ReplicateProvider)

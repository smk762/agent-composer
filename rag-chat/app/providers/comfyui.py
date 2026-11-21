"""ComfyUI provider — local image/video generation via ComfyUI's queue API.

ComfyUI exposes a REST + WebSocket API.  We POST a workflow JSON to
``/prompt``, then poll ``/history/{prompt_id}`` until the outputs appear.
"""

import asyncio
import json
import logging
import os
import secrets
from typing import List, Optional
from uuid import uuid4

import httpx

from app.providers.base import (
    Capability, GeneratedMedia, GenerationResult, ImageProvider, VideoProvider,
)
from app.providers.registry import register_provider

log = logging.getLogger("rag-chat.provider.comfyui")

_COMFYUI_URL = os.getenv("COMFYUI_URL", "http://comfyui:8188")
_POLL_INTERVAL = 2
_MAX_POLLS = 150  # 5 min at 2s interval
# Bounds on one request against the shared ComfyUI worker (each image is a 30-step job).
_MIN_EDGE, _MAX_EDGE = 256, 2048
_MAX_IMAGES = 4


class _WorkflowFailed(RuntimeError):
    """ComfyUI finished the prompt with an error (it no longer holds the queue)."""


def _parse_size(size: str) -> tuple[int, int]:
    """``WxH`` (or ``N`` for a square) within the supported range, multiples of 8.
    Anything else is a LookupError, which the router answers with a 400."""
    parts = (size or "").lower().split("x")
    try:
        w = int(parts[0])
        h = int(parts[1]) if len(parts) == 2 else w
    except (ValueError, IndexError):
        raise LookupError(f"ComfyUI size must be WIDTHxHEIGHT, got {size!r}") from None
    if len(parts) > 2 or not all(_MIN_EDGE <= d <= _MAX_EDGE and d % 8 == 0 for d in (w, h)):
        raise LookupError(
            f"ComfyUI size {size!r} is out of range: each side {_MIN_EDGE}-{_MAX_EDGE}, a multiple of 8"
        )
    return w, h


class ComfyUIProvider(ImageProvider, VideoProvider):
    name = "comfyui"
    # Only txt2img has a workflow. Edit and video aren't advertised (and list no models) until
    # they do, so capability routing never picks ComfyUI for a request it can only fail.
    capabilities = [
        Capability.IMAGE_GENERATE,
    ]

    def __init__(self, api_key: Optional[str] = None, comfyui_url: Optional[str] = None, **kwargs):
        self.base_url = comfyui_url or _COMFYUI_URL

    async def validate_config(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(f"{self.base_url}/system_stats")
            return r.status_code == 200
        except Exception:
            return False

    def available_models(self, capability: Capability) -> List[str]:
        if capability == Capability.IMAGE_GENERATE:
            return ["stable-diffusion-xl", "flux-dev", "flux-schnell"]
        return []

    async def _queue_prompt(self, workflow: dict) -> str:
        """Queue a workflow and return the prompt_id."""
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow, "client_id": uuid4().hex},
            )
        r.raise_for_status()
        return r.json()["prompt_id"]

    async def _poll_result(self, prompt_id: str) -> dict:
        """Poll /history until the prompt completes."""
        for _ in range(_MAX_POLLS):
            await asyncio.sleep(_POLL_INTERVAL)
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get(f"{self.base_url}/history/{prompt_id}")
            if r.status_code != 200:
                continue
            data = r.json()
            entry = data.get(prompt_id)
            if entry and entry.get("status", {}).get("completed", False):
                return entry
            if entry and entry.get("status", {}).get("status_str") == "error":
                raise _WorkflowFailed(f"ComfyUI workflow failed: {entry}")
        raise TimeoutError("ComfyUI prompt timed out")

    async def _cancel_prompt(self, prompt_id: str) -> None:
        """Best-effort: drop a prompt we gave up on so it stops holding the shared GPU queue.

        Deletes it if still pending; interrupts it only if it is the one running (a bare
        /interrupt would stop whatever job is running, possibly someone else's)."""
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                q = await c.get(f"{self.base_url}/queue")
                running = q.json().get("queue_running", []) if q.status_code == 200 else []
                # queue entries are [number, prompt_id, prompt, extra, outputs]
                if any(len(item) > 1 and item[1] == prompt_id for item in running):
                    await c.post(f"{self.base_url}/interrupt", json={"prompt_id": prompt_id})
                await c.post(f"{self.base_url}/queue", json={"delete": [prompt_id]})
        except Exception as exc:
            log.warning("Could not cancel ComfyUI prompt %s: %s", prompt_id, exc)

    async def _download_output(self, filename: str, subfolder: str = "", output_type: str = "output") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": output_type}
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.get(f"{self.base_url}/view", params=params)
        r.raise_for_status()
        return r.content

    def _build_txt2img_workflow(
        self, prompt: str, negative: str, width: int, height: int, model: str, seed: Optional[int] = None,
    ) -> dict:
        """Build a minimal SDXL/Flux txt2img workflow graph.

        ComfyUI's KSampler rejects a negative seed (-1 is an A1111 convention, not
        "random" here), so each call gets a fresh non-negative one unless given.
        """
        if seed is None:
            seed = secrets.randbits(63)
        ckpt = _CHECKPOINT_MAP.get(model, model)
        return {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
            "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                    "latent_image": ["4", 0], "seed": seed, "steps": 30,
                    "cfg": 7.5, "sampler_name": "euler_ancestral", "scheduler": "normal",
                    "denoise": 1.0,
                },
            },
            "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "gen"}},
        }

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
        w, h = _parse_size(size)
        if not 1 <= n <= _MAX_IMAGES:
            raise LookupError(f"ComfyUI generates 1-{_MAX_IMAGES} images per request, got n={n}")
        model = model or "stable-diffusion-xl"
        negative = negative_prompt or ""

        all_media: list[GeneratedMedia] = []
        for _ in range(n):
            workflow = self._build_txt2img_workflow(prompt, negative, w, h, model)
            prompt_id = await self._queue_prompt(workflow)
            try:
                result = await self._poll_result(prompt_id)
            except _WorkflowFailed:
                raise  # the prompt already finished (with an error); nothing to cancel
            except BaseException:
                # Timeout, cancellation, a network blip or a bad /history body: drop the
                # prompt so it doesn't keep the shared GPU queue busy for nobody.
                await self._cancel_prompt(prompt_id)
                raise
            outputs = result.get("outputs", {})
            for node_id, node_out in outputs.items():
                for img in node_out.get("images", []):
                    data = await self._download_output(
                        img["filename"], img.get("subfolder", ""), img.get("type", "output"),
                    )
                    all_media.append(GeneratedMedia(
                        data=data,
                        mime_type="image/png",
                        filename=img["filename"],
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
        raise NotImplementedError(
            "ComfyUI image editing requires a custom inpainting workflow. "
            "Upload a workflow JSON via the API to use this feature."
        )

    async def generate_video(
        self,
        prompt: str,
        *,
        image: Optional[bytes] = None,
        model: Optional[str] = None,
        duration: Optional[float] = None,
        aspect_ratio: Optional[str] = None,
    ) -> GenerationResult:
        raise NotImplementedError(
            "ComfyUI video generation requires a custom AnimateDiff workflow. "
            "Upload a workflow JSON via the API to use this feature."
        )


_CHECKPOINT_MAP = {
    "stable-diffusion-xl": "sd_xl_base_1.0.safetensors",
    "flux-dev": "flux1-dev.safetensors",
    "flux-schnell": "flux1-schnell.safetensors",
}


register_provider("comfyui", ComfyUIProvider)

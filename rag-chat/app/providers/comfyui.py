"""ComfyUI provider — local image/video generation via ComfyUI's queue API.

ComfyUI exposes a REST + WebSocket API.  We POST a workflow JSON to
``/prompt``, then poll ``/history/{prompt_id}`` until the outputs appear.
"""

import asyncio
import json
import logging
import os
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


class ComfyUIProvider(ImageProvider, VideoProvider):
    name = "comfyui"
    capabilities = [
        Capability.IMAGE_GENERATE,
        Capability.IMAGE_EDIT,
        Capability.VIDEO_GENERATE,
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
        if capability in (Capability.IMAGE_GENERATE, Capability.IMAGE_EDIT):
            return ["stable-diffusion-xl", "flux-dev", "flux-schnell"]
        if capability == Capability.VIDEO_GENERATE:
            return ["animatediff"]
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
                raise RuntimeError(f"ComfyUI workflow failed: {entry}")
        raise TimeoutError("ComfyUI prompt timed out")

    async def _download_output(self, filename: str, subfolder: str = "", output_type: str = "output") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": output_type}
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.get(f"{self.base_url}/view", params=params)
        r.raise_for_status()
        return r.content

    def _build_txt2img_workflow(self, prompt: str, negative: str, width: int, height: int, model: str) -> dict:
        """Build a minimal SDXL/Flux txt2img workflow graph."""
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
                    "latent_image": ["4", 0], "seed": -1, "steps": 30,
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
        parts = size.split("x")
        w = int(parts[0]) if len(parts) >= 1 else 1024
        h = int(parts[1]) if len(parts) >= 2 else w
        model = model or "stable-diffusion-xl"
        negative = negative_prompt or ""

        all_media: list[GeneratedMedia] = []
        for _ in range(n):
            workflow = self._build_txt2img_workflow(prompt, negative, w, h, model)
            prompt_id = await self._queue_prompt(workflow)
            result = await self._poll_result(prompt_id)
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

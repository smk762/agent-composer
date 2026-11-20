"""Runtime model control (/api/runtime): status, wake and evict for Ollama models and the Infinity idle manager,
plus the GPU-resident gateway argus-iris uses to read and unload the sidecars on this host."""

import asyncio
import time
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.auth import require_user
from app.upstream_errors import upstream_error
from app.config import (
    ARGUS_ADMIN_TOKEN,
    CHAT_MODEL,
    INFINITY_URL,
    MODERNBERT_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    RESIDENT_ADMIN_TIMEOUT,
    RUNTIME_EXTRA_RESIDENTS,
    TTS_ENGINES,
    WHISPER_URL,
    ollama_keep_alive,
)


router = APIRouter(prefix="/api/runtime", tags=["runtime"], dependencies=[Depends(require_user)])


class RuntimeModelRequest(BaseModel):
    model: Optional[str] = None


async def _ollama_post(path: str, payload: Dict[str, Any], *, timeout: int = OLLAMA_TIMEOUT) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{OLLAMA_URL}{path}", json=payload)
    except httpx.TimeoutException:
        raise HTTPException(504, detail="Ollama timed out")
    except httpx.TransportError:
        raise HTTPException(503, detail="Ollama unreachable")
    if response.status_code >= 400:
        raise upstream_error(response, "Ollama")
    return response.json() if response.content else {}


async def _ollama_get(path: str, *, timeout: int = 30) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{OLLAMA_URL}{path}")
    except httpx.TimeoutException:
        raise HTTPException(504, detail="Ollama timed out")
    except httpx.TransportError:
        raise HTTPException(503, detail="Ollama unreachable")
    if response.status_code >= 400:
        raise upstream_error(response, "Ollama")
    return response.json() if response.content else {}


def _loaded_ollama_models(body: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for item in body.get("models") or []:
        name = item.get("name") or item.get("model")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


@router.get("/ollama/status")
async def ollama_status():
    ps = await _ollama_get("/api/ps")
    return {
        "state": "loaded" if _loaded_ollama_models(ps) else "idle",
        "keep_alive": OLLAMA_KEEP_ALIVE or "server",
        "models": ps.get("models", []),
    }


@router.post("/ollama/wake")
async def ollama_wake(req: RuntimeModelRequest):
    model = (req.model or "").strip() or CHAT_MODEL
    await _ollama_post(
        "/api/generate",
        {"model": model, "prompt": "", "stream": False, **ollama_keep_alive()},
    )
    return {"state": "loaded", "model": model, "keep_alive": OLLAMA_KEEP_ALIVE or "server"}


def _same_model(a: str, b: str) -> bool:
    """Ollama names a model with or without its default ``:latest`` tag."""
    def full(name: str) -> str:
        return name if ":" in name else f"{name}:latest"
    return full(a) == full(b)


@router.post("/ollama/evict")
async def ollama_evict(req: RuntimeModelRequest):
    # Always look at what is loaded: an empty-prompt generate for a model that isn't
    # resident would load it (pushing others out) only to drop it again.
    loaded = _loaded_ollama_models(await _ollama_get("/api/ps"))
    requested = req.model.strip() if req.model else ""
    models = [m for m in loaded if _same_model(m, requested)] if requested else loaded
    evicted: List[str] = []
    for model in models:
        try:
            await _ollama_post(
                "/api/generate",
                {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            )
        except HTTPException as exc:
            # Keep Ollama's status, but say what already happened: a bare error would hide
            # the models this call did unload, and a retry couldn't tell where it stopped.
            return JSONResponse(status_code=exc.status_code, content={
                "state": "partial" if evicted else "failed", "evicted": evicted, "failed": model,
                "detail": exc.detail, "loaded": [m for m in loaded if m not in evicted],
            })
        evicted.append(model)
    if not evicted:
        return {"state": "already_idle", "evicted": evicted}
    # Evicting one model leaves the others resident: report what is still loaded.
    remaining = [m for m in loaded if m not in evicted]
    return {"state": "loaded" if remaining else "idle", "evicted": evicted, "loaded": remaining}


def _admin_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {ARGUS_ADMIN_TOKEN}"} if ARGUS_ADMIN_TOKEN else {}


async def _infinity_get(path: str, *, wake: bool = False) -> Dict[str, Any]:
    headers = {"X-Infinity-Wake": "1"} if wake else {}
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.get(f"{INFINITY_URL}{path}", headers=headers)
    except httpx.TimeoutException:
        raise HTTPException(504, detail="Infinity timed out")
    except httpx.TransportError:
        raise HTTPException(503, detail="Infinity unreachable")
    if response.status_code >= 400:
        raise upstream_error(response, "Infinity")
    return response.json() if response.content else {}


async def _infinity_post(path: str) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(f"{INFINITY_URL}{path}", headers=_admin_headers())
    except httpx.TimeoutException:
        raise HTTPException(504, detail="Infinity timed out")
    except httpx.TransportError:
        raise HTTPException(503, detail="Infinity unreachable")
    if response.status_code >= 400:
        raise upstream_error(response, "Infinity")
    return response.json() if response.content else {}


@router.get("/infinity/status")
async def infinity_status():
    try:
        return await _infinity_get("/admin/status")
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        models = await _infinity_get("/models")
        return {
            "state": "external",
            "models": models.get("data", []),
            "managed": False,
            "checked_at": time.time(),
        }


@router.post("/infinity/wake")
async def infinity_wake():
    try:
        return await _infinity_post("/admin/wake")
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        models = await _infinity_get("/models")
        return {"state": "external", "models": models.get("data", []), "managed": False}


@router.post("/infinity/evict")
async def infinity_evict():
    try:
        return await _infinity_post("/admin/evict")
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(501, detail="Infinity backend does not expose eviction")
        raise


# ------------------------------------------------------------ GPU residents
# One authenticated, LAN-bindable door to the sidecars, whose own ports are
# loopback/internal only. Each resident speaks the argus-iris "argus-http"
# contract (GET /health with a ``gpu`` block, POST /admin/{unload,evict,wake}),
# so iris's base_url for one is ``<rag-chat>/api/runtime/residents/<name>``.
# Ollama is not proxied here: iris's ollama adapter talks to :11434 directly.

ResidentAction = Literal["unload", "evict", "wake"]
_HEALTH_TIMEOUT = 10.0


def _residents() -> Dict[str, str]:
    # Later sources win a name clash: extras < fixed backends < TTS engines. Each source drops its
    # empty URLs first, so an unset backend never removes a resident another source configured.
    found: Dict[str, str] = {}
    for source in (
        RUNTIME_EXTRA_RESIDENTS,  # e.g. tts (Qwen, used by argus-mask); never TTS engines
        {"modernbert": MODERNBERT_URL, "whisper": WHISPER_URL, "infinity": INFINITY_URL},
        TTS_ENGINES,  # xtts, miso (remote), tts (legacy Qwen)
    ):
        found.update({name: url.rstrip("/") for name, url in source.items() if url})
    return found


def _resident_url(name: str) -> str:
    url = _residents().get(name)
    if not url:
        raise HTTPException(404, detail=f"unknown resident {name!r}; known: {sorted(_residents())}")
    return url


async def _call(name: str, method: str, path: str, timeout: float) -> tuple[int, Any]:
    """One sidecar call: (status code, JSON body). Unreachable → 503, slow → 504."""
    url = _resident_url(name)
    headers = _admin_headers() if method == "POST" else {}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, f"{url}{path}", headers=headers)
    except httpx.TimeoutException:
        raise HTTPException(504, detail=f"{name} timed out")
    except httpx.HTTPError:
        raise HTTPException(503, detail=f"{name} unreachable")
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, {"detail": response.text[:500]}


async def _forward(name: str, method: str, path: str, timeout: float) -> JSONResponse:
    """Relay a sidecar call, keeping its status code (409 busy, 503 loading...)."""
    status, body = await _call(name, method, path, timeout)
    return JSONResponse(status_code=status, content=body)


async def _summary(name: str) -> Dict[str, Any]:
    try:
        status, body = await _call(name, "GET", "/health", _HEALTH_TIMEOUT)
    except HTTPException as exc:
        return {"name": name, "reachable": False, "healthy": False, "status_code": exc.status_code,
                "detail": exc.detail}
    healthy = 200 <= status < 300
    raw = body
    body = body if isinstance(body, dict) else {}
    summary = {
        "name": name,
        "reachable": True,
        # Reachable is not healthy: a sidecar answering 500/503 is up but failing.
        "healthy": healthy,
        "status_code": status,
        "model_state": body.get("model_state"),
        "busy": body.get("busy"),
        "idle_s": body.get("idle_s"),
        "gpu": body.get("gpu"),
    }
    if not healthy:
        summary["detail"] = (body.get("detail") or body.get("error") or str(raw))[:500] if raw else ""
    return summary


@router.get("/residents")
async def list_residents():
    """Every GPU sidecar this host knows, with its health ``gpu`` block."""
    names = sorted(_residents())
    return {"residents": list(await asyncio.gather(*(_summary(n) for n in names)))}


@router.get("/residents/{name}/health")
async def resident_health(name: str):
    return await _forward(name, "GET", "/health", _HEALTH_TIMEOUT)


@router.post("/residents/{name}/admin/{action}")
async def resident_admin(name: str, action: ResidentAction):
    """``unload`` frees VRAM and RAM, ``evict`` drops to CPU where the sidecar has
    a CPU stage (else unloads), ``wake`` loads onto the GPU."""
    return await _forward(name, "POST", f"/admin/{action}", RESIDENT_ADMIN_TIMEOUT)

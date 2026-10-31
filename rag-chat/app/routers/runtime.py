import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import require_user
from app.config import CHAT_MODEL, INFINITY_URL, OLLAMA_KEEP_ALIVE, OLLAMA_TIMEOUT, OLLAMA_URL


router = APIRouter(prefix="/api/runtime", tags=["runtime"], dependencies=[Depends(require_user)])


class RuntimeModelRequest(BaseModel):
    model: Optional[str] = None


async def _ollama_post(path: str, payload: Dict[str, Any], *, timeout: int = OLLAMA_TIMEOUT) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{OLLAMA_URL}{path}", json=payload)
    except httpx.ConnectError:
        raise HTTPException(503, detail="Ollama unreachable")
    if response.status_code >= 400:
        raise HTTPException(response.status_code, detail=response.text)
    return response.json() if response.content else {}


async def _ollama_get(path: str, *, timeout: int = 30) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{OLLAMA_URL}{path}")
    except httpx.ConnectError:
        raise HTTPException(503, detail="Ollama unreachable")
    if response.status_code >= 400:
        raise HTTPException(response.status_code, detail=response.text)
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
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "models": ps.get("models", []),
    }


@router.post("/ollama/wake")
async def ollama_wake(req: RuntimeModelRequest):
    model = (req.model or CHAT_MODEL).strip()
    await _ollama_post(
        "/api/generate",
        {"model": model, "prompt": "", "stream": False, "keep_alive": OLLAMA_KEEP_ALIVE},
    )
    return {"state": "loaded", "model": model, "keep_alive": OLLAMA_KEEP_ALIVE}


@router.post("/ollama/evict")
async def ollama_evict(req: RuntimeModelRequest):
    models = [req.model.strip()] if req.model and req.model.strip() else []
    if not models:
        models = _loaded_ollama_models(await _ollama_get("/api/ps"))
    evicted: List[str] = []
    for model in models:
        await _ollama_post(
            "/api/generate",
            {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
        )
        evicted.append(model)
    return {"state": "idle" if evicted else "already_idle", "evicted": evicted}


async def _infinity_get(path: str, *, wake: bool = False) -> Dict[str, Any]:
    headers = {"X-Infinity-Wake": "1"} if wake else {}
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.get(f"{INFINITY_URL}{path}", headers=headers)
    except httpx.ConnectError:
        raise HTTPException(503, detail="Infinity unreachable")
    if response.status_code >= 400:
        raise HTTPException(response.status_code, detail=response.text)
    return response.json() if response.content else {}


async def _infinity_post(path: str) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(f"{INFINITY_URL}{path}")
    except httpx.ConnectError:
        raise HTTPException(503, detail="Infinity unreachable")
    if response.status_code >= 400:
        raise HTTPException(response.status_code, detail=response.text)
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

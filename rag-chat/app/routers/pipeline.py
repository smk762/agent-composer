from typing import List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import MODERNBERT_URL

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

_TIMEOUT_FAST = 10.0   # status / label ops
_TIMEOUT_INFER = 120.0 # classify (may trigger model load from disk)


async def _get(path: str, timeout: float = _TIMEOUT_FAST):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{MODERNBERT_URL}{path}")
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        raise HTTPException(503, detail="ModernBERT service unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)


async def _post(path: str, json=None, timeout: float = _TIMEOUT_FAST):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{MODERNBERT_URL}{path}", json=json)
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        raise HTTPException(503, detail="ModernBERT service unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)


# ------------------------------------------------------------------ model ops

@router.get("/status")
async def status():
    return await _get("/model/status")


@router.post("/load")
async def load():
    return await _post("/model/load", timeout=_TIMEOUT_INFER)


@router.post("/evict")
async def evict():
    return await _post("/model/evict")


@router.post("/unload")
async def unload():
    return await _post("/model/unload")


# ------------------------------------------------------------------ labels

@router.get("/labels")
async def get_labels():
    return await _get("/labels")


class LabelUpdate(BaseModel):
    labels: List[str]


@router.put("/labels")
async def update_labels(body: LabelUpdate):
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_INFER) as client:
            r = await client.put(f"{MODERNBERT_URL}/labels", json=body.model_dump())
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        raise HTTPException(503, detail="ModernBERT service unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)


# ------------------------------------------------------------------ classify

class ClassifyRequest(BaseModel):
    text: str
    context: Optional[List[str]] = None
    top_k: Optional[int] = None
    threshold: Optional[float] = None


@router.post("/classify")
async def classify(req: ClassifyRequest):
    return await _post(
        "/classify",
        json=req.model_dump(exclude_none=True),
        timeout=_TIMEOUT_INFER,
    )

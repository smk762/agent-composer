import json
from typing import AsyncIterator, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import CHAT_MODEL, MODERNBERT_URL, OLLAMA_KEEP_ALIVE, OLLAMA_TIMEOUT, OLLAMA_URL

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

_TIMEOUT_FAST = 10.0   # status / label ops
_TIMEOUT_INFER = 120.0 # classify (may trigger model load from disk)

# ------------------------------------------------------------------ prompts

_OVERLAY_SYSTEM = (
    "You are a voice director for a character-driven AI. "
    "Given emotional signal and character context, write a concise voice directive "
    "(80–120 words) telling the prose writer exactly HOW to write: tone, register, "
    "sentence rhythm, what to emphasise, what to avoid. Output only the directive text, "
    "no labels or headers."
)

_BRAIN_SYSTEM_TPL = (
    "You are the reasoning core for a character named {name}. "
    "You see everything — full context, history, emotional arc — but you do NOT write in character voice. "
    "Plan what {name} should say: what to reveal, what emotional beat to hit, what to avoid. "
    "Output structured notes using these headings:\n"
    "INTENT: (what {name} wants to achieve)\n"
    "KEY POINTS: (bullet list of content to include)\n"
    "EMOTIONAL ARC: (how the tone should shift across the response)\n"
    "AVOID: (what not to say or do)"
)

_PROSE_SYSTEM_TPL = (
    "You are writing as {name}. Follow the voice directive precisely. "
    "Write naturally in first person in {name}'s voice. "
    "Do not include labels, headers, or meta-commentary — only the character's words."
)


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


# ------------------------------------------------------------------ Ollama helper

async def _ollama_generate(model: str, system: str, prompt: str) -> str:
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            r = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            r.raise_for_status()
            return r.json().get("response", "")
    except httpx.ConnectError:
        raise HTTPException(503, detail="Ollama unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)


async def _ollama_stream(model: str, system: str, prompt: str) -> AsyncIterator[str]:
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
    except httpx.ConnectError:
        yield "\n[Ollama unreachable]"


# ------------------------------------------------------------------ Stage 2: overlay

class OverlayRequest(BaseModel):
    character_name: str
    character_persona: str
    character_state: str = ""
    labels: List[str]
    scores: List[float]
    turns: List[str] = []
    model: str = ""


@router.post("/overlay")
async def overlay(req: OverlayRequest):
    model = req.model or CHAT_MODEL
    signal = ", ".join(
        f"{lbl} ({score:.2f})" for lbl, score in zip(req.labels[:5], req.scores[:5])
    )
    ctx = "\n".join(f"  {t}" for t in req.turns[-5:]) if req.turns else "  (none)"
    prompt = (
        f"Character: {req.character_name}\n"
        f"Persona: {req.character_persona}\n"
        + (f"Current state: {req.character_state}\n" if req.character_state else "")
        + f"Classifier signal: {signal}\n"
        f"Recent context:\n{ctx}\n\n"
        "Voice directive:"
    )
    directive = await _ollama_generate(model, _OVERLAY_SYSTEM, prompt)
    return {"directive": directive, "model": model}


# ------------------------------------------------------------------ Stage 3: brain

class BrainRequest(BaseModel):
    character_name: str
    character_persona: str
    character_state: str = ""
    style_directive: str
    labels: List[str]
    turns: List[str] = []
    message: str
    model: str = ""


@router.post("/brain")
async def brain(req: BrainRequest):
    model = req.model or CHAT_MODEL
    system = _BRAIN_SYSTEM_TPL.format(name=req.character_name)
    signal = ", ".join(req.labels[:5])
    history = "\n".join(f"  {t}" for t in req.turns[-5:]) if req.turns else "  (none)"
    prompt = (
        f"Voice directive: {req.style_directive}\n"
        f"Classifier: {signal}\n"
        f"History:\n{history}\n"
        f"User message: {req.message}\n\n"
        "Plan:"
    )
    skeleton = await _ollama_generate(model, system, prompt)
    return {"skeleton": skeleton, "model": model}


# ------------------------------------------------------------------ Stage 4: prose (streaming)

class ProseRequest(BaseModel):
    character_name: str
    style_directive: str
    skeleton: str
    turns: List[str] = []
    message: str
    model: str = ""


@router.post("/prose")
async def prose(req: ProseRequest):
    model = req.model or CHAT_MODEL
    system = _PROSE_SYSTEM_TPL.format(name=req.character_name)
    recent = "\n".join(f"  {t}" for t in req.turns[-3:]) if req.turns else "  (none)"
    prompt = (
        f"Voice directive: {req.style_directive}\n"
        f"Plan:\n{req.skeleton}\n"
        f"Recent exchanges:\n{recent}\n"
        f"User: {req.message}\n\n"
        f"{req.character_name}:"
    )

    async def event_stream():
        async for token in _ollama_stream(model, system, prompt):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

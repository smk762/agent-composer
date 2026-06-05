"""LlamaGuard content-safety router.

Runs input (and optionally a conversation) through the LlamaGuard model
on the shared Ollama instance.  Exposes a JSON API at ``/api/guard/check``
that any service — internal or external — can call.
"""

from __future__ import annotations

import time
from typing import List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import (
    GUARD_MODEL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    log,
)

router = APIRouter(prefix="/api/guard", tags=["guard"])


# ── Request / response models ────────────────────────────────────────────────

class GuardMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class GuardCheckRequest(BaseModel):
    """Check a single prompt or a multi-turn conversation."""
    input: str = Field(
        ...,
        description="The primary text to evaluate (treated as the latest user turn).",
    )
    conversation: Optional[List[GuardMessage]] = Field(
        default=None,
        description="Optional prior turns for context.  Sent as preceding messages.",
    )
    model: str = Field(
        default="",
        description="Override the guard model (defaults to GUARD_MODEL env var).",
    )


class GuardCategory(BaseModel):
    code: str
    label: str


class GuardCheckResponse(BaseModel):
    safe: bool
    verdict: str
    categories: List[GuardCategory]
    raw: str
    model: str
    elapsed_ms: int


# ── Helpers ───────────────────────────────────────────────────────────────────

_LLAMAGUARD_CATEGORIES = [
    ("S1", "Violent Crimes"),
    ("S2", "Non-Violent Crimes"),
    ("S3", "Sex-Related Crimes"),
    ("S4", "Child Sexual Exploitation"),
    ("S5", "Defamation"),
    ("S6", "Specialized Advice"),
    ("S7", "Privacy"),
    ("S8", "Intellectual Property"),
    ("S9", "Indiscriminate Weapons"),
    ("S10", "Hate"),
    ("S11", "Suicide & Self-Harm"),
    ("S12", "Sexual Content"),
    ("S13", "Elections"),
    ("S14", "Code Interpreter Abuse"),
]

_CODE_TO_LABEL = {code: label for code, label in _LLAMAGUARD_CATEGORIES}


def _parse_verdict(raw: str) -> tuple[bool, str, list[GuardCategory]]:
    """Parse LlamaGuard output into structured fields.

    LlamaGuard typically responds with either:
      ``safe``
    or:
      ``unsafe\\nS1,S3``   (comma- or newline-separated category codes)
    """
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return True, "safe", []

    verdict = lines[0].lower()
    is_safe = verdict == "safe"

    categories: list[GuardCategory] = []
    for line in lines[1:]:
        for token in line.replace(",", "\n").split("\n"):
            code = token.strip().upper()
            if code in _CODE_TO_LABEL:
                categories.append(GuardCategory(code=code, label=_CODE_TO_LABEL[code]))

    return is_safe, "safe" if is_safe else "unsafe", categories


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/check", response_model=GuardCheckResponse)
async def guard_check(req: GuardCheckRequest):
    model = req.model or GUARD_MODEL

    messages: list[dict[str, str]] = []
    if req.conversation:
        for msg in req.conversation:
            messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": req.input})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            r.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(503, detail="Ollama unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, detail=exc.response.text)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    raw = r.json().get("message", {}).get("content", "").strip()
    is_safe, verdict, categories = _parse_verdict(raw)

    log.info(
        "guard: model=%s safe=%s verdict=%s categories=%s elapsed=%dms",
        model, is_safe, verdict,
        [c.code for c in categories], elapsed_ms,
    )

    return GuardCheckResponse(
        safe=is_safe,
        verdict=verdict,
        categories=categories,
        raw=raw,
        model=model,
        elapsed_ms=elapsed_ms,
    )


@router.get("/categories")
async def guard_categories():
    """Return the LlamaGuard category taxonomy."""
    return [{"code": c, "label": l} for c, l in _LLAMAGUARD_CATEGORIES]

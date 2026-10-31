"""SingGuard content-safety router.

Runs a prompt, and any prior turns, through SingGuard on the shared Ollama
instance. The moderation policy is natural-language input: omit it to use
SingGuard's default rules, or send a replacement and the model judges only
those rules.

The response is the safe/unsafe label, the triggered rule title, and why
that rule applies. Fast-slow writes that reason after a first-line verdict;
slow writes the reason first so it is not anchored; cascade runs fast, then
slow only when fast flags the content.
"""

from __future__ import annotations

import time
from typing import List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import (
    GUARD_MODEL,
    GUARD_THINKING,
    GUARD_WORKER_CONCURRENCY,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    log,
)
from app.guard_image import image_b64
from app.inference_retry import InferenceWorkers
from app.guard_policy import (
    DEFAULT_POLICY,
    THINKING_CASCADE,
    THINKING_FAST,
    THINKING_MODES,
    THINKING_SLOW,
    GuardVerdict,
    build_image_chat_messages,
    build_singguard_prompt,
    finish_cascade,
    max_new_tokens,
    needs_slow_pass,
    normalize_thinking,
    parse_singguard_output,
    policy_rules,
)

router = APIRouter(prefix="/api/guard", tags=["guard"])
_guard_workers = InferenceWorkers(GUARD_WORKER_CONCURRENCY)


class GuardMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class GuardCheckRequest(BaseModel):
    """Check a user turn, or an assistant turn in the context of the user."""

    input: str = Field(
        default="",
        description="Text to evaluate. For a generated image, this is the prompt.",
    )
    image_base64: str = Field(
        default="",
        description="Optional image (raw base64 or a data URL). Judged with the latest user turn.",
    )
    conversation: Optional[List[GuardMessage]] = Field(
        default=None,
        description="Prior turns. Included so a reply can be judged in context.",
    )
    input_role: str = Field(
        default="user",
        pattern="^(user|assistant)$",
        description="Role of `input`. Use assistant to judge a model reply.",
    )
    policy: str = Field(
        default="",
        description="Active moderation policy. Blank uses SingGuard's default rules.",
    )
    thinking: str = Field(
        default="",
        description=(
            "fast: label and rule. fast-slow: first-line verdict then why. "
            "slow: reason first, then the verdict. "
            "cascade: fast, then slow only when fast flags the content."
        ),
    )
    model: str = Field(
        default="",
        description="Override the guard model (defaults to GUARD_MODEL).",
    )


class GuardCategory(BaseModel):
    code: str
    label: str
    why: str = ""


class GuardCheckResponse(BaseModel):
    safe: bool
    verdict: str
    rule: str
    rule_text: str = ""
    explanation: str
    categories: List[GuardCategory]
    raw: str
    model: str
    thinking: str
    elapsed_ms: int


def _turns(req: GuardCheckRequest) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    if req.conversation:
        for message in req.conversation:
            turns.append((message.role, message.content))
    has_image = bool(req.image_base64.strip())
    has_user = any(role == "user" for role, _ in turns)
    # The image has to sit on a user turn. An assistant caption is still
    # included when the prompt is already in the conversation.
    if has_image and (req.input_role == "user" or not has_user):
        turns.append(("user", req.input))
    elif req.input.strip() or not has_image:
        turns.append((req.input_role, req.input))
    return turns


def _ollama_options(thinking: str) -> dict:
    # The GGUF advertises a 262k context. A classifier prompt is a few
    # thousand tokens; reserving the full window wastes VRAM.
    return {
        "temperature": 0,
        "num_predict": max_new_tokens(thinking),
        "num_ctx": 16384,
    }


def _categories(verdict: GuardVerdict) -> list[GuardCategory]:
    if verdict.safe or not verdict.rule or verdict.rule.casefold() == "safe":
        return []
    return [
        GuardCategory(
            code=verdict.rule,
            label=verdict.rule,
            why=verdict.rule_text,
        )
    ]


def _ollama_request(
    *,
    model: str,
    turns: list[tuple[str, str]],
    policy: str,
    thinking: str,
    image: str,
) -> tuple[str, dict]:
    if image:
        # Chat, not a raw prompt: Ollama has to run the vision projector.
        # The GGUF template still applies the policy from the system message.
        messages = build_image_chat_messages(
            turns,
            policy=policy,
            thinking=thinking,
            image_b64=image,
        )
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": _ollama_options(thinking),
        }
        return f"{OLLAMA_URL}/api/chat", payload
    prompt = build_singguard_prompt(turns, policy=policy, thinking=thinking)
    # raw=true keeps Ollama from wrapping this in a second chat template.
    payload = {
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {**_ollama_options(thinking), "stop": ["<|im_end|>"]},
    }
    return f"{OLLAMA_URL}/api/generate", payload


def _raw_from_body(body: dict) -> str:
    return body.get("response") or body.get("message", {}).get("content", "") or ""


async def _generate(
    client: httpx.AsyncClient,
    *,
    model: str,
    turns: list[tuple[str, str]],
    policy: str,
    thinking: str,
    image: str,
) -> str:
    endpoint, payload = _ollama_request(
        model=model,
        turns=turns,
        policy=policy,
        thinking=thinking,
        image=image,
    )
    response = await client.post(endpoint, json=payload)
    response.raise_for_status()
    return _raw_from_body(response.json())


@router.post("/check", response_model=GuardCheckResponse)
async def guard_check(req: GuardCheckRequest):
    model = req.model or GUARD_MODEL
    try:
        requested = normalize_thinking(req.thinking or GUARD_THINKING)
        image = image_b64(req.image_base64)
        turns = _turns(req)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            # One Ollama guard at a time by default: cull scans and darkroom
            # pre-flights share the model, and a pile-up is a timeout, not speed.
            async def _run() -> tuple[str, GuardVerdict]:
                if requested != THINKING_CASCADE:
                    raw = await _generate(
                        client,
                        model=model,
                        turns=turns,
                        policy=req.policy,
                        thinking=requested,
                        image=image,
                    )
                    return requested, parse_singguard_output(raw, req.policy)
                raw_fast = await _generate(
                    client,
                    model=model,
                    turns=turns,
                    policy=req.policy,
                    thinking=THINKING_FAST,
                    image=image,
                )
                fast = parse_singguard_output(raw_fast, req.policy)
                if not needs_slow_pass(fast):
                    return THINKING_FAST, fast
                raw_slow = await _generate(
                    client,
                    model=model,
                    turns=turns,
                    policy=req.policy,
                    thinking=THINKING_SLOW,
                    image=image,
                )
                return THINKING_SLOW, finish_cascade(fast, parse_singguard_output(raw_slow, req.policy))

            used, verdict = await _guard_workers.run(_run)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except httpx.ConnectError:
        raise HTTPException(503, detail="Ollama unreachable")
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        try:
            body = exc.response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and body.get("error"):
            detail = str(body["error"])
        raise HTTPException(exc.response.status_code, detail=detail)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    categories = _categories(verdict)

    log.info(
        "guard: model=%s safe=%s rule=%s thinking=%s requested=%s image=%s elapsed=%dms",
        model,
        verdict.safe,
        verdict.rule,
        used,
        requested,
        bool(image),
        elapsed_ms,
    )

    return GuardCheckResponse(
        safe=verdict.safe,
        verdict=verdict.verdict,
        rule=verdict.rule,
        rule_text=verdict.rule_text,
        explanation=verdict.explanation,
        categories=categories,
        raw=verdict.raw,
        model=model,
        thinking=used,
        elapsed_ms=elapsed_ms,
    )


@router.get("/categories")
async def guard_categories():
    """Default SingGuard rules. A check can replace these with `policy`."""
    return [
        {"title": rule.title, "text": rule.text}
        for rule in policy_rules(None)
    ]


@router.get("/policy")
async def guard_policy():
    """The default policy text, so clients can edit it and send it back."""
    return {
        "policy": DEFAULT_POLICY,
        "thinking": GUARD_THINKING,
        "thinking_modes": list(THINKING_MODES),
    }

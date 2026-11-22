"""SingGuard content-safety router: a thin proxy to argus-retina's SingGuard server.

``POST /api/guard/check`` keeps rag-chat's request and response shape, so its callers (cull
scans, darkroom pre-flights, safelight, imogen, the guard tester) don't change, and forwards
the check to ``GUARD_URL``'s ``/api/guard/check`` (ghcr.io/smk762/argus-retina-singguard).
Retina renders the prompt, neutralizes control tokens and role markers, shrinks the image,
takes its own GPU lease from argus-iris and runs one check at a time; rag-chat fills the
default ``thinking`` and maps the answer and its errors.

The moderation policy is natural-language request input: omit it to use SingGuard's
default rules, or send a replacement and the model judges only those rules. The response is
the safe/unsafe label, the triggered rule title, and why that rule applies.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import GUARD_THINKING, GUARD_TIMEOUT, GUARD_URL, log
from app.guard_policy import DEFAULT_POLICY, THINKING_MODES, normalize_thinking, policy_rules, rule_text

router = APIRouter(prefix="/api/guard", tags=["guard"])

_DETAIL_CAP = 300
# The model the upstream serves, read from its /health the first time a check names one.
_upstream_model: str | None = None


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
            "cascade: fast, then slow only when fast flags the content. Blank: GUARD_THINKING."
        ),
    )
    model: str = Field(
        default="",
        description="Must be blank or the upstream's model: argus-retina serves one SingGuard.",
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
    # The generation text, which retina does not return: always "".
    raw: str = ""
    model: str
    thinking: str
    elapsed_ms: int
    # Retina's first-line verdict; on a cascade that slow decided, also the rule fast flagged.
    fast_verdict: Optional[str] = None
    fast_rule: Optional[str] = None


def _detail(response: httpx.Response) -> str:
    """The upstream's own message about this request (FastAPI's ``detail``), short."""
    try:
        detail: Any = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, list):  # a 422: pydantic's errors
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', ())[1:])}: {e.get('msg', '')}".lstrip(": ")
            for e in detail
            if isinstance(e, dict)
        )
    return str(detail or f"guard refused the check ({response.status_code})")[:_DETAIL_CAP]


async def _model(client: httpx.AsyncClient) -> str:
    global _upstream_model
    if _upstream_model is None:
        response = await client.get(f"{GUARD_URL}/health")
        response.raise_for_status()
        model = response.json().get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("guard /health names no model")
        _upstream_model = model
    return _upstream_model


def _response(body: dict, policy: str, elapsed_ms: int) -> GuardCheckResponse:
    categories = [GuardCategory(**c) for c in body.get("categories") or []]
    rule = body.get("rule") or ""
    # The rule's own text from the policy the check was judged against (retina's category
    # `why` is the explanation, already in `explanation`); the category's why when the rule
    # is not a heading of that policy.
    text = rule_text(rule, policy) or (categories[0].why if categories else "")
    return GuardCheckResponse(
        safe=body["safe"],
        verdict=body.get("verdict") or ("safe" if body["safe"] else "unsafe"),
        rule=rule,
        rule_text=text,
        explanation=body.get("explanation") or "",
        categories=categories,
        raw="",
        model=body.get("model") or "",
        thinking=body.get("thinking") or "",
        elapsed_ms=elapsed_ms,
        fast_verdict=body.get("fast_verdict"),
        fast_rule=body.get("fast_rule"),
    )


@router.post("/check", response_model=GuardCheckResponse, response_model_exclude_none=True)
async def guard_check(req: GuardCheckRequest):
    try:
        thinking = normalize_thinking(req.thinking or GUARD_THINKING)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    if not GUARD_URL:
        raise HTTPException(503, detail="guard not configured (set GUARD_URL)")

    payload = {
        "input": req.input,
        "image_base64": req.image_base64,
        "input_role": req.input_role,
        "conversation": [m.model_dump() for m in req.conversation] if req.conversation is not None else None,
        "policy": req.policy or None,
        "thinking": thinking,
    }
    t0 = time.monotonic()
    try:
        # GUARD_URL is operator config, never request input.
        async with httpx.AsyncClient(timeout=GUARD_TIMEOUT) as client:
            if req.model.strip():
                served = await _model(client)
                if req.model.strip() != served:
                    raise HTTPException(
                        400, detail=f"the guard serves only {served}; leave model blank"
                    )
            response = await client.post(f"{GUARD_URL}/api/guard/check", json=payload)
            response.raise_for_status()
            result = _response(response.json(), req.policy, int((time.monotonic() - t0) * 1000))
    except httpx.TimeoutException:
        raise HTTPException(504, detail="guard timed out")
    except (httpx.TransportError, httpx.InvalidURL):
        raise HTTPException(503, detail="guard unreachable")
    except httpx.HTTPStatusError as exc:
        upstream = exc.response.status_code
        if upstream in (400, 422):
            raise HTTPException(400, detail=_detail(exc.response))
        if upstream == 503:
            # info-leak: retina's own 503 detail (model loading, no GPU lease), JSON field, capped
            raise HTTPException(503, detail=f"guard unavailable: {_detail(exc.response)}")
        log.warning("guard: upstream answered %s", upstream)
        if upstream == 500:
            raise HTTPException(502, detail="guard returned no verdict")
        raise HTTPException(502, detail=f"guard failed ({upstream})")
    except (ValueError, KeyError, TypeError) as exc:
        # Undecodable JSON or a body without the fields: an upstream fault, not the caller's.
        log.warning("guard: unusable upstream response: %s", exc)
        raise HTTPException(502, detail="guard returned an unusable response")

    log.info(
        "guard: model=%s safe=%s rule=%s thinking=%s requested=%s image=%s elapsed=%dms",
        result.model,
        result.safe,
        result.rule,
        result.thinking,
        thinking,
        bool(req.image_base64.strip()),
        result.elapsed_ms,
    )
    return result


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

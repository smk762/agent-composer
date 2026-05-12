"""Multi-model repair loop: fixer → critic → optional adversary.

Each iteration yields a RepairIteration so the SSE router can stream
progress to the client without waiting for the full loop to finish.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

import httpx

from app.config import CHAT_MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_TIMEOUT, OLLAMA_URL
from app.repair.context_builder import build_repair_context, fetch_qdrant_snippets
from app.repair.models import RepairIteration, RepairRequest
from app.repair.patch_applier import extract_patch

log = logging.getLogger("rag-chat.repair")


# ── Prompt templates ──────────────────────────────────────────────────────────

_FIXER_SYSTEM_AUTO = """\
You are an expert software engineer performing a code repair.
You will be given a unified diff, any validation errors, and related code context.
Your task is to produce a corrected unified diff that fixes all stated problems
while preserving the original intent of the change.

Output ONLY a unified diff inside a ```diff ... ``` fenced block.
Do not include explanations outside the fence — put them in comments inside the diff.
If you cannot produce a valid diff, output an empty ```diff ``` block and explain why
in a brief paragraph before the fence."""

_FIXER_SYSTEM_REVIEW = """\
You are a senior code reviewer performing an annotated review.
You will be given a unified diff, any validation errors, and related code context.
Produce:
1. A brief analysis (bullet points) of the issues found in the diff.
2. A corrected unified diff inside a ```diff ... ``` fenced block (if changes are needed).
3. A short summary of what was changed and why.

If no changes are needed, say so and output an empty ```diff ``` block."""

_FIXER_SYSTEM_SUGGEST = """\
You are a code improvement advisor.
You will be given a unified diff and related code context.
Suggest improvements WITHOUT producing a new diff.
Format your suggestions as a numbered list, grouped by: correctness, security, performance, style.
Be concise and reference specific line numbers where relevant."""

_FIXER_SYSTEM = _FIXER_SYSTEM_AUTO  # default; overridden per mode below

_CRITIC_SYSTEM = """\
You are a senior code reviewer critiquing a proposed repair diff.
Identify any remaining issues: logic errors, security risks, style violations,
missing tests, or opportunities to simplify.
Be concise and specific. Reference line numbers where possible."""

_ADVERSARY_SYSTEM = """\
You are a hostile code reviewer trying to find everything wrong with this repair.
Challenge assumptions, probe edge cases, look for subtle bugs, race conditions,
and ways the change could break downstream callers.
Be thorough and adversarial — the goal is to surface every possible objection."""

_VALIDATE_SYSTEM = """\
You are performing a final validation pass on a repaired diff.
Given the original errors and the proposed repair, answer in one word:
PASS if all stated errors are addressed and no new issues are introduced,
FAIL otherwise. Then give one concise sentence of justification."""


def _fixer_prompt(context: str, iteration: int) -> str:
    prefix = f"## Iteration {iteration}\n\n" if iteration > 1 else ""
    return (
        f"{prefix}{context}\n\n"
        "---\n\n"
        "Produce the corrected unified diff now:"
    )


def _critic_prompt(context: str, patch: str) -> str:
    patch_block = f"```diff\n{patch}\n```" if patch else "(no valid patch extracted)"
    return (
        f"{context}\n\n"
        "---\n\n"
        f"## Proposed repair\n\n{patch_block}\n\n"
        "Review the proposed repair and list all remaining issues:"
    )


def _adversary_prompt(context: str, patch: str, critic_notes: str) -> str:
    patch_block = f"```diff\n{patch}\n```" if patch else "(no valid patch extracted)"
    return (
        f"{context}\n\n"
        f"## Proposed repair\n\n{patch_block}\n\n"
        f"## Critic's review\n\n{critic_notes or '(none)'}\n\n"
        "---\n\n"
        "Now find every remaining flaw — be exhaustive:"
    )


# ── Ollama call ───────────────────────────────────────────────────────────────

async def _generate(prompt: str, model: str, system: str) -> str:
    """Call Ollama /api/generate and return the full response text."""
    payload = {
        "model": model or CHAT_MODEL,
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
        log.warning("Ollama unreachable at %s", OLLAMA_URL)
        return ""
    except Exception as exc:
        log.warning("Ollama generate error: %s", exc)
        return ""


# ── Validation via ai-code-auditor audit API ──────────────────────────────────

async def _validate_via_api(
    patch: str,
    repo: str,
    audit_api_url: str,
) -> tuple[bool, list[str]]:
    """POST patch to /audit/diff (static analysis), return (passed, errors)."""
    if not audit_api_url or not patch:
        return True, []
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{audit_api_url.rstrip('/')}/audit/diff",
                json={"repo": repo, "diff": patch},
            )
        if r.status_code != 200:
            log.debug("Audit API returned %d", r.status_code)
            return True, []
        data = r.json()
        findings = data.get("findings", [])
        critical = [
            f"{f.get('severity')} {f.get('file_path')}:{f.get('line')} — {f.get('title')}"
            for f in findings
            if f.get("severity") in ("CRITICAL", "HIGH")
        ]
        return len(critical) == 0, critical
    except Exception as exc:
        log.debug("Validation call failed: %s", exc)
        return True, []


async def _validate_via_tests(
    patch: str,
    repo: str,
    audit_api_url: str,
    timeout: int = 180,
) -> tuple[bool, list[str]]:
    """Apply the patch to a temp copy and run the repo's test/lint suite.

    Calls ``POST /audit/validate_patch`` on the audit API.  The response
    includes a ``failure_summary`` — compact, LLM-ready lines that are
    pre-filtered from the raw test output.  This avoids filling the local
    model's context with thousands of lines of pytest boilerplate.

    Degrades gracefully:
    - Returns (True, []) when the endpoint is unavailable (404 or connection error).
    - Returns (True, []) when validation is skipped (read-only mount, no test runner).
    - Returns (False, [<patch-error>]) when git apply fails on the temp copy.
    """
    if not audit_api_url or not patch:
        return True, []
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{audit_api_url.rstrip('/')}/audit/validate_patch",
                json={"repo": repo, "patch": patch, "timeout_s": max(60, timeout - 30)},
            )
        # 404 means the endpoint isn't deployed yet — skip silently
        if r.status_code == 404:
            return True, []
        if r.status_code != 200:
            log.debug("validate_patch returned %d", r.status_code)
            return True, []

        data = r.json()

        # Skipped (read-only mount, no test runner) → don't penalise
        if data.get("skipped_reason"):
            log.debug("validate_patch skipped: %s", data["skipped_reason"])
            return True, []

        # Patch didn't apply cleanly — this IS a real error to feed back
        if not data.get("patch_applied", True):
            patch_err = data.get("patch_error", "patch did not apply cleanly")
            return False, [f"Patch apply error: {patch_err}"]

        passed = data.get("passed", True)
        # Use failure_summary (pre-filtered, LLM-ready) as the canonical error source.
        # Never fall back to raw errors — they may contain infrastructure noise
        # (broken venv symlinks, missing runners) that would mislead the model.
        failure_summary: list[str] = data.get("failure_summary") or []
        if not passed and not failure_summary:
            # Test runner failed but produced no actionable failure lines.
            # This is an infrastructure issue (broken venv, missing runner, bad shebang
            # in a copied .venv, etc.) — not something the repair model can fix.
            # Log as warning so ops can investigate, but don't feed it to the model.
            raw_errors = data.get("errors") or []
            log.warning(
                "validate_patch: runner failed with no actionable output "
                "(tool=%s, repo=%s) — infra issue, not propagating to repair loop. "
                "Raw errors: %s",
                data.get("tool"), repo, raw_errors[:3],
            )
            return True, []
        return passed, failure_summary

    except httpx.ConnectError:
        log.debug("validate_patch: audit API unreachable")
        return True, []
    except Exception as exc:
        log.debug("validate_patch call failed: %s", exc)
        return True, []


# ── Main repair loop ──────────────────────────────────────────────────────────

async def repair_loop(
    request: RepairRequest,
    qdrant_url: str = "",
) -> AsyncIterator[RepairIteration]:
    """Async generator: yield one RepairIteration per loop pass.

    Stops early if validation passes.  If validation is disabled (no audit_api_url)
    the loop runs for exactly max_iterations.
    """
    fix_model = request.fix_model or CHAT_MODEL
    critique_model = request.critique_model or ""
    adversary_model = request.adversarial_model or ""

    # Select system prompt based on mode
    mode_system = {
        "auto_fix": _FIXER_SYSTEM_AUTO,
        "review": _FIXER_SYSTEM_REVIEW,
        "suggest": _FIXER_SYSTEM_SUGGEST,
    }.get(request.mode, _FIXER_SYSTEM_AUTO)

    # Build initial Qdrant context (best-effort)
    qdrant_snippets: list[str] = []
    if qdrant_url:
        qdrant_snippets = await fetch_qdrant_snippets(
            query=request.diff[:500],
            qdrant_url=qdrant_url,
        )

    critic_notes = ""
    adversary_notes = ""
    current_errors = list(request.errors)

    for iteration in range(1, request.max_iterations + 1):
        context = build_repair_context(
            diff=request.diff,
            errors=current_errors,
            critic_notes=critic_notes if iteration > 1 else "",
            adversary_notes=adversary_notes if iteration > 1 else "",
            qdrant_snippets=qdrant_snippets,
        )

        # ── Model A: fixer ────────────────────────────────────────────────────
        fixer_prompt_text = _fixer_prompt(context, iteration)
        fixer_response = await _generate(fixer_prompt_text, fix_model, mode_system)
        patch = extract_patch(fixer_response)

        iter_result = RepairIteration(
            iteration=iteration,
            fixer_prompt=fixer_prompt_text,
            fixer_response=fixer_response,
            patch=patch,
        )

        # ── Model B: critic ───────────────────────────────────────────────────
        if critique_model and patch:
            critic_prompt_text = _critic_prompt(context, patch)
            critic_notes = await _generate(critic_prompt_text, critique_model, _CRITIC_SYSTEM)
            iter_result.critic_prompt = critic_prompt_text
            iter_result.critic_response = critic_notes

        # ── Model C: adversary ────────────────────────────────────────────────
        if adversary_model and patch:
            adversary_prompt_text = _adversary_prompt(context, patch, critic_notes)
            adversary_notes = await _generate(adversary_prompt_text, adversary_model, _ADVERSARY_SYSTEM)
            iter_result.adversary_prompt = adversary_prompt_text
            iter_result.adversary_response = adversary_notes

        # ── Static diff-audit ─────────────────────────────────────────────────
        if request.audit_api_url and patch:
            passed, val_errors = await _validate_via_api(
                patch, request.repo, request.audit_api_url
            )
            iter_result.validation_passed = passed
            iter_result.validation_errors = val_errors
        elif not request.audit_api_url:
            iter_result.validation_passed = None  # static validation disabled

        # ── Test-suite validation ─────────────────────────────────────────────
        # Only run when:
        # - The static audit didn't already reject the patch (avoid double-spending
        #   inference on a known-bad diff).
        # - The mode produces a patch (suggest mode never does).
        # - validate_tests is enabled (caller can disable for speed).
        test_passed = True
        test_errors: list[str] = []
        if (
            request.audit_api_url
            and patch
            and getattr(request, "validate_tests", True)
            and request.mode != "suggest"
            and iter_result.validation_passed is not False
        ):
            test_passed, test_errors = await _validate_via_tests(
                patch, request.repo, request.audit_api_url
            )
            iter_result.test_validation_passed = test_passed
            iter_result.test_validation_errors = test_errors

        # Merge all error signals for the next iteration's context.
        # Static-audit errors come first (higher signal), test errors follow.
        current_errors = list(iter_result.validation_errors) + test_errors

        yield iter_result

        # Stop when all validations pass, or when the model produced nothing useful
        static_ok = iter_result.validation_passed is not False
        if static_ok and test_passed and patch:
            log.info("Repair loop: all validations passed at iteration %d", iteration)
            break
        if not patch and iteration == 1:
            log.warning("Repair loop: model produced no patch on first iteration")
            break

        # Small yield to avoid blocking the event loop between iterations
        await asyncio.sleep(0)

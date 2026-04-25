"""Agentic repair pipeline router.

POST /api/repair/run  — start a repair job, stream SSE events
GET  /api/repair/jobs/{job_id}  — retrieve final job state
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import AUDIT_API_URL, ECOSYSTEM_CONFIG_PATH, QDRANT_URL
from app.repair.models import RepairIteration, RepairRequest, RepairResult
from app.repair.multi_model import repair_loop
from app.repair.patch_applier import apply_inplace, apply_to_temp

log = logging.getLogger("rag-chat.repair")
router = APIRouter(prefix="/api/repair", tags=["repair"])

# ── In-memory job store ───────────────────────────────────────────────────────
# Keyed by job_id.  Capped at 200 entries (oldest evicted first).

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_MAX_JOBS = 200


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _store_job(result: RepairResult) -> None:
    with _jobs_lock:
        if len(_jobs) >= _MAX_JOBS:
            oldest = next(iter(_jobs))
            _jobs.pop(oldest)
        _jobs[result.job_id] = result.model_dump()


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def _get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/run")
async def run_repair(request: RepairRequest) -> StreamingResponse:
    """Start a repair job and stream SSE events until completion.

    Event types:
    - ``started``   — job created, contains ``job_id``
    - ``iteration`` — one repair pass completed, contains ``RepairIteration`` fields
    - ``complete``  — loop finished, contains ``final_patch`` and whether it was applied
    - ``error``     — unrecoverable failure
    """
    if not request.diff and not request.compare_branch:
        raise HTTPException(
            status_code=422,
            detail="Provide 'diff' (unified diff text) or 'compare_branch'.",
        )

    job_id = f"repair-{secrets.token_hex(8)}"
    result = RepairResult(
        job_id=job_id,
        repo=request.repo,
        status="pending",
        mode=request.mode,
        created_at=_now_iso(),
    )
    _store_job(result)

    # Use configured AUDIT_API_URL as default when the client doesn't override it
    effective_audit_url = request.audit_api_url or AUDIT_API_URL

    async def event_stream():
        _update_job(job_id, status="running")
        yield _sse("started", {"job_id": job_id, "repo": request.repo, "mode": request.mode})

        iterations: list[dict] = []
        final_patch = ""
        error_msg = ""

        try:
            # If compare_branch given, generate diff via the audit API or local git
            diff_text = request.diff
            if not diff_text and request.compare_branch:
                diff_text = await _fetch_diff_from_branch(
                    request.repo, request.compare_branch, request.audit_api_url
                )
                if not diff_text:
                    yield _sse("error", {"message": "Could not generate diff from branch"})
                    _update_job(job_id, status="failed", error="Could not generate diff from branch", completed_at=_now_iso())
                    return

            # Run the multi-model repair loop
            effective_request = request.model_copy(update={
                "diff": diff_text,
                "audit_api_url": effective_audit_url,
            })
            async for iteration in repair_loop(effective_request, qdrant_url=QDRANT_URL):
                iter_dict = iteration.model_dump()
                iterations.append(iter_dict)
                final_patch = iteration.patch or final_patch
                yield _sse("iteration", iter_dict)

                # In auto_fix mode, stop at first successful validation
                if request.mode == "auto_fix" and iteration.validation_passed is True:
                    break

            # Apply in-place if requested
            applied = False
            if request.apply and final_patch and request.mode == "auto_fix":
                try:
                    repo_path = await _resolve_repo_path(request.repo, request.audit_api_url)
                    if repo_path:
                        apply_inplace(repo_path, final_patch)
                        applied = True
                except Exception as exc:
                    log.warning("apply_inplace failed: %s", exc)
                    error_msg = f"Patch generated but apply failed: {exc}"

            _update_job(
                job_id,
                status="completed",
                iterations=iterations,
                final_patch=final_patch,
                applied=applied,
                error=error_msg,
                completed_at=_now_iso(),
            )
            yield _sse("complete", {
                "job_id": job_id,
                "final_patch": final_patch,
                "applied": applied,
                "iterations": len(iterations),
                "error": error_msg,
            })

        except Exception as exc:
            log.exception("Repair job %s failed: %s", job_id, exc)
            _update_job(job_id, status="failed", error=str(exc), completed_at=_now_iso())
            yield _sse("error", {"job_id": job_id, "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    """Retrieve the current state of a repair job."""
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return job


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _fetch_diff_from_branch(
    repo: str,
    compare_branch: str,
    audit_api_url: str,
) -> str:
    """Ask the audit API to generate a git diff, or fall back to local git."""
    import httpx

    if audit_api_url:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    f"{audit_api_url.rstrip('/')}/audit/diff",
                    json={"repo": repo, "compare_branch": compare_branch},
                )
            if r.status_code == 200:
                data = r.json()
                # The audit/diff endpoint doesn't return the raw diff — it returns findings.
                # We need to call /audit/diff to get findings but the diff text comes
                # from the audit API differently. For now, fall through to local git.
                pass
        except Exception as exc:
            log.debug("Audit API fetch_diff failed: %s", exc)

    # Fall back: resolve path from ecosystem config and run git locally
    repo_path = await _resolve_repo_path(repo, audit_api_url)
    if repo_path is None:
        return ""
    import subprocess
    result = subprocess.run(
        ["git", "-C", str(repo_path), "diff", f"{compare_branch}...HEAD"],
        capture_output=True, text=True,
    )
    return result.stdout if result.returncode == 0 else ""


async def _resolve_repo_path(repo: str, audit_api_url: str):
    """Best-effort: find the local path for a named repo.

    Reads the ai-code-auditor ecosystem config if it's on this host.
    Returns None if the path cannot be determined.
    """
    from pathlib import Path

    config_path = ECOSYSTEM_CONFIG_PATH
    try:
        import yaml
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        for entry in raw.get("repos", []):
            if entry.get("name") == repo and entry.get("enabled", True):
                p = Path(entry["path"])
                if p.exists():
                    return p
    except Exception as exc:
        log.debug("Could not read ecosystem config: %s", exc)
    return None

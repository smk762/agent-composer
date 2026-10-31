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
from app.repair.models import GitOpsConfig, RepairIteration, RepairRequest, RepairResult
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
            # If compare_branch given, fetch raw diff via the audit API
            diff_text = request.diff
            if not diff_text and request.compare_branch:
                diff_text, diff_error = await _fetch_diff_from_branch(
                    request.repo, request.compare_branch, effective_audit_url
                )
                if diff_error:
                    yield _sse("error", {"message": diff_error})
                    _update_job(job_id, status="failed", error=diff_error, completed_at=_now_iso())
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

                # In auto_fix mode, stop once both static audit AND test suite pass.
                # The multi_model loop also breaks on this condition; this guard
                # covers the case where the caller set validate_tests=False.
                if request.mode == "auto_fix" and iteration.validation_passed is True:
                    tests_skipped = not getattr(request, "validate_tests", True)
                    test_ok = (
                        tests_skipped
                        or iteration.test_validation_passed is None   # endpoint not available
                        or iteration.test_validation_passed is True
                    )
                    if test_ok:
                        break

            # Apply in-place if requested
            applied = False
            repo_path = None
            if request.apply and final_patch and request.mode == "auto_fix":
                try:
                    repo_path = await _resolve_repo_path(request.repo, request.audit_api_url)
                    if repo_path:
                        apply_inplace(repo_path, final_patch)
                        applied = True
                except Exception as exc:
                    log.warning("apply_inplace failed: %s", exc)
                    error_msg = f"Patch generated but apply failed: {exc}"

            # Git operations after successful apply
            branch = ""
            commit_sha = ""
            push_url = ""
            git_error = ""
            if applied and request.git_ops and request.git_ops.enabled:
                branch, commit_sha, push_url, git_error = await _run_git_ops(
                    job_id=job_id,
                    final_patch=final_patch,
                    git_ops=request.git_ops,
                    repo=request.repo,
                    audit_api_url=effective_audit_url,
                )

            _update_job(
                job_id,
                status="completed",
                iterations=iterations,
                final_patch=final_patch,
                applied=applied,
                branch=branch,
                commit_sha=commit_sha,
                push_url=push_url,
                git_error=git_error,
                error=error_msg,
                completed_at=_now_iso(),
            )
            yield _sse("complete", {
                "job_id":      job_id,
                "final_patch": final_patch,
                "applied":     applied,
                "branch":      branch,
                "commit_sha":  commit_sha,
                "push_url":    push_url,
                "git_error":   git_error,
                "iterations":  len(iterations),
                "error":       error_msg,
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
) -> tuple[str, str]:
    """Ask the audit API for the raw diff between *compare_branch* and HEAD.

    Returns ``(diff_text, error_message)``.  On success, error_message is "".
    On failure, diff_text is "" and error_message describes exactly what went wrong.

    Uses ``POST /audit/git/diff`` — the audit API is the authority on repo paths
    and git operations; rag-chat has no local access to the ecosystem config.
    """
    import httpx

    if not audit_api_url:
        return "", "No audit_api_url configured — cannot generate diff from branch."

    try:
        async with httpx.AsyncClient(timeout=35) as client:
            r = await client.post(
                f"{audit_api_url.rstrip('/')}/audit/git/diff",
                json={"repo": repo, "compare_branch": compare_branch},
            )
    except httpx.ConnectError:
        return "", f"Audit API unreachable at {audit_api_url}."
    except Exception as exc:
        return "", f"Audit API request failed: {exc}"

    try:
        data = r.json()
    except Exception:
        return "", f"Audit API returned non-JSON (status {r.status_code})."

    if r.status_code == 404:
        return "", data.get("error", f"Repo {repo!r} not found in audit API ecosystem config.")

    if r.status_code != 200 or data.get("error"):
        reason = data.get("reason", "")
        error  = data.get("error", f"Audit API returned status {r.status_code}.")
        # Append a hint for the most common mistake
        if reason == "branch_not_found":
            error += (
                f" Check available branches via GET {audit_api_url}/audit/repos/status"
                f" — the repo may be on a feature branch with no local {compare_branch!r} ref."
            )
        return "", error

    diff_text = data.get("diff", "")
    if data.get("is_empty") or not diff_text.strip():
        return "", (
            f"Diff between {compare_branch!r} and HEAD is empty — "
            f"the branches are identical or {compare_branch!r} is the current HEAD."
        )

    return diff_text, ""


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


async def _run_git_ops(
    *,
    job_id: str,
    final_patch: str,
    git_ops: GitOpsConfig,
    repo: str,
    audit_api_url: str,
) -> tuple[str, str, str, str]:
    """Call the audit API git endpoints after a successful apply.

    Returns (branch, commit_sha, push_url, git_error).
    git_error is non-empty on partial failure (e.g. commit succeeded but push failed);
    the apply is already done at this point so we don't raise.
    """
    import httpx

    base = audit_api_url.rstrip("/")
    branch = ""
    commit_sha = ""
    push_url = ""
    error_parts: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            # 1. Create repair branch
            slug = job_id.replace("repair-", "")
            r = await client.post(
                f"{base}/audit/git/branch",
                json={
                    "repo":        repo,
                    "slug":        slug,
                    "base_branch": git_ops.base_branch,
                },
            )
            if r.status_code == 200:
                branch = r.json().get("branch", "")
                log.info("Git ops: created branch %r for job %s", branch, job_id)
            else:
                error_parts.append(f"branch creation failed ({r.status_code}): {r.text[:200]}")
                # Don't proceed to commit/push if branch creation failed
                return branch, commit_sha, push_url, "; ".join(error_parts)

            # 2. Commit
            if git_ops.commit:
                commit_msg = (
                    f"repair: {slug}\n\n"
                    f"Auto-generated by ai-code-auditor repair loop (job {job_id}).\n"
                    f"Repo: {repo}"
                )
                r = await client.post(
                    f"{base}/audit/git/commit",
                    json={
                        "repo":         repo,
                        "message":      commit_msg,
                        "patch":        final_patch,
                        "author_name":  git_ops.author_name,
                        "author_email": git_ops.author_email,
                    },
                )
                if r.status_code == 200:
                    data = r.json()
                    commit_sha = data.get("sha", "")
                    if data.get("skipped"):
                        log.info("Git ops: nothing to commit for job %s", job_id)
                    else:
                        log.info(
                            "Git ops: committed %s (%d files) for job %s",
                            commit_sha[:12], data.get("files_staged", 0), job_id,
                        )
                else:
                    error_parts.append(f"commit failed ({r.status_code}): {r.text[:200]}")

            # 3. Push (only attempt when commit succeeded)
            if git_ops.push and commit_sha:
                r = await client.post(
                    f"{base}/audit/git/push",
                    json={
                        "repo":   repo,
                        "branch": branch,
                        "remote": git_ops.remote,
                    },
                )
                if r.status_code == 200:
                    data = r.json()
                    if data.get("pushed"):
                        push_url = data.get("remote_url", "")
                        log.info("Git ops: pushed %r → %s", branch, push_url)
                    else:
                        reason = data.get("skipped_reason", "unknown")
                        log.info("Git ops: push skipped — %s", reason)
                        error_parts.append(f"push skipped: {reason}")
                else:
                    error_parts.append(f"push failed ({r.status_code}): {r.text[:200]}")

    except Exception as exc:
        log.warning("Git ops failed for job %s: %s", job_id, exc)
        error_parts.append(str(exc)[:300])

    git_error = "; ".join(error_parts)
    return branch, commit_sha, push_url, git_error

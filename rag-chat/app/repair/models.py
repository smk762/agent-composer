"""Pydantic schemas for the agentic repair pipeline."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class GitOpsConfig(BaseModel):
    """Controls branch creation, committing, and pushing after a successful repair.

    Set ``enabled=True`` to activate.  ``commit`` and ``push`` are independent
    flags so callers can stage without pushing (the default), or skip both and
    just track the branch name for manual handling.
    """
    enabled: bool = False
    commit: bool = Field(True, description="Stage and commit the applied patch.")
    push: bool = Field(False, description="Push the repair branch to the remote after commit.")
    remote: str = Field("origin", description="Git remote to push to.")
    base_branch: str = Field(
        "",
        description="Branch to fork the repair branch from.  Empty = current HEAD.",
    )
    author_name: str = Field("ai-code-auditor", description="Git author name for repair commits.")
    author_email: str = Field(
        "noreply@ai-audit.local",
        description="Git author email for repair commits.",
    )


class RepairRequest(BaseModel):
    repo: str = Field(..., description="Repo name from ai-code-auditor ecosystem config")
    diff: str = Field("", description="Unified diff text to repair")
    compare_branch: str = Field("", description="Generate diff via git diff <branch>...HEAD")
    errors: list[str] = Field(default_factory=list, description="Validation errors to fix")
    mode: Literal["auto_fix", "review", "suggest"] = Field(
        "review",
        description=(
            "auto_fix: apply the best repair automatically; "
            "review: return annotated analysis + proposed patch; "
            "suggest: return improvement suggestions without patching"
        ),
    )
    apply: bool = Field(
        False,
        description="Apply final patch in-place to the repo (requires write access). "
                    "Default: temp copy only — patch is returned but not applied.",
    )
    fix_model: str = Field("", description="Override the fixer (Model A) — defaults to CHAT_MODEL")
    critique_model: str = Field("", description="Critic (Model B) — empty disables critique step")
    adversarial_model: str = Field("", description="Adversary (Model C) — empty disables adversarial step")
    max_iterations: int = Field(3, ge=1, le=10)
    audit_api_url: str = Field(
        "",
        description="Base URL of the ai-code-auditor audit API for validation; "
                    "empty disables validation between iterations.",
    )
    validate_tests: bool = Field(
        True,
        description=(
            "Run the repo's test/lint suite via /audit/validate_patch after each "
            "iteration (in addition to the static diff-audit).  Requires audit_api_url. "
            "Failures are fed back as compact error lines optimised for local model context windows."
        ),
    )
    git_ops: Optional[GitOpsConfig] = Field(
        None,
        description=(
            "When set with enabled=True, create a repair branch, commit, and optionally push "
            "after a successful auto_fix apply.  Ignored for review and suggest modes."
        ),
    )


class RepairIteration(BaseModel):
    iteration: int
    fixer_prompt: str = ""
    fixer_response: str = ""
    patch: str = ""                 # extracted unified diff from fixer response
    critic_prompt: str = ""
    critic_response: str = ""
    adversary_prompt: str = ""
    adversary_response: str = ""
    validation_passed: Optional[bool] = None        # diff-audit (static analysis)
    validation_errors: list[str] = Field(default_factory=list)
    test_validation_passed: Optional[bool] = None   # test/lint suite on patched copy
    test_validation_errors: list[str] = Field(default_factory=list)


class RepairResult(BaseModel):
    job_id: str
    repo: str
    status: Literal["pending", "running", "completed", "failed"]
    mode: str
    iterations: list[RepairIteration] = Field(default_factory=list)
    final_patch: str = ""
    applied: bool = False
    branch: str = ""        # repair branch created (if git_ops.enabled)
    commit_sha: str = ""    # commit hash after apply (if git_ops.commit)
    push_url: str = ""      # remote URL pushed to (if git_ops.push and push succeeded)
    git_error: str = ""     # non-fatal git ops error (apply still succeeded)
    error: str = ""
    created_at: str = ""
    completed_at: str = ""

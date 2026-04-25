"""Pydantic schemas for the agentic repair pipeline."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


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


class RepairIteration(BaseModel):
    iteration: int
    fixer_prompt: str = ""
    fixer_response: str = ""
    patch: str = ""                 # extracted unified diff from fixer response
    critic_prompt: str = ""
    critic_response: str = ""
    adversary_prompt: str = ""
    adversary_response: str = ""
    validation_passed: Optional[bool] = None
    validation_errors: list[str] = Field(default_factory=list)


class RepairResult(BaseModel):
    job_id: str
    repo: str
    status: Literal["pending", "running", "completed", "failed"]
    mode: str
    iterations: list[RepairIteration] = Field(default_factory=list)
    final_patch: str = ""
    applied: bool = False
    error: str = ""
    created_at: str = ""
    completed_at: str = ""

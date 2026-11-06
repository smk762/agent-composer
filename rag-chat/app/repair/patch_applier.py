"""Apply a unified diff to a temporary repo copy for safe validation.

The real working tree is never modified unless the caller explicitly passes
apply_inplace=True (requires write access to the repo path).
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

log = logging.getLogger("rag-chat.repair")


def extract_patch(llm_response: str) -> str:
    """Pull a unified diff out of a free-form LLM response.

    Tries (in order):
    1. Fenced code block:  ```diff ... ```  or  ``` ... ```
    2. First line that starts ``diff --git``, ``---``, or ``@@ ``
    3. Empty string if nothing plausible is found.
    """
    # Fenced block
    m = re.search(r"```(?:diff|patch)?\n(.*?)```", llm_response, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        if _looks_like_diff(candidate):
            return candidate

    # Inline diff — find the first diff-looking line
    lines = llm_response.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(("diff --git ", "--- ", "@@ ")):
            return "\n".join(lines[i:]).strip()

    return ""


def _looks_like_diff(text: str) -> bool:
    return bool(
        text.startswith(("---", "diff --git ", "@@ "))
        or re.search(r"^@@\s+-\d+", text, re.MULTILINE)
    )


def apply_to_temp(
    repo_path: Path,
    patch_text: str,
) -> tuple[Path, Callable[[], None]]:
    """Copy the repo to a temp directory and apply *patch_text* via ``git apply``.

    Args:
        repo_path: original repository root (may be read-only SSHFS).
        patch_text: unified diff string.

    Returns:
        ``(patched_copy_path, cleanup)`` — call ``cleanup()`` when done.

    Raises:
        RuntimeError: if the patch cannot be applied cleanly.
    """
    tmp_root = Path(tempfile.mkdtemp(prefix="ai-repair-"))

    def cleanup() -> None:
        shutil.rmtree(tmp_root, ignore_errors=True)

    try:
        repo_copy = tmp_root / "repo"
        shutil.copytree(str(repo_path), str(repo_copy), symlinks=True)

        patch_file = tmp_root / "changes.patch"
        patch_file.write_text(patch_text, encoding="utf-8")

        result = subprocess.run(
            ["git", "apply", "--whitespace=fix", str(patch_file)],
            cwd=repo_copy,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            err = result.stderr.strip()[:600]
            raise RuntimeError(f"git apply failed: {err}")

        return repo_copy, cleanup

    except Exception:
        cleanup()
        raise


def apply_inplace(repo_path: Path, patch_text: str) -> None:
    """Apply *patch_text* directly to *repo_path* using ``git apply``.

    Raises:
        PermissionError: if the path is not writable.
        RuntimeError: if the patch does not apply cleanly.
    """
    _check_writable(repo_path)

    patch_file = repo_path / ".ai_repair_patch.tmp"
    try:
        patch_file.write_text(patch_text, encoding="utf-8")
        result = subprocess.run(
            ["git", "apply", "--whitespace=fix", str(patch_file)],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
    finally:
        patch_file.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"git apply failed: {result.stderr.strip()[:600]}")


def _check_writable(path: Path) -> None:
    probe = path / ".ai_repair_write_probe"
    try:
        probe.touch()
        probe.unlink()
    except (PermissionError, OSError) as exc:
        raise PermissionError(f"Repo path is not writable: {path}") from exc

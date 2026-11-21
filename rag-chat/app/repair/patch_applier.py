"""Apply a unified diff to a temporary repo copy for safe validation.

The real working tree is never modified unless the caller explicitly passes
apply_inplace=True (requires write access to the repo path).
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

log = logging.getLogger("rag-chat.repair")


# An opening fence at the start of a line, with an optional info string, then the body up
# to a closing fence alone on its line (a diff line never starts with ```).
_FENCE_RE = re.compile(r"^```[ \t]*([\w+-]*)[^\n]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)

# Lines a unified / git diff can contain once it has started.
_DIFF_LINE_PREFIXES = (
    " ", "+", "-", "\\", "@@",
    "diff --git ", "index ", "new file mode", "deleted file mode", "old mode", "new mode",
    "similarity index", "dissimilarity index", "rename from", "rename to", "copy from", "copy to",
    "Binary files ",
)


def extract_patch(llm_response: str) -> str:
    """Pull a unified diff out of a free-form LLM response.

    Tries (in order):
    1. Each fenced code block (```diff ... ```, ``` ... ```, ...), first one that looks like a diff
    2. Inline: from the first line that starts ``diff --git``, ``---`` or ``@@ `` up to the
       first line that is not diff syntax (prose, a closing fence)
    3. Empty string if nothing plausible is found.

    A found patch ends with a newline. Only blank lines are trimmed around it: a hunk's
    last context line may be a lone space, and dropping it would corrupt the hunk.
    """
    llm_response = llm_response.replace("\r\n", "\n")
    for m in _FENCE_RE.finditer(llm_response):
        candidate = m.group(2).strip("\n")
        if _looks_like_diff(candidate):
            return candidate + "\n"

    # Inline diff — find the first diff-looking line, stop where the diff ends
    # Only "\n": splitlines() also breaks on \f, \v, \x1c-\x1e, \x85, \u2028/9, which can
    # sit inside a source line and would split a hunk line in two.
    lines = llm_response.split("\n")
    for i, line in enumerate(lines):
        if line.startswith(("diff --git ", "--- ", "@@ ")):
            end = i + 1
            while end < len(lines) and _is_diff_line(lines[end]):
                end += 1
            while end > i + 1 and lines[end - 1] == "":
                end -= 1
            return "\n".join(lines[i:end]) + "\n"

    return ""


def _looks_like_diff(text: str) -> bool:
    return bool(
        text.startswith(("---", "diff --git ", "@@ "))
        or re.search(r"^@@\s+-\d+", text, re.MULTILINE)
    )


def _is_diff_line(line: str) -> bool:
    if line.startswith("```"):
        return False
    return line == "" or line.startswith(_DIFF_LINE_PREFIXES)


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

    The patch goes to git on stdin: nothing is written inside the working tree, where a
    committed symlink at a fixed name could redirect the write.

    Raises:
        PermissionError: if the path is not writable.
        RuntimeError: if the patch does not apply cleanly.
    """
    _check_writable(repo_path)

    result = subprocess.run(
        ["git", "apply", "--whitespace=fix", "-"],
        cwd=repo_path,
        input=patch_text,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git apply failed: {result.stderr.strip()[:600]}")


def _check_writable(path: Path) -> None:
    # mkstemp creates a fresh file (O_EXCL), so a planted symlink can't redirect the probe.
    try:
        fd, probe = tempfile.mkstemp(prefix=".ai_repair_write_probe", dir=path)
        os.close(fd)
        os.unlink(probe)
    except (PermissionError, OSError) as exc:
        raise PermissionError(f"Repo path is not writable: {path}") from exc

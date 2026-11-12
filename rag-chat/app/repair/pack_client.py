"""Fetch ai-code-auditor agent coding packs and turn them into repair context.

A *pack* is the compressed work item the auditor emits for a coding agent
(``GET /audit/packs/{id}`` JSON, ``GET /audit/packs/{id}/brief`` markdown).
Driving the repair loop by ``pack_id`` means the caller no longer has to paste
the brief into the request.

Every network call here has a hard timeout and fails soft: on any error the
caller gets a :class:`PackContext` with ``warning`` set and an empty ``block``,
and the repair loop continues without pack context.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlparse

import httpx

log = logging.getLogger("rag-chat.repair")

# Pack ids look like ``pack-world-ecosystem`` / ``ext-retry-core-001``.  Keep the
# charset tight: the id is interpolated into a URL path.
_PACK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")

_BRIEF_CAP = 3_000  # chars — the brief repeats some key fields; keep it bounded
_LIST_CAP = 12      # max bullets per list section


@dataclass
class PackContext:
    """Result of resolving a pack for a repair run."""

    pack_id: str
    ok: bool = False
    title: str = ""
    repo: str = ""
    kind: str = ""
    block: str = ""            # markdown context block (empty when fetch failed)
    warning: str = ""          # non-fatal problem to surface to the caller
    acceptance_criteria: list[str] = field(default_factory=list)
    suggested_test_commands: list[str] = field(default_factory=list)


def validate_pack_id(pack_id: str) -> bool:
    return bool(_PACK_ID_RE.match(pack_id or ""))


def normalize_base_url(base: str) -> str:
    """Return a trimmed http(s) base URL, or "" when *base* is unusable."""
    base = (base or "").strip().rstrip("/")
    parsed = urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    return base


def _as_str_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    out: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = item.get("rule_name") or item.get("name") or ""
            evidence = item.get("evidence") or ""
            item = f"{name}: {evidence}".strip(": ") if (name or evidence) else str(item)
        out.append(str(item))
    return out


def _bullets(items: list[str]) -> str:
    shown = items[:_LIST_CAP]
    text = "\n".join(f"- {i}" for i in shown)
    if len(items) > _LIST_CAP:
        text += f"\n- ... ({len(items) - _LIST_CAP} more)"
    return text


def build_pack_block(pack: dict[str, Any], brief: str = "") -> PackContext:
    """Render the repair-context block from a pack JSON dict (+ optional brief)."""
    narrative = pack.get("narrative") or {}
    repo = pack.get("repo") or {}
    loc = pack.get("location") or {}
    verify = pack.get("verification") or {}
    governance = pack.get("governance") or {}
    priority = pack.get("priority") or {}

    pack_id = str(pack.get("work_item_id") or "")
    title = str(narrative.get("title") or pack_id)
    repo_name = str(repo.get("name") or "")
    kind = str(pack.get("kind") or "")
    criteria = _as_str_list(narrative.get("acceptance_criteria"))
    commands = _as_str_list(verify.get("suggested_test_commands"))
    constraints = _as_str_list(governance.get("architecture_rules_hit"))

    where = str(loc.get("file_path") or "")
    if where and loc.get("line"):
        where += f":{loc['line']}"
    if where and loc.get("symbol"):
        where += f" ({loc['symbol']})"

    lines = [
        "## Agent work pack",
        "",
        f"- **id:** `{pack_id}`",
        f"- **title:** {title}",
    ]
    if kind:
        lines.append(f"- **kind:** {kind}")
    if priority.get("severity"):
        lines.append(f"- **severity:** {priority['severity']}")
    if repo_name:
        lines.append(f"- **repo:** {repo_name}")
    if where:
        lines.append(f"- **location:** `{where}`")
    recommendation = narrative.get("recommendation")
    if recommendation:
        lines += ["", "### Recommendation", str(recommendation)[:1_500]]

    lines += [
        "",
        "### Acceptance criteria",
        _bullets(criteria) if criteria else "- Fix the finding without regressing related paths.",
    ]
    if commands:
        lines += ["", "### Suggested test commands", _bullets([f"`{c}`" for c in commands])]
    if constraints:
        lines += ["", "### Constraints", _bullets(constraints)]

    snippet = loc.get("code_snippet") or ""
    if snippet:
        lines += ["", "### Code snippet", "```", str(snippet)[:2_000], "```"]

    if brief.strip():
        text = brief.strip()
        if len(text) > _BRIEF_CAP:
            text = text[:_BRIEF_CAP] + "\n... (truncated)"
        lines += ["", "### Pack brief", "", text]

    return PackContext(
        pack_id=pack_id,
        ok=True,
        title=title,
        repo=repo_name,
        kind=kind,
        block="\n".join(lines),
        acceptance_criteria=criteria,
        suggested_test_commands=commands,
    )


def resolve_repo(request_repo: str, pack_repo: str, *, write_ops: bool) -> tuple[str, str, str]:
    """Decide which repo a pack-driven repair targets.

    Returns ``(repo, warning, error)``.

    - pack has no repo name -> keep the request's repo.
    - request repo empty    -> adopt the pack's repo.
    - repos match (case-insensitive) -> unchanged.
    - mismatch, read-only run (no apply / git_ops) -> prefer the pack's repo + warning.
    - mismatch, write run (apply / commit / push) -> error: never write to a repo
      the caller did not clearly intend.
    """
    if not pack_repo:
        return request_repo, "", ""
    if not request_repo:
        return pack_repo, "", ""
    if request_repo.lower() == pack_repo.lower():
        return request_repo, "", ""
    if write_ops:
        return request_repo, "", (
            f"Pack targets repo {pack_repo!r} but request repo is {request_repo!r}; "
            "refusing to apply/commit. Fix 'repo' or drop apply/git_ops."
        )
    return pack_repo, (
        f"Request repo {request_repo!r} differs from pack repo {pack_repo!r}; "
        f"using the pack's repo (read-only run)."
    ), ""


async def fetch_pack_context(
    pack_id: str,
    audit_api_base: str,
    *,
    timeout_s: float = 10.0,
) -> PackContext:
    """GET the pack (+ brief) from the audit API.  Never raises."""
    if not validate_pack_id(pack_id):
        return PackContext(pack_id=pack_id, warning=f"pack_id {pack_id!r} is not a valid pack id; ignoring.")
    base = normalize_base_url(audit_api_base)
    if not base:
        return PackContext(
            pack_id=pack_id,
            warning=f"audit_api_base {audit_api_base!r} is not an http(s) URL; pack context skipped.",
        )

    url = f"{base}/audit/packs/{quote(pack_id, safe='')}"
    pack: dict[str, Any] | None = None
    brief = ""
    warning = ""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
            r = await client.get(url)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict):
                    pack = data
                else:
                    warning = f"pack {pack_id!r}: unexpected JSON shape from {url}"
            elif r.status_code == 404:
                return PackContext(pack_id=pack_id, warning=f"pack {pack_id!r} not found at {base}")
            else:
                warning = f"pack fetch for {pack_id!r} returned HTTP {r.status_code}"

            # Brief is best-effort and independent of the JSON fetch.
            try:
                rb = await client.get(f"{url}/brief")
                if rb.status_code == 200:
                    brief = rb.text
            except Exception as exc:  # noqa: BLE001 - fail soft
                log.debug("pack brief fetch failed for %s: %s", pack_id, exc)
    except Exception as exc:  # noqa: BLE001 - fail soft (timeout, connect, bad JSON)
        log.warning("pack fetch failed for %s: %s", pack_id, exc)
        return PackContext(pack_id=pack_id, warning=f"pack fetch for {pack_id!r} failed: {exc}")

    if pack is not None:
        ctx = build_pack_block(pack, brief)
        if not ctx.pack_id:
            ctx.pack_id = pack_id
        return ctx
    if brief.strip():
        # JSON unavailable but the markdown brief came through — still useful.
        return PackContext(
            pack_id=pack_id,
            ok=True,
            title=pack_id,
            block="## Agent work pack\n\n- **id:** `" + pack_id + "`\n\n### Pack brief\n\n" + brief.strip()[:_BRIEF_CAP],
            warning=warning,
        )
    return PackContext(pack_id=pack_id, warning=warning or f"pack {pack_id!r} unavailable")


# ── Status callback (not wired by default) ────────────────────────────────────

async def report_pack_status(
    pack_id: str,
    audit_api_base: str,
    status: str,
    *,
    detail: dict[str, Any] | None = None,
    timeout_s: float = 10.0,
) -> str:
    """POST ``{base}/audit/packs/{id}/status``.  Returns "" on success, else a reason.

    TODO(worker-a): ``POST /audit/packs/{id}/status`` does not exist in
    ai-code-auditor yet.  Confirm the request contract once it lands, then
    enable via ``REPAIR_PACK_STATUS_CALLBACK=1``.  A 404/405 is reported as
    "endpoint unavailable" and never fails the repair job.
    """
    base = normalize_base_url(audit_api_base)
    if not base or not validate_pack_id(pack_id):
        return "invalid pack id or audit_api_base"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
            r = await client.post(
                f"{base}/audit/packs/{quote(pack_id, safe='')}/status",
                json={"status": status, **(detail or {})},
            )
    except Exception as exc:  # noqa: BLE001 - fail soft
        return f"status callback failed: {exc}"
    if r.status_code in (404, 405):
        return "status endpoint unavailable"
    if r.status_code >= 300:
        return f"status callback returned HTTP {r.status_code}"
    return ""

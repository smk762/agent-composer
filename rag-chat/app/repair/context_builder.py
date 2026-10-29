"""Build the prompt context block for each repair iteration.

Context priority (highest to lowest):
  1. The unified diff itself (always present)
  2. Validation / compilation errors
  3. Qdrant semantic hits from the code_audit collection (best-effort)
  4. Critic / adversary notes from the previous iteration
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger("rag-chat.repair")

_DIFF_CAP = 12_000     # chars — cap raw diff to avoid blowing context
_SNIPPET_CAP = 4_000   # chars — cap qdrant snippets
_ERROR_CAP = 2_000     # chars — cap error block


def build_repair_context(
    diff: str,
    errors: Optional[list[str]] = None,
    critic_notes: str = "",
    adversary_notes: str = "",
    qdrant_snippets: Optional[list[str]] = None,
) -> str:
    """Return a markdown context block that prefaces each repair prompt."""
    parts: list[str] = []

    # 1. Diff
    diff_body = diff[:_DIFF_CAP]
    if len(diff) > _DIFF_CAP:
        diff_body += f"\n... (truncated — {len(diff) - _DIFF_CAP} chars omitted)"
    parts.append("## Diff under review\n\n```diff\n" + diff_body + "\n```")

    # 2. Validation / compilation errors
    if errors:
        err_text = "\n".join(f"- {e}" for e in errors)
        if len(err_text) > _ERROR_CAP:
            err_text = err_text[:_ERROR_CAP] + "\n... (truncated)"
        parts.append("## Validation errors\n\n" + err_text)

    # 3. Qdrant semantic hits
    if qdrant_snippets:
        combined = "\n\n---\n\n".join(qdrant_snippets)
        if len(combined) > _SNIPPET_CAP:
            combined = combined[:_SNIPPET_CAP] + "\n... (truncated)"
        parts.append("## Related code (semantic search)\n\n" + combined)

    # 4. Previous iteration notes
    if critic_notes:
        parts.append("## Critic's notes (previous iteration)\n\n" + critic_notes[:2000])
    if adversary_notes:
        parts.append("## Adversary's objections (previous iteration)\n\n" + adversary_notes[:2000])

    return "\n\n".join(parts)


async def fetch_qdrant_snippets(
    query: str,
    qdrant_url: str,
    collection: str = "code_audit",
    top_k: int = 5,
) -> list[str]:
    """Query Qdrant's code_audit collection and return text snippets.

    Returns an empty list on any error so the repair loop degrades gracefully.
    """
    try:
        import httpx
        from app.routers.chat import ollama_embed

        vector = await ollama_embed(query[:500])
        payload = {
            "vector": {"name": "dense", "vector": vector},
            "limit": top_k,
            "with_payload": True,
        }
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{qdrant_url}/collections/{collection}/points/search",
                json=payload,
            )
        if r.status_code != 200:
            return []
        hits = r.json().get("result", [])
        snippets: list[str] = []
        for hit in hits:
            p = hit.get("payload", {})
            text = p.get("text") or p.get("content") or p.get("raw_text") or ""
            if text:
                snippets.append(text[:600])
        return snippets
    except Exception as exc:
        log.debug("Qdrant fetch skipped: %s", exc)
        return []

"""Related-code lookup for the repair loop (#35).

Searches rag-ingest's code collection (``QDRANT_COLLECTION_CODE``) for code related to the
change under repair. The collection is written by Infinity + ``EMBED_MODEL_CODE`` (label
``CODE_EMBEDDER``), so the query is embedded the same way, with the model's query
instruction (``CODE_QUERY_PREFIX``). Before searching, the collection's ``embedder`` labels
are checked: a collection written by another embedder yields no snippets and a warning,
never snippets ranked by incomparable vectors.

Best effort throughout: a missing collection, an unreachable Qdrant or Infinity just means
no snippets (``fetch_code_snippets`` never raises).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app import retrieval
from app.config import (
    CODE_EMBEDDER,
    CODE_QUERY_PREFIX,
    EMBED_MODEL_CODE,
    INFINITY_URL,
    QDRANT_COLLECTION_CODE,
    QDRANT_URL,
)

log = logging.getLogger("rag-chat.repair")

EmbedFn = Callable[[str], Awaitable[List[float]]]

_QUERY_CAP = 1000
_SNIPPET_CHARS = 600
_DEF_RE = re.compile(r"^[+-]\s*(?:export\s+)?(?:async\s+)?(?:def|class|function|func|fn|interface|struct|type)\s+([A-Za-z_]\w*)", re.M)
_PATH_RE = re.compile(r"^(?:\+\+\+ b/|--- a/)(\S+)", re.M)
_HUNK_CTX_RE = re.compile(r"^@@[^@]*@@\s*(\S.*)$", re.M)
_PACK_LOCATION_RE = re.compile(r"\*\*location:\*\*\s*`([^`]+)`")
_PACK_TITLE_RE = re.compile(r"^#+\s*Agent work pack[^\n]*?:\s*(.+)$|\*\*title:\*\*\s*(.+)$", re.M)


@dataclass
class CodeSnippets:
    snippets: List[str]
    warning: str = ""  # non-fatal problem worth telling the caller (a mismatched collection)


def _unique(items: List[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        item = item.strip()
        if item and item not in out:
            out.append(item)
    return out


def code_query(diff: str = "", pack_context: str = "", pack_title: str = "") -> str:
    """A retrieval query for the change: paths, symbols and a few changed lines, not raw diff text.

    From a diff: the touched files, the enclosing functions from hunk headers, symbols the diff
    defines or removes, then the first changed lines. Without a diff: the pack's title and location."""
    parts: List[str] = []
    if diff:
        paths = _unique([p for p in _PATH_RE.findall(diff) if p != "/dev/null"])
        symbols = _unique(_DEF_RE.findall(diff) + _HUNK_CTX_RE.findall(diff))
        changed = [line[1:].strip() for line in diff.splitlines()
                   if line[:1] in "+-" and not line.startswith(("+++", "---")) and line[1:].strip()]
        if paths:
            parts.append("files: " + ", ".join(paths[:10]))
        if symbols:
            parts.append("symbols: " + ", ".join(symbols[:15]))
        if changed:
            parts.append("\n".join(changed[:12]))
    if not parts and (pack_context or pack_title):
        title = pack_title or next((g for m in _PACK_TITLE_RE.finditer(pack_context) for g in m.groups() if g), "")
        location = _PACK_LOCATION_RE.search(pack_context or "")
        parts = [p for p in (title.strip(), f"files: {location.group(1)}" if location else "") if p]
    return "\n".join(parts)[:_QUERY_CAP]


async def embed_code_query(text: str) -> List[float]:
    """Embed a code-search query the way rag-ingest embedded the code: Infinity + EMBED_MODEL_CODE."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{INFINITY_URL}/embeddings",
                              json={"model": EMBED_MODEL_CODE, "input": [CODE_QUERY_PREFIX + text]})
    r.raise_for_status()
    data = r.json().get("data") or []
    if not data or not data[0].get("embedding"):
        raise RuntimeError("Infinity returned no embedding")
    return [float(x) for x in data[0]["embedding"]]


def _format(hit: Any) -> Optional[str]:
    p = hit.payload or {}
    text = p.get("text") or ""
    if not isinstance(text, str) or not text.strip():
        return None
    where = "/".join(x for x in (p.get("repo"), p.get("path")) if x) or "unknown file"
    if p.get("start_line"):
        where += f":{p['start_line']}" + (f"-{p['end_line']}" if p.get("end_line") else "")
    if p.get("symbol"):
        where += f" ({p['symbol']})"
    return f"`{where}`\n```\n{text.strip()[:_SNIPPET_CHARS]}\n```"


async def fetch_code_snippets(query: str, *, repo: Optional[str] = None, top_k: int = 5,
                              collection: str = QDRANT_COLLECTION_CODE, client: Any = None,
                              embed: Optional[EmbedFn] = None) -> CodeSnippets:
    """Up to ``top_k`` related-code snippets for ``query`` (only from ``repo`` when given)."""
    if not query.strip():
        return CodeSnippets([])
    qc = client or AsyncQdrantClient(url=QDRANT_URL)
    try:
        status = await retrieval.embedder_status(collection=collection, client=qc, expected=CODE_EMBEDDER)
        if status["ok"] is None:  # no collection / Qdrant down: the normal state before code is ingested
            log.debug("code search skipped: %s", status.get("error"))
            return CodeSnippets([])
        if status["ok"] is False:
            return CodeSnippets([], warning="Related-code search skipped: " + retrieval.mismatch_message(status))
        try:
            vector = await (embed or embed_code_query)(query)
        except Exception as exc:
            return CodeSnippets([], warning=f"Related-code search skipped: embedding the query failed ({exc})")
        flt = qm.Filter(must=[qm.FieldCondition(key="repo", match=qm.MatchValue(value=repo))]) if repo else None
        hits = await qc.search(collection_name=collection, query_vector=qm.NamedVector(name="dense", vector=vector),
                               query_filter=flt, limit=top_k, with_payload=True)
        return CodeSnippets([s for s in (_format(h) for h in hits or []) if s])
    except Exception as exc:
        log.debug("code search skipped: %s", exc)
        return CodeSnippets([])
    finally:
        if client is None:
            await qc.close()

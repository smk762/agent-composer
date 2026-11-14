"""Re-embed a text collection into another one under a different embedder setting (#32).

    python -m app.reembed --src audit_docs --dst audit_docs_v2 --prefixes nomic [--dry-run]

Every text point stores the chunk it was embedded from (``text``), so a collection can be
rebuilt without the original sender: each source point is re-embedded with Ollama +
``EMBED_MODEL`` under ``--prefixes`` and written to ``--dst`` with the same id and payload,
relabelled ``embedder``.

The run is a *sync*, so it can be repeated while the source keeps receiving writes:

* a point missing from dst, or whose ``text`` differs, is re-embedded and upserted;
* a point whose text matches but whose payload differs (tags, ``ingested_at``) only has
  its payload overwritten (no re-embedding);
* a dst point whose id is no longer in src is deleted.

dst must hold nothing but this embedder's points (strict check, unlabelled points included),
so a sync can't mix vector spaces. Run it once to fill dst, then once more right after the
cut-over so the writes that reached src in between are carried across.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app.embeddings import EMBED_TASK_PREFIXES, PREFIX_SCHEMES, OllamaEmbeddingClient, doc_prefix, text_embedder_label
from app.ingest_pipeline import (
    QDRANT_URL,
    EmbedderMismatchError,
    _check_embedder,
    _collection_exists,
    ensure_legacy_collection,
)

_PAGE = 256


async def _scroll_all(q: AsyncQdrantClient, collection: str) -> Dict[Any, Dict[str, Any]]:
    """id -> payload for every point of ``collection`` (no vectors)."""
    out: Dict[Any, Dict[str, Any]] = {}
    offset = None
    while True:
        pts, offset = await q.scroll(collection_name=collection, limit=_PAGE, offset=offset,
                                     with_payload=True, with_vectors=False)
        for p in pts:
            out[p.id] = dict(p.payload or {})
        if offset is None:
            return out


def _without_label(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if k != "embedder"}


async def sync(src: str, dst: str, *, scheme: str, dry_run: bool = False, batch: int = 64,
               client: Optional[AsyncQdrantClient] = None, embedder: Any = None,
               log=print) -> Dict[str, Any]:
    if src == dst:
        raise ValueError("--src and --dst must differ")
    label = text_embedder_label(scheme)
    emb = embedder or OllamaEmbeddingClient(prefix=doc_prefix(scheme))
    q = client or AsyncQdrantClient(url=QDRANT_URL)
    started = time.monotonic()
    try:
        if not await _collection_exists(q, src):
            raise ValueError(f"source collection {src!r} does not exist")
        src_points = await _scroll_all(q, src)
        dst_exists = await _collection_exists(q, dst)
        if dst_exists:
            await _check_embedder(q, dst, label, strict=True)
            dst_points = await _scroll_all(q, dst)
        else:
            dst_points = {}

        to_embed = [pid for pid, p in src_points.items()
                    if pid not in dst_points or dst_points[pid].get("text") != p.get("text")]
        to_relabel = [pid for pid, p in src_points.items()
                      if pid in dst_points and pid not in to_embed
                      and _without_label(dst_points[pid]) != _without_label(p)]
        to_delete = [pid for pid in dst_points if pid not in src_points]
        plan = {"src": src, "dst": dst, "embedder": label, "src_points": len(src_points),
                "dst_points": len(dst_points), "embed": len(to_embed), "payload_only": len(to_relabel),
                "delete": len(to_delete)}
        log(json.dumps({"plan": plan}))
        if dry_run:
            return {**plan, "dry_run": True}

        failed: List[Any] = []
        written = 0
        pending: List[qm.PointStruct] = []

        async def flush() -> None:
            nonlocal written
            if pending:
                await q.upsert(collection_name=dst, points=list(pending), wait=True)
                written += len(pending)
                pending.clear()

        for i, pid in enumerate(to_embed, 1):
            payload = src_points[pid]
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                failed.append(pid)
                continue
            try:
                res = await emb.embed([text], profile="text")
            except Exception as e:  # keep going; report at the end
                log(json.dumps({"failed": str(pid), "error": str(e)[:200]}))
                failed.append(pid)
                continue
            vector = res.vectors[0]
            if not dst_exists:
                await ensure_legacy_collection(q, vector_size=len(vector), name=dst)
                dst_exists = True
            pending.append(qm.PointStruct(id=pid, vector=vector, payload={**payload, "embedder": label}))
            if len(pending) >= batch:
                await flush()
            if i % 100 == 0:
                log(json.dumps({"progress": i, "of": len(to_embed)}))
        await flush()

        for pid in to_relabel:
            await q.overwrite_payload(collection_name=dst, payload={**src_points[pid], "embedder": label},
                                      points=[pid], wait=True)
        if to_delete:
            await q.delete(collection_name=dst, points_selector=qm.PointIdsList(points=to_delete), wait=True)

        result = {**plan, "embedded": written, "payload_updated": len(to_relabel), "deleted": len(to_delete),
                  "failed": len(failed), "seconds": round(time.monotonic() - started, 1)}
        log(json.dumps({"result": result}))
        return result
    finally:
        if client is None:
            await q.close()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.reembed", description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="collection to read (its points' stored text)")
    ap.add_argument("--dst", required=True, help="collection to fill/sync (created if missing)")
    ap.add_argument("--prefixes", default=EMBED_TASK_PREFIXES, choices=sorted(PREFIX_SCHEMES),
                    help="task-prefix scheme to embed dst with (default: EMBED_TASK_PREFIXES)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args(argv)
    try:
        result = asyncio.run(sync(args.src, args.dst, scheme=args.prefixes, dry_run=args.dry_run, batch=args.batch))
    except (ValueError, EmbedderMismatchError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 1 if result.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())

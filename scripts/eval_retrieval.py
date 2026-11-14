"""Compare retrieval quality of two text collections (e.g. before/after an embedder change, #32).

Runs where rag-chat's environment is (OLLAMA_URL, QDRANT_URL, EMBED_MODEL, CHAT_MODEL):

    docker compose exec -T rag-chat python - \
        --baseline audit_docs:none --candidate audit_docs_v2:nomic --n 80 < scripts/eval_retrieval.py

Method (synthetic QA retrieval):
1. Sample ``--n`` chunks (seeded) from the baseline collection, skipping very short ones.
2. Ask the chat model for one question each chunk answers, phrased without copying it.
3. For each collection, embed the question under that collection's prefix scheme
   (``none`` or ``nomic`` -> "search_query: ") and search the top 10.
4. Score the rank of the source chunk (same point id; the re-embed sync keeps ids) and of
   the first hit from the same doc. Report recall@1/5/10 and MRR@10 for both, plus wins/losses.

Questions are generated from the chunks themselves, so they share some vocabulary with them;
that bias is the same for both collections, which is what a comparison needs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from qdrant_client import AsyncQdrantClient

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("EVAL_QUESTION_MODEL") or os.getenv("CHAT_MODEL", "qwen3:8b")
QUERY_PREFIX = {"none": "", "nomic": "search_query: "}
K = 10

QUESTION_PROMPT = """Here is a passage from internal engineering documentation.

Write ONE question a developer might ask that this passage answers. Use your own words: do not
copy phrases or headings from the passage, and do not mention "the passage". Reply with the
question only.

Passage:
{text}"""


async def generate_question(client: httpx.AsyncClient, text: str) -> str:
    r = await client.post(f"{OLLAMA_URL}/api/chat", json={
        "model": CHAT_MODEL, "stream": False, "think": False,
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": QUESTION_PROMPT.format(text=text[:2500])}],
    })
    r.raise_for_status()
    out = (r.json().get("message") or {}).get("content", "")
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().strip('"')
    return out.splitlines()[0].strip() if out else ""


async def embed(client: httpx.AsyncClient, text: str) -> List[float]:
    r = await client.post(f"{OLLAMA_URL}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    return [float(x) for x in r.json()["embedding"]]


def _ranks(hits: List[Any], pid: Any, doc_id: Optional[str]) -> Dict[str, Optional[int]]:
    chunk = next((i for i, h in enumerate(hits, 1) if h.id == pid), None)
    doc = next((i for i, h in enumerate(hits, 1) if doc_id and (h.payload or {}).get("doc_id") == doc_id), None)
    return {"chunk": chunk, "doc": doc}


def _metrics(ranks: List[Optional[int]]) -> Dict[str, float]:
    n = len(ranks) or 1
    return {
        "recall@1": round(sum(1 for r in ranks if r and r <= 1) / n, 3),
        "recall@5": round(sum(1 for r in ranks if r and r <= 5) / n, 3),
        "recall@10": round(sum(1 for r in ranks if r and r <= 10) / n, 3),
        "mrr@10": round(sum(1 / r for r in ranks if r) / n, 3),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="collection:scheme, e.g. audit_docs:none")
    ap.add_argument("--candidate", required=True, help="collection:scheme, e.g. audit_docs_v2:nomic")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--details", action="store_true", help="include per-question rows")
    args = ap.parse_args()
    arms = []
    for spec in (args.baseline, args.candidate):
        coll, _, scheme = spec.partition(":")
        if scheme not in QUERY_PREFIX:
            raise SystemExit(f"unknown scheme in {spec!r}; use none or nomic")
        arms.append((coll, scheme))

    q = AsyncQdrantClient(url=QDRANT_URL)
    started = time.monotonic()
    try:
        points, offset = [], None
        while True:
            page, offset = await q.scroll(arms[0][0], limit=256, offset=offset, with_payload=True, with_vectors=False)
            points += [p for p in page if len(str((p.payload or {}).get("text") or "")) >= args.min_chars]
            if offset is None:
                break
        random.Random(args.seed).shuffle(points)
        sample = points[: args.n]

        rows: List[Dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=180) as http:
            for p in sample:
                payload = p.payload or {}
                question = await generate_question(http, payload["text"])
                if not question:
                    continue
                row: Dict[str, Any] = {"id": str(p.id), "doc_id": payload.get("doc_id"),
                                       "title": (payload.get("meta") or {}).get("title"), "question": question}
                for coll, scheme in arms:
                    vec = await embed(http, QUERY_PREFIX[scheme] + question)
                    hits = await q.search(collection_name=coll, query_vector=vec, limit=K, with_payload=["doc_id"])
                    row[coll] = _ranks(hits, p.id, payload.get("doc_id"))
                rows.append(row)

        (b, _), (c, _) = arms
        report: Dict[str, Any] = {
            "baseline": args.baseline, "candidate": args.candidate, "questions": len(rows),
            "pool": len(points), "seed": args.seed, "question_model": CHAT_MODEL, "embed_model": EMBED_MODEL,
            "seconds": round(time.monotonic() - started, 1),
        }
        for level in ("chunk", "doc"):
            report[level] = {coll: _metrics([r[coll][level] for r in rows]) for coll in (b, c)}
            better = sum(1 for r in rows if (r[c][level] or K + 1) < (r[b][level] or K + 1))
            worse = sum(1 for r in rows if (r[c][level] or K + 1) > (r[b][level] or K + 1))
            report[level]["candidate_better"], report[level]["candidate_worse"] = better, worse
        if args.details:
            report["rows"] = rows
        print(json.dumps(report, indent=2))
    finally:
        await q.close()


if __name__ == "__main__":
    asyncio.run(main())

import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "project_docs")

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE = _env_int("INGEST_CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _env_int("INGEST_CHUNK_OVERLAP", 120)
UPSERT_BATCH_SIZE = _env_int("INGEST_UPSERT_BATCH_SIZE", 64)
OLLAMA_TIMEOUT_S = _env_int("OLLAMA_TIMEOUT_S", 120)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    if chunk_size <= 0:
        return [t]

    overlap = max(0, min(overlap, max(0, chunk_size - 1)))
    step = max(1, chunk_size - overlap)

    chunks: List[str] = []
    i = 0
    n = len(t)
    while i < n:
        chunk = t[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


def extract_doc_fields(doc: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Return (text, meta). Meta is safe-ish metadata; you must still avoid secrets.
    """
    text = None
    for k in ("text", "content", "page_content", "body"):
        v = doc.get(k)
        if isinstance(v, str) and v.strip():
            text = v
            break
    if text is None:
        # Fallback: stringify doc (can be noisy, but better than dropping silently)
        text = json.dumps(doc, ensure_ascii=False)

    meta: Dict[str, Any] = {}
    for k in ("id", "doc_id", "title", "url", "path", "source"):
        v = doc.get(k)
        if v is not None:
            meta[k] = v
    return text, meta


async def ollama_embed(text: str, *, model: str = EMBED_MODEL) -> List[float]:
    payload = {"model": model, "prompt": text}
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_S) as client:
        r = await client.post(f"{OLLAMA_URL}/api/embeddings", json=payload)
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError("Ollama embeddings returned empty embedding")
    return [float(x) for x in emb]


async def ensure_collection(client: AsyncQdrantClient, vector_size: int) -> None:
    try:
        await client.get_collection(QDRANT_COLLECTION)
        return
    except Exception:
        # Create if missing
        pass

    await client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )


def _point_id(source: Optional[str], doc_id: Optional[str], chunk_index: int, chunk_text: str) -> str:
    s = source or ""
    d = doc_id or ""
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    h.update(b"|")
    h.update(d.encode("utf-8"))
    h.update(b"|")
    h.update(str(chunk_index).encode("ascii"))
    h.update(b"|")
    h.update(hashlib.sha256(chunk_text.encode("utf-8")).digest())
    return h.hexdigest()


async def ingest_docs(
    docs: Iterable[Dict[str, Any]],
    *,
    source: Optional[str],
    tags: List[str],
) -> Dict[str, Any]:
    q = AsyncQdrantClient(url=QDRANT_URL)

    total_chunks = 0
    upserted = 0

    batch: List[qm.PointStruct] = []
    created_collection = False

    for doc in docs:
        text, meta = extract_doc_fields(doc)
        doc_id = str(meta.get("id") or meta.get("doc_id") or "")

        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            emb = await ollama_embed(chunk, model=EMBED_MODEL)

            if not created_collection:
                await ensure_collection(q, vector_size=len(emb))
                created_collection = True

            pid = _point_id(source, doc_id, idx, chunk)
            payload = {
                "text": chunk,
                "source": source,
                "tags": tags,
                "doc_id": doc_id or None,
                "chunk_index": idx,
                "meta": meta,
            }
            batch.append(qm.PointStruct(id=pid, vector=emb, payload=payload))
            total_chunks += 1

            if len(batch) >= UPSERT_BATCH_SIZE:
                await q.upsert(collection_name=QDRANT_COLLECTION, points=batch, wait=True)
                upserted += len(batch)
                batch.clear()

    if batch:
        await q.upsert(collection_name=QDRANT_COLLECTION, points=batch, wait=True)
        upserted += len(batch)
        batch.clear()

    await q.close()
    return {"ok": True, "chunks": total_chunks, "upserted": upserted, "collection": QDRANT_COLLECTION, "embed_model": EMBED_MODEL}


import hashlib
import uuid
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app.inference_retry import InferenceWorkers, run_with_oom_wait
from app.metrics import observe_gpu_oom


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
EMBED_MAX_CHARS = _env_int("EMBED_MAX_CHARS", 4000)
EMBED_MIN_CHARS = _env_int("EMBED_MIN_CHARS", 200)
EMBED_RETRIES = _env_int("EMBED_RETRIES", 6)
OOM_WAIT_MAX_SECONDS = _env_int("OOM_WAIT_MAX_SECONDS", 180)
OOM_WAIT_INTERVAL_SECONDS = _env_int("OOM_WAIT_INTERVAL_SECONDS", 5)
INGEST_WORKER_CONCURRENCY = _env_int("INGEST_WORKER_CONCURRENCY", 1)
_EMBED_WORKERS = InferenceWorkers(INGEST_WORKER_CONCURRENCY)


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


def _is_context_len_error(status: int, body_text: str) -> bool:
    if status not in (400, 404, 500):
        return False
    b = (body_text or "").lower()
    return ("input length exceeds the context length" in b) or ("context length" in b and "exceed" in b)


async def ollama_embed(text: str, *, model: str = EMBED_MODEL) -> Tuple[List[float], str]:
    """
    Return (embedding, prompt_used). We may shrink the prompt to avoid Ollama context-length errors.
    """
    prompt = (text or "").strip()
    if not prompt:
        raise RuntimeError("Empty text cannot be embedded")

    # Hard cap as a first line of defense (chars != tokens, but this avoids pathological inputs).
    if EMBED_MAX_CHARS > 0 and len(prompt) > EMBED_MAX_CHARS:
        prompt = prompt[:EMBED_MAX_CHARS].strip()

    tries = max(1, EMBED_RETRIES)
    min_len = max(1, EMBED_MIN_CHARS)

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_S) as client:
        last_r: Optional[httpx.Response] = None
        for _ in range(tries):
            payload = {"model": model, "prompt": prompt}
            r = await _EMBED_WORKERS.run(
                lambda: run_with_oom_wait(
                    lambda: client.post(f"{OLLAMA_URL}/api/embeddings", json=payload),
                    max_wait_s=OOM_WAIT_MAX_SECONDS,
                    interval_s=OOM_WAIT_INTERVAL_SECONDS,
                    on_oom=lambda: observe_gpu_oom(operation="ingest_embed"),
                )
            )
            last_r = r
            if r.status_code == 200:
                data = r.json()
                emb = data.get("embedding")
                if not isinstance(emb, list) or not emb:
                    raise RuntimeError("Ollama embeddings returned empty embedding")
                return [float(x) for x in emb], prompt

            # If the prompt is still too big, shrink and retry.
            body = r.text or ""
            if _is_context_len_error(r.status_code, body) and len(prompt) > min_len:
                prompt = prompt[: max(min_len, len(prompt) // 2)].strip()
                continue

            # Otherwise, treat as a hard error (include response body for debugging).
            raise RuntimeError(f"Ollama embeddings failed: {r.status_code} {body}")

        # Exceeded retries; include last response body for debugging.
        if last_r is not None:
            raise RuntimeError(f"Ollama embeddings failed after retries: {last_r.status_code} {last_r.text}")
        raise RuntimeError("Ollama embeddings failed after retries")


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
    """
    Generate a deterministic UUID for the point ID (Qdrant accepts int or UUID).
    """
    base = f"{source or ''}|{doc_id or ''}|{chunk_index}|{hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


async def ingest_docs(
    docs: Iterable[Dict[str, Any]],
    *,
    source: Optional[str],
    tags: List[str],
) -> Dict[str, Any]:
    q = AsyncQdrantClient(url=QDRANT_URL)

    total_chunks = 0
    upserted = 0
    skipped_chunks = 0
    truncated_chunks = 0

    batch: List[qm.PointStruct] = []
    created_collection = False

    # Keep chunking under a sane size for embeddings.
    effective_max = EMBED_MAX_CHARS if EMBED_MAX_CHARS > 0 else 4000
    effective_chunk_size = CHUNK_SIZE if CHUNK_SIZE > 0 else effective_max
    effective_chunk_size = min(effective_chunk_size, effective_max)
    effective_overlap = max(0, min(CHUNK_OVERLAP, max(0, effective_chunk_size - 1)))

    for doc in docs:
        text, meta = extract_doc_fields(doc)
        doc_id = str(meta.get("id") or meta.get("doc_id") or "")

        chunks = chunk_text(text, chunk_size=effective_chunk_size, overlap=effective_overlap)
        for idx, chunk in enumerate(chunks):
            original_chunk = chunk
            try:
                emb, prompt_used = await ollama_embed(chunk, model=EMBED_MODEL)
            except Exception as e:
                # Only skip the specific "context length exceeded" class of failures.
                msg = str(e).lower()
                if "context length" in msg and "exceed" in msg:
                    skipped_chunks += 1
                    continue
                raise

            was_truncated = prompt_used != original_chunk
            if was_truncated:
                truncated_chunks += 1
                chunk = prompt_used

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
                "truncated_for_embedding": was_truncated,
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
    return {
        "ok": True,
        "chunks": total_chunks,
        "upserted": upserted,
        "skipped_chunks": skipped_chunks,
        "truncated_chunks": truncated_chunks,
        "collection": QDRANT_COLLECTION,
        "embed_model": EMBED_MODEL,
        "effective_chunk_size": effective_chunk_size,
        "effective_overlap": effective_overlap,
        "embed_max_chars": effective_max,
    }


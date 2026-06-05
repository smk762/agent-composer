"""Document ingestion pipeline.

Two paths share most of the code:

* **Text (legacy)** — ``RETRIEVAL_PROFILE=text``. One unnamed dense vector per
  chunk via Ollama. Backward compatible with the existing ``project_docs``
  collection layout. No tree-sitter, no sparse vectors.

* **Code-audit** — ``RETRIEVAL_PROFILE=code``. Tree-sitter chunks per
  function/class, dense vector via the Infinity sidecar (default
  ``nomic-ai/CodeRankEmbed``), optional BM25 sparse vector for hybrid
  retrieval, written to a SEPARATE collection (``QDRANT_COLLECTION_CODE``)
  with named-vector layout so the two paths can coexist.

The pipeline branches per-document so a single ingest call can mix code and
prose if the caller marks each doc accordingly.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app.code_chunker import CodeChunk, chunk_code, language_for_path
from app.embeddings import (
    EMBED_MODEL,
    EMBED_MODEL_CODE,
    InfinityEmbeddingClient,
    OllamaEmbeddingClient,
    make_client,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "project_docs")
QDRANT_COLLECTION_CODE = os.getenv("QDRANT_COLLECTION_CODE", "code_audit")
RETRIEVAL_PROFILE = os.getenv("RETRIEVAL_PROFILE", "text").strip().lower()
HYBRID_SPARSE = os.getenv("HYBRID_SPARSE", "0") == "1"

CHUNK_SIZE = _env_int("INGEST_CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _env_int("INGEST_CHUNK_OVERLAP", 120)
UPSERT_BATCH_SIZE = _env_int("INGEST_UPSERT_BATCH_SIZE", 64)
EMBED_MAX_CHARS = _env_int("EMBED_MAX_CHARS", 4000)

# Single shared client; for the code path we ALWAYS go through Infinity even
# if EMBED_BACKEND happens to be set to ollama, because Ollama doesn't host
# code-specialized embedders.
_text_client = make_client()
_code_client: Optional[InfinityEmbeddingClient] = None


def _get_code_client() -> InfinityEmbeddingClient:
    global _code_client
    if _code_client is None:
        _code_client = InfinityEmbeddingClient(code_model=EMBED_MODEL_CODE, text_model=EMBED_MODEL)
    return _code_client


# Lazy singletons for sparse vectors; only loaded when HYBRID_SPARSE=1.
_bm25 = None
def _get_bm25():
    global _bm25
    if _bm25 is None:
        from fastembed import SparseTextEmbedding  # heavy import; defer until used
        _bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _bm25


# ---- Text chunking (legacy) --------------------------------------------------

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
    """Return (text, meta). Caller is still responsible for keeping secrets out."""
    text: Optional[str] = None
    for k in ("text", "content", "page_content", "body"):
        v = doc.get(k)
        if isinstance(v, str) and v.strip():
            text = v
            break
    if text is None:
        text = json.dumps(doc, ensure_ascii=False)

    meta: Dict[str, Any] = {}
    for k in ("id", "doc_id", "title", "url", "path", "source"):
        v = doc.get(k)
        if v is not None:
            meta[k] = v
    return text, meta


def _point_id(source: Optional[str], doc_id: Optional[str], chunk_index: int, chunk_text: str) -> str:
    base = f"{source or ''}|{doc_id or ''}|{chunk_index}|{hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


def _doc_is_code(doc: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    kind = (doc.get("kind") or "").strip().lower()
    if kind == "code":
        return True
    if kind == "text":
        return False
    path = doc.get("path") or meta.get("path") or ""
    return language_for_path(path) is not None


# ---- Collection lifecycle ----------------------------------------------------

async def ensure_legacy_collection(client: AsyncQdrantClient, *, vector_size: int) -> None:
    """Legacy unnamed-dense-vector collection (RETRIEVAL_PROFILE=text)."""
    try:
        await client.get_collection(QDRANT_COLLECTION)
        return
    except Exception:
        pass
    await client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )


async def ensure_hybrid_collection(
    client: AsyncQdrantClient, *, dense_dim: int, name: str, with_sparse: bool
) -> None:
    """Named-dense + optional sparse-bm25 collection (RETRIEVAL_PROFILE=code)."""
    try:
        await client.get_collection(name)
        return
    except Exception:
        pass

    sparse_cfg = None
    if with_sparse:
        sparse_cfg = {
            "bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF),
        }
    await client.create_collection(
        collection_name=name,
        vectors_config={"dense": qm.VectorParams(size=dense_dim, distance=qm.Distance.COSINE)},
        sparse_vectors_config=sparse_cfg,
    )


# ---- Per-doc ingest helpers --------------------------------------------------

async def _flush_batch(
    q: AsyncQdrantClient,
    state: Dict[str, Any],
    collection: str,
    *,
    force: bool = False,
) -> None:
    batch = state["batches"][collection]
    if not batch:
        return
    if not force and len(batch) < UPSERT_BATCH_SIZE:
        return
    await q.upsert(collection_name=collection, points=batch, wait=True)
    state["upserted"] += len(batch)
    batch.clear()


async def _ingest_text_doc(
    q: AsyncQdrantClient,
    doc: Dict[str, Any],
    *,
    source: Optional[str],
    tags: List[str],
    state: Dict[str, Any],
) -> None:
    text, meta = extract_doc_fields(doc)
    doc_id = str(meta.get("id") or meta.get("doc_id") or "")

    effective_max = EMBED_MAX_CHARS if EMBED_MAX_CHARS > 0 else 4000
    effective_chunk_size = min(CHUNK_SIZE if CHUNK_SIZE > 0 else effective_max, effective_max)
    effective_overlap = max(0, min(CHUNK_OVERLAP, max(0, effective_chunk_size - 1)))
    state["effective_chunk_size"] = effective_chunk_size
    state["effective_overlap"] = effective_overlap
    state["embed_max_chars"] = effective_max

    chunks = chunk_text(text, chunk_size=effective_chunk_size, overlap=effective_overlap)
    if not chunks:
        return

    # Embed one-by-one so per-chunk shrink-on-overflow still works.
    for idx, chunk in enumerate(chunks):
        try:
            res = await _text_client.embed([chunk], profile="text")
        except Exception as e:
            msg = str(e).lower()
            if "context length" in msg and "exceed" in msg:
                state["skipped_chunks"] += 1
                continue
            raise

        emb = res.vectors[0]
        was_truncated = bool(res.truncated and res.truncated[0])
        if was_truncated:
            state["truncated_chunks"] += 1

        if not state["created_collections"].get(QDRANT_COLLECTION):
            await ensure_legacy_collection(q, vector_size=len(emb))
            state["created_collections"][QDRANT_COLLECTION] = True

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
        state["batches"][QDRANT_COLLECTION].append(
            qm.PointStruct(id=pid, vector=emb, payload=payload)
        )
        state["total_chunks"] += 1
        await _flush_batch(q, state, QDRANT_COLLECTION)


async def _ingest_code_doc(
    q: AsyncQdrantClient,
    doc: Dict[str, Any],
    *,
    source: Optional[str],
    tags: List[str],
    state: Dict[str, Any],
) -> None:
    text, meta = extract_doc_fields(doc)
    doc_id = str(meta.get("id") or meta.get("doc_id") or "")

    path = doc.get("path") or meta.get("path") or ""
    language = (
        doc.get("language")
        or meta.get("language")
        or language_for_path(path)
        or "text"
    )
    repo = doc.get("repo") or meta.get("repo")
    commit_sha = doc.get("commit_sha") or meta.get("commit_sha")

    code_chunks: List[CodeChunk] = chunk_code(text, language)
    if not code_chunks:
        return

    code_client = _get_code_client()
    res = await code_client.embed([c.text for c in code_chunks], profile="code")
    if HYBRID_SPARSE:
        sparse_vecs = list(_get_bm25().embed([c.text for c in code_chunks]))
    else:
        sparse_vecs = [None] * len(code_chunks)

    if not state["created_collections"].get(QDRANT_COLLECTION_CODE):
        await ensure_hybrid_collection(
            q,
            dense_dim=res.dim,
            name=QDRANT_COLLECTION_CODE,
            with_sparse=HYBRID_SPARSE,
        )
        state["created_collections"][QDRANT_COLLECTION_CODE] = True

    truncated_flags = res.truncated or [False] * len(code_chunks)
    for idx, (chunk, dvec, svec, was_trunc) in enumerate(
        zip(code_chunks, res.vectors, sparse_vecs, truncated_flags)
    ):
        if was_trunc:
            state["truncated_chunks"] += 1

        pid = _point_id(source, doc_id or path, idx, chunk.text)
        payload: Dict[str, Any] = {
            "text": chunk.text,
            "source": source,
            "tags": tags,
            "doc_id": doc_id or None,
            "chunk_index": idx,
            "meta": meta,
            "repo": repo,
            "path": path or None,
            "language": chunk.language,
            "node_type": chunk.node_type,
            "symbol": chunk.symbol,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "commit_sha": commit_sha,
            "truncated_for_embedding": was_trunc,
        }

        vector: Dict[str, Any] = {"dense": dvec}
        if svec is not None:
            vector["bm25"] = qm.SparseVector(
                indices=svec.indices.tolist(),
                values=svec.values.tolist(),
            )

        state["batches"][QDRANT_COLLECTION_CODE].append(
            qm.PointStruct(id=pid, vector=vector, payload=payload)
        )
        state["total_chunks"] += 1
        await _flush_batch(q, state, QDRANT_COLLECTION_CODE)


# ---- Public entrypoint -------------------------------------------------------

async def ingest_docs(
    docs: Iterable[Dict[str, Any]],
    *,
    source: Optional[str],
    tags: List[str],
) -> Dict[str, Any]:
    q = AsyncQdrantClient(url=QDRANT_URL)

    # Separate batches per collection so a mixed code+text ingest doesn't
    # accidentally upsert points into the wrong place.
    state: Dict[str, Any] = {
        "batches": {QDRANT_COLLECTION: [], QDRANT_COLLECTION_CODE: []},
        "created_collections": {},
        "total_chunks": 0,
        "upserted": 0,
        "skipped_chunks": 0,
        "truncated_chunks": 0,
        # legacy-only diagnostics
        "effective_chunk_size": None,
        "effective_overlap": None,
        "embed_max_chars": None,
    }

    code_docs = 0
    text_docs = 0

    # The selected profile sets the *default* kind. Per-doc `kind: "code"|"text"`
    # still wins, so a single batch can mix both.
    profile_default_code = RETRIEVAL_PROFILE == "code"

    for doc in docs:
        explicit_kind = (doc.get("kind") or "").strip().lower()
        if explicit_kind in ("code", "text"):
            is_code = explicit_kind == "code"
        elif doc.get("path"):
            is_code = language_for_path(doc["path"]) is not None
        else:
            is_code = profile_default_code

        if is_code:
            await _ingest_code_doc(q, doc, source=source, tags=tags, state=state)
            code_docs += 1
        else:
            await _ingest_text_doc(q, doc, source=source, tags=tags, state=state)
            text_docs += 1

    # Final flush of both batches.
    await _flush_batch(q, state, QDRANT_COLLECTION, force=True)
    await _flush_batch(q, state, QDRANT_COLLECTION_CODE, force=True)
    await q.close()

    target_collection = QDRANT_COLLECTION_CODE if profile_default_code else QDRANT_COLLECTION
    return {
        "ok": True,
        "chunks": state["total_chunks"],
        "upserted": state["upserted"],
        "skipped_chunks": state["skipped_chunks"],
        "truncated_chunks": state["truncated_chunks"],
        "collection": target_collection,
        "embed_model": EMBED_MODEL_CODE if profile_default_code else EMBED_MODEL,
        "retrieval_profile": RETRIEVAL_PROFILE,
        "hybrid_sparse": HYBRID_SPARSE,
        "code_docs": code_docs,
        "text_docs": text_docs,
        "effective_chunk_size": state["effective_chunk_size"],
        "effective_overlap": state["effective_overlap"],
        "embed_max_chars": state["embed_max_chars"],
    }

"""Document ingestion pipeline.

Two paths share most of the code:

* **Text (legacy)** — ``RETRIEVAL_PROFILE=text``. One unnamed dense vector per
  chunk via Ollama. Writes ``QDRANT_COLLECTION`` (default ``audit_docs``;
  the old ``project_docs`` is deprecated) in the legacy unnamed-vector layout. No tree-sitter, no sparse vectors.

* **Code-audit** — ``RETRIEVAL_PROFILE=code``. Tree-sitter chunks per
  function/class, dense vector via the Infinity sidecar (default
  ``nomic-ai/CodeRankEmbed``), optional BM25 sparse vector for hybrid
  retrieval, written to a SEPARATE collection (``QDRANT_COLLECTION_CODE``)
  with named-vector layout so the two paths can coexist.

The pipeline branches per-document so a single ingest call can mix code and
prose if the caller marks each doc accordingly.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
import weakref
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Tuple

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


log = logging.getLogger("rag-ingest.ingest")

QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.1.109:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "audit_docs")
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


class IngestRequestError(ValueError):
    """The request itself is unusable (rag-ingest answers 400)."""


def _point_id(source: Optional[str], doc_id: Optional[str], chunk_index: int, chunk_text: str) -> str:
    base = f"{source or ''}|{doc_id or ''}|{chunk_index}|{hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


def _doc_id(doc: Dict[str, Any]) -> str:
    """The doc's ``id`` (else ``doc_id``) as a string, or "" when it has none.

    ``0`` is an id: only a missing, null or blank value means "no id"."""
    for k in ("id", "doc_id"):
        v = doc.get(k)
        if v is not None and str(v).strip():
            return str(v)
    return ""


def _route_is_code(doc: Dict[str, Any]) -> bool:
    """Per-doc ``kind`` wins; else a path decides by its language; else the profile's default."""
    explicit_kind = (doc.get("kind") or "").strip().lower()
    if explicit_kind in ("code", "text"):
        return explicit_kind == "code"
    if doc.get("path"):
        return language_for_path(doc["path"]) is not None
    return RETRIEVAL_PROFILE == "code"


# ---- Collection lifecycle ----------------------------------------------------

def allowed_collections() -> set[str]:
    """Text collections a request may target: the default plus INGEST_ALLOWED_COLLECTIONS."""
    extra = {c.strip() for c in os.getenv("INGEST_ALLOWED_COLLECTIONS", "").split(",") if c.strip()}
    return {QDRANT_COLLECTION} | extra


async def _collection_exists(client: AsyncQdrantClient, name: str) -> bool:
    try:
        await client.get_collection(name)
        return True
    except Exception:
        return False


async def _create_collection_if_missing(client: AsyncQdrantClient, name: str, **config: Any) -> None:
    """Create ``name`` unless it exists. Losing a creation race to a concurrent request is fine."""
    if await _collection_exists(client, name):
        return
    try:
        await client.create_collection(collection_name=name, **config)
    except Exception:
        if await _collection_exists(client, name):
            return
        raise


async def ensure_legacy_collection(client: AsyncQdrantClient, *, vector_size: int,
                                   name: str = QDRANT_COLLECTION) -> None:
    """Legacy unnamed-dense-vector collection (RETRIEVAL_PROFILE=text)."""
    await _create_collection_if_missing(
        client, name, vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )


async def ensure_hybrid_collection(
    client: AsyncQdrantClient, *, dense_dim: int, name: str, with_sparse: bool
) -> None:
    """Named-dense + optional sparse-bm25 collection (RETRIEVAL_PROFILE=code)."""
    sparse_cfg = None
    if with_sparse:
        sparse_cfg = {
            "bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF),
        }
    await _create_collection_if_missing(
        client,
        name,
        vectors_config={"dense": qm.VectorParams(size=dense_dim, distance=qm.Distance.COSINE)},
        sparse_vectors_config=sparse_cfg,
    )


async def ensure_collections_on_startup() -> None:
    """Idempotently create the active Qdrant collection(s) at launch.

    Vector dimensionality is derived from a one-shot sample embedding so it
    always matches the live model rather than a hardcoded constant. Best-effort:
    callers must not let a failure here block startup — the per-upsert lazy
    creation in the ingest path remains the fallback if the embedding backend
    (Ollama/Infinity) or Qdrant isn't ready yet.
    """
    client = AsyncQdrantClient(url=QDRANT_URL)
    try:
        if RETRIEVAL_PROFILE == "code":
            sample = await _get_code_client().embed(["def _ping():\n    return 1\n"], profile="code")
            await ensure_hybrid_collection(
                client, dense_dim=sample.dim, name=QDRANT_COLLECTION_CODE, with_sparse=HYBRID_SPARSE
            )
            log.info(
                "qdrant bootstrap: ensured code collection %r (dim=%d, sparse=%s) @ %s",
                QDRANT_COLLECTION_CODE, sample.dim, HYBRID_SPARSE, QDRANT_URL,
            )
        else:
            sample = await _text_client.embed(["ping"], profile="text")
            await ensure_legacy_collection(client, vector_size=sample.dim)
            log.info(
                "qdrant bootstrap: ensured text collection %r (dim=%d) @ %s",
                QDRANT_COLLECTION, sample.dim, QDRANT_URL,
            )
    finally:
        await client.close()


# ---- Replace on re-send ------------------------------------------------------
#
# A doc that carries an ``id`` has an identity: (collection, source, doc_id). Code docs
# without an id fall back to (collection, source, repo, path). When a request writes an
# identity, every older point of that identity that the request didn't write is deleted,
# after all of the request's upserts succeeded. Docs with neither id nor path keep the old
# append-only behaviour.
#
# What keeps that true:
# * A request may name each identity once (else IngestRequestError -> 400): two versions of
#   one doc in one request have no defined winner.
# * Requests touching the same identity are serialised by a per-identity lock, held from the
#   first embedding to the last prune, so two re-sends can't delete each other's chunks.
#   ``/delete`` takes the same locks.
# * Every point carries ``ingest_seq`` (ns timestamp of its request) and a prune only removes
#   points older than its own request. If two processes ever write one identity without
#   sharing the lock, the worst case is both versions surviving, never neither.
# * Points are only upserted once every doc in the request has been embedded, so a failed
#   embedding leaves the previous version alone and writes nothing new.

Identity = Tuple[str, Optional[str], str, str, Optional[str]]  # collection, source, field, value, repo

_INDEXED: set = set()
_IDENTITY_LOCKS: "weakref.WeakValueDictionary[Identity, asyncio.Lock]" = weakref.WeakValueDictionary()


@asynccontextmanager
async def _identity_locks(identities: Iterable[Identity]) -> AsyncIterator[None]:
    """Hold the lock of every identity, taken in one global order so requests can't deadlock."""
    locks: List[asyncio.Lock] = []
    for ident in sorted(set(identities), key=repr):
        lock = _IDENTITY_LOCKS.get(ident)
        if lock is None:
            lock = asyncio.Lock()
            _IDENTITY_LOCKS[ident] = lock
        locks.append(lock)  # the strong reference keeps the entry alive while we hold it
    held: List[asyncio.Lock] = []
    try:
        for lock in locks:
            await lock.acquire()
            held.append(lock)
        yield
    finally:
        for lock in reversed(held):
            lock.release()


def _doc_identity(doc: Dict[str, Any], *, is_code: bool, text_collection: str,
                  source: Optional[str]) -> Optional[Identity]:
    doc_id = _doc_id(doc)
    if not is_code:
        return (text_collection, source, "doc_id", doc_id, None) if doc_id else None
    if doc_id:
        return (QDRANT_COLLECTION_CODE, source, "doc_id", doc_id, None)
    path = doc.get("path") or ""
    if path:
        return (QDRANT_COLLECTION_CODE, source, "path", path, doc.get("repo"))
    return None


def _describe(identity: Identity) -> str:
    _, _, field, value, repo = identity
    return f"{field}={value!r}" + (f" (repo {repo!r})" if field == "path" else "")


def identity_filter(source: Optional[str], field: str, value: str, repo: Optional[str] = None,
                    by_repo: bool = False) -> qm.Filter:
    """Points of one doc identity. ``source=None`` matches points stored without a source."""
    must: List[Any] = [qm.FieldCondition(key=field, match=qm.MatchValue(value=value))]
    if source is None:
        must.append(qm.IsNullCondition(is_null=qm.PayloadField(key="source")))
    else:
        must.append(qm.FieldCondition(key="source", match=qm.MatchValue(value=source)))
    if by_repo:
        if repo:
            must.append(qm.FieldCondition(key="repo", match=qm.MatchValue(value=repo)))
        else:
            must.append(qm.IsNullCondition(is_null=qm.PayloadField(key="repo")))
    return qm.Filter(must=must)


async def ensure_payload_indexes(q: AsyncQdrantClient, collection: str, fields: Iterable[str]) -> None:
    """Keyword indexes so identity filters stay cheap. Best effort; retried until one succeeds."""
    for field in fields:
        if (collection, field) in _INDEXED:
            continue
        try:
            await q.create_payload_index(collection_name=collection, field_name=field,
                                         field_schema=qm.PayloadSchemaType.KEYWORD, wait=True)
        except Exception as e:  # already there with another schema, or Qdrant trouble
            log.debug("payload index %s.%s not created: %s", collection, field, e)
            continue
        _INDEXED.add((collection, field))


async def _delete_matching(q: AsyncQdrantClient, collection: str, flt: qm.Filter) -> int:
    n = (await q.count(collection_name=collection, count_filter=flt, exact=True)).count
    if n:
        await q.delete(collection_name=collection, points_selector=qm.FilterSelector(filter=flt), wait=True)
    return n


async def _prune_replaced(q: AsyncQdrantClient, state: Dict[str, Any]) -> int:
    """Delete each written identity's older points that this request didn't write. Returns the count."""
    pruned = 0
    exists: Dict[str, bool] = {}
    for (collection, source, field, value, repo), keep in state["written"].items():
        if collection not in exists:
            exists[collection] = await _collection_exists(q, collection)
        if not exists[collection]:
            continue  # nothing was ever stored there, so nothing to replace
        by_repo = field == "path"
        await ensure_payload_indexes(q, collection, ["doc_id", "source"] + (["path", "repo"] if by_repo else []))
        flt = identity_filter(source, field, value, repo, by_repo)
        if keep:
            flt.must_not = [qm.HasIdCondition(has_id=sorted(keep))]
        flt.should = [
            qm.FieldCondition(key="ingest_seq", range=qm.Range(lt=state["ingest_seq"])),
            qm.IsEmptyCondition(is_empty=qm.PayloadField(key="ingest_seq")),  # written before ingest_seq
        ]
        pruned += await _delete_matching(q, collection, flt)
    return pruned


async def delete_docs(doc_ids: List[str], *, source: Optional[str], collection: str,
                      paths: Optional[List[str]] = None, repo: Optional[str] = None) -> Dict[str, Any]:
    """Remove every point of the given doc ids (for ``source``) from ``collection``.

    ``paths`` (code collection only) removes docs that were ingested without an id, keyed by
    ``repo`` + ``path``."""
    doc_ids = list(dict.fromkeys(doc_ids))
    paths = list(dict.fromkeys(paths or []))
    identities: List[Identity] = [(collection, source, "doc_id", d, None) for d in doc_ids]
    identities += [(collection, source, "path", p, repo) for p in paths]
    q = AsyncQdrantClient(url=QDRANT_URL)
    deleted = 0
    try:
        async with _identity_locks(identities):
            if await _collection_exists(q, collection):
                await ensure_payload_indexes(q, collection, ["doc_id", "source"] + (["path", "repo"] if paths else []))
                for _, src, field, value, rep in identities:
                    flt = identity_filter(src, field, value, rep, by_repo=field == "path")
                    deleted += await _delete_matching(q, collection, flt)
    finally:
        await q.close()
    out: Dict[str, Any] = {"ok": True, "collection": collection, "docs": len(doc_ids), "deleted": deleted}
    if paths:
        out["paths"] = len(paths)
    return out


# ---- Per-doc ingest helpers --------------------------------------------------

def _stage(state: Dict[str, Any], collection: str, identity: Optional[Identity], point: qm.PointStruct) -> None:
    """Queue a point for the upsert that happens once the whole request is embedded."""
    if identity is not None:
        state["written"][identity].add(point.id)
    state["batches"][collection].append(point)
    state["total_chunks"] += 1


async def _upsert_staged(q: AsyncQdrantClient, state: Dict[str, Any], collection: str) -> None:
    batch = state["batches"][collection]
    size = UPSERT_BATCH_SIZE if UPSERT_BATCH_SIZE > 0 else len(batch) or 1
    for i in range(0, len(batch), size):
        part = batch[i : i + size]
        await q.upsert(collection_name=collection, points=part, wait=True)
        state["upserted"] += len(part)
    batch.clear()


def _mark_incomplete(state: Dict[str, Any], doc: Dict[str, Any]) -> None:
    label = _doc_id(doc) or doc.get("path") or None
    if label is not None and label not in state["incomplete_docs"]:
        state["incomplete_docs"].append(label)


async def _ingest_text_doc(
    q: AsyncQdrantClient,
    doc: Dict[str, Any],
    *,
    source: Optional[str],
    tags: List[str],
    state: Dict[str, Any],
    identity: Optional[Identity],
) -> None:
    text, meta = extract_doc_fields(doc)
    doc_id = _doc_id(doc)

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
                _mark_incomplete(state, doc)
                continue
            raise

        emb = res.vectors[0]
        was_truncated = bool(res.truncated and res.truncated[0])
        if was_truncated:
            state["truncated_chunks"] += 1

        coll = state["text_collection"]
        if not state["created_collections"].get(coll):
            await ensure_legacy_collection(q, vector_size=len(emb), name=coll)
            state["created_collections"][coll] = True

        pid = _point_id(source, doc_id, idx, chunk)
        payload = {
            "text": chunk,
            "source": source,
            "tags": tags,
            "doc_id": doc_id or None,
            "chunk_index": idx,
            "meta": meta,
            "truncated_for_embedding": was_truncated,
            "ingested_at": state["ingested_at"],
            "ingest_seq": state["ingest_seq"],
        }
        _stage(state, coll, identity, qm.PointStruct(id=pid, vector=emb, payload=payload))


async def _ingest_code_doc(
    q: AsyncQdrantClient,
    doc: Dict[str, Any],
    *,
    source: Optional[str],
    tags: List[str],
    state: Dict[str, Any],
    identity: Optional[Identity],
) -> None:
    text, meta = extract_doc_fields(doc)
    doc_id = _doc_id(doc)

    path = doc.get("path") or ""
    language = doc.get("language") or language_for_path(path) or "text"
    repo = doc.get("repo")
    commit_sha = doc.get("commit_sha")

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

        # repo is part of a path-keyed id: the same file in two repos must not share points.
        pid = _point_id(source, doc_id or (f"{repo}|{path}" if repo else path), idx, chunk.text)
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
            "ingested_at": state["ingested_at"],
            "ingest_seq": state["ingest_seq"],
        }

        vector: Dict[str, Any] = {"dense": dvec}
        if svec is not None:
            vector["bm25"] = qm.SparseVector(
                indices=svec.indices.tolist(),
                values=svec.values.tolist(),
            )

        _stage(state, QDRANT_COLLECTION_CODE, identity, qm.PointStruct(id=pid, vector=vector, payload=payload))


# ---- Public entrypoint -------------------------------------------------------

async def ingest_docs(
    docs: Iterable[Dict[str, Any]],
    *,
    source: Optional[str],
    tags: List[str],
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingest docs. ``collection`` overrides the text collection for this request; it must be
    in allowed_collections(). Code docs always go to QDRANT_COLLECTION_CODE.

    Raises IngestRequestError for an unusable request (a disallowed collection, or one doc
    identity sent twice)."""
    text_collection = collection or QDRANT_COLLECTION
    if text_collection not in allowed_collections():
        raise IngestRequestError(f"collection {text_collection!r} is not allowed")

    # Route every doc and work out its identity before touching anything.
    plan: List[Tuple[Dict[str, Any], bool, Optional[Identity]]] = []
    first_seen: Dict[Identity, int] = {}
    for i, doc in enumerate(docs):
        is_code = _route_is_code(doc)
        identity = _doc_identity(doc, is_code=is_code, text_collection=text_collection, source=source)
        if identity is not None:
            if identity in first_seen:
                raise IngestRequestError(
                    f"docs[{first_seen[identity]}] and docs[{i}] are the same doc ({_describe(identity)}); "
                    "send each doc once per request"
                )
            first_seen[identity] = i
        plan.append((doc, is_code, identity))

    state: Dict[str, Any] = {
        "text_collection": text_collection,
        # Separate batches per collection so a mixed code+text ingest doesn't
        # accidentally upsert points into the wrong place.
        "batches": {text_collection: [], QDRANT_COLLECTION_CODE: []},
        "created_collections": {},
        # Every identity in the request, even one that ends up with no chunks: its re-send
        # still replaces what was stored before.
        "written": {identity: set() for identity in first_seen},
        "ingested_at": int(time.time()),
        "ingest_seq": time.time_ns(),
        "total_chunks": 0,
        "upserted": 0,
        "skipped_chunks": 0,
        "truncated_chunks": 0,
        "incomplete_docs": [],
        # legacy-only diagnostics
        "effective_chunk_size": None,
        "effective_overlap": None,
        "embed_max_chars": None,
    }

    code_docs = sum(1 for _, is_code, _ in plan if is_code)
    text_docs = len(plan) - code_docs

    q = AsyncQdrantClient(url=QDRANT_URL)
    try:
        async with _identity_locks(first_seen):
            for doc, is_code, identity in plan:
                ingest_one = _ingest_code_doc if is_code else _ingest_text_doc
                await ingest_one(q, doc, source=source, tags=tags, state=state, identity=identity)
            # Nothing is written until every doc is embedded; only then are old versions retired.
            await _upsert_staged(q, state, text_collection)
            await _upsert_staged(q, state, QDRANT_COLLECTION_CODE)
            pruned = await _prune_replaced(q, state)
    finally:
        await q.close()

    # Report where this request's docs went, not just the profile's default.
    collections = [c for c, n in ((text_collection, text_docs), (QDRANT_COLLECTION_CODE, code_docs)) if n]
    if bool(code_docs) != bool(text_docs):
        to_code = bool(code_docs)
    else:  # mixed (or empty): the profile's default, as before
        to_code = RETRIEVAL_PROFILE == "code"
    return {
        "ok": True,
        "chunks": state["total_chunks"],
        "upserted": state["upserted"],
        "docs_with_id": sum(1 for ident in state["written"] if ident[2] == "doc_id"),
        "docs_with_identity": len(state["written"]),
        "pruned": pruned,
        "skipped_chunks": state["skipped_chunks"],
        "incomplete_docs": state["incomplete_docs"],
        "truncated_chunks": state["truncated_chunks"],
        "collection": QDRANT_COLLECTION_CODE if to_code else text_collection,
        "collections": collections,
        "embed_model": EMBED_MODEL_CODE if to_code else EMBED_MODEL,
        "retrieval_profile": RETRIEVAL_PROFILE,
        "hybrid_sparse": HYBRID_SPARSE,
        "code_docs": code_docs,
        "text_docs": text_docs,
        "effective_chunk_size": state["effective_chunk_size"],
        "effective_overlap": state["effective_overlap"],
        "embed_max_chars": state["embed_max_chars"],
    }

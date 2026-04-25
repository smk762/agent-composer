"""Embedding backend abstraction.

Two backends:
* ``ollama`` (default): one request per text against ``/api/embeddings``.
* ``infinity``: batched OpenAI-shape ``/embeddings`` against the Infinity
  sidecar. Used when ``RETRIEVAL_PROFILE=code`` so we can serve a code-
  specialized embedder (default ``nomic-ai/CodeRankEmbed``).

Picking the model:
    The ``profile`` argument selects between the prose embedder
    (``EMBED_MODEL``) and the code embedder (``EMBED_MODEL_CODE``).
    For the Ollama backend, ``profile`` is ignored — Ollama is only used for
    the prose model in the legacy text path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence

import httpx

from app.inference_retry import InferenceWorkers, run_with_oom_wait
from app.metrics import observe_gpu_oom


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


EMBED_BACKEND = os.getenv("EMBED_BACKEND", "ollama").strip().lower()
EMBED_BACKEND_URL = os.getenv("EMBED_BACKEND_URL", "http://infinity:7997").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_MODEL_CODE = os.getenv("EMBED_MODEL_CODE", "nomic-ai/CodeRankEmbed")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
EMBED_TIMEOUT_S = _env_int("EMBED_TIMEOUT_S", 120)
EMBED_MAX_CHARS = _env_int("EMBED_MAX_CHARS", 4000)
EMBED_MIN_CHARS = _env_int("EMBED_MIN_CHARS", 200)
EMBED_RETRIES = _env_int("EMBED_RETRIES", 6)
OOM_WAIT_MAX_SECONDS = _env_int("OOM_WAIT_MAX_SECONDS", 180)
OOM_WAIT_INTERVAL_SECONDS = _env_int("OOM_WAIT_INTERVAL_SECONDS", 5)
INGEST_WORKER_CONCURRENCY = _env_int("INGEST_WORKER_CONCURRENCY", 1)

_EMBED_WORKERS = InferenceWorkers(INGEST_WORKER_CONCURRENCY)


@dataclass
class EmbedResult:
    vectors: List[List[float]]
    model: str
    dim: int
    truncated: List[bool]


def _is_context_len_error(status: int, body_text: str) -> bool:
    if status not in (400, 404, 500):
        return False
    b = (body_text or "").lower()
    return ("input length exceeds the context length" in b) or (
        "context length" in b and "exceed" in b
    )


class EmbeddingClient:
    async def embed(self, texts: Sequence[str], *, profile: str = "text") -> EmbedResult:
        raise NotImplementedError


class OllamaEmbeddingClient(EmbeddingClient):
    """Single-text /api/embeddings calls with prompt-shrinking on context overflow."""

    def __init__(self, *, model: str = EMBED_MODEL):
        self.model = model

    async def _embed_one(self, client: httpx.AsyncClient, text: str) -> tuple[List[float], bool]:
        prompt = (text or "").strip()
        if not prompt:
            raise RuntimeError("Empty text cannot be embedded")
        original_len = len(prompt)
        if EMBED_MAX_CHARS > 0 and len(prompt) > EMBED_MAX_CHARS:
            prompt = prompt[:EMBED_MAX_CHARS].strip()

        tries = max(1, EMBED_RETRIES)
        min_len = max(1, EMBED_MIN_CHARS)
        last_r: httpx.Response | None = None
        for _ in range(tries):
            payload = {"model": self.model, "prompt": prompt}
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
                return [float(x) for x in emb], len(prompt) != original_len

            body = r.text or ""
            if _is_context_len_error(r.status_code, body) and len(prompt) > min_len:
                prompt = prompt[: max(min_len, len(prompt) // 2)].strip()
                continue
            raise RuntimeError(f"Ollama embeddings failed: {r.status_code} {body}")

        if last_r is not None:
            raise RuntimeError(
                f"Ollama embeddings failed after retries: {last_r.status_code} {last_r.text}"
            )
        raise RuntimeError("Ollama embeddings failed after retries")

    async def embed(self, texts: Sequence[str], *, profile: str = "text") -> EmbedResult:
        # The Ollama backend only serves the prose model (profile is informational).
        vectors: List[List[float]] = []
        truncated: List[bool] = []
        async with httpx.AsyncClient(timeout=EMBED_TIMEOUT_S) as client:
            for t in texts:
                vec, was_trunc = await self._embed_one(client, t)
                vectors.append(vec)
                truncated.append(was_trunc)
        return EmbedResult(
            vectors=vectors,
            model=self.model,
            dim=len(vectors[0]) if vectors else 0,
            truncated=truncated,
        )


class InfinityEmbeddingClient(EmbeddingClient):
    """Batched OpenAI-compatible /embeddings against the Infinity sidecar.

    Infinity transparently handles padding and truncation against the model's
    own max_seq_len, so we don't pre-shrink the prompt. We still report
    ``truncated`` based on a coarse char heuristic for parity with the Ollama
    path and to keep payloads honest.
    """

    def __init__(
        self,
        *,
        code_model: str = EMBED_MODEL_CODE,
        text_model: str = EMBED_MODEL,
        base_url: str = EMBED_BACKEND_URL,
    ):
        self.code_model = code_model
        self.text_model = text_model
        self.base_url = base_url.rstrip("/")

    def _model_for(self, profile: str) -> str:
        return self.code_model if profile == "code" else self.text_model

    async def embed(self, texts: Sequence[str], *, profile: str = "text") -> EmbedResult:
        if not texts:
            return EmbedResult(vectors=[], model=self._model_for(profile), dim=0, truncated=[])

        model = self._model_for(profile)
        # Coarse char-level cap; Infinity will further token-truncate per-model.
        cap = EMBED_MAX_CHARS if EMBED_MAX_CHARS > 0 else None
        prepared: List[str] = []
        truncated: List[bool] = []
        for t in texts:
            s = (t or "").strip()
            if not s:
                raise RuntimeError("Empty text cannot be embedded")
            if cap is not None and len(s) > cap:
                prepared.append(s[:cap])
                truncated.append(True)
            else:
                prepared.append(s)
                truncated.append(False)

        async with httpx.AsyncClient(timeout=EMBED_TIMEOUT_S) as client:
            r = await _EMBED_WORKERS.run(
                lambda: run_with_oom_wait(
                    lambda: client.post(
                        f"{self.base_url}/embeddings",
                        json={"model": model, "input": prepared},
                    ),
                    max_wait_s=OOM_WAIT_MAX_SECONDS,
                    interval_s=OOM_WAIT_INTERVAL_SECONDS,
                    on_oom=lambda: observe_gpu_oom(operation="ingest_embed"),
                )
            )
        if r.status_code != 200:
            raise RuntimeError(f"Infinity embeddings failed: {r.status_code} {r.text}")

        body = r.json()
        data = body.get("data", [])
        if not isinstance(data, list) or len(data) != len(prepared):
            raise RuntimeError(
                f"Infinity embeddings returned unexpected shape: {len(data)} != {len(prepared)}"
            )
        # Infinity returns items in input order with an "index" field; sort defensively.
        try:
            data = sorted(data, key=lambda d: int(d.get("index", 0)))
        except Exception:
            pass

        vectors: List[List[float]] = []
        for d in data:
            emb = d.get("embedding")
            if not isinstance(emb, list) or not emb:
                raise RuntimeError("Infinity embeddings returned empty embedding")
            vectors.append([float(x) for x in emb])

        return EmbedResult(
            vectors=vectors,
            model=model,
            dim=len(vectors[0]) if vectors else 0,
            truncated=truncated,
        )


def make_client() -> EmbeddingClient:
    if EMBED_BACKEND == "infinity":
        return InfinityEmbeddingClient()
    return OllamaEmbeddingClient()


# ---- Reranker (Infinity only) ------------------------------------------------

RERANK_ENABLED = os.getenv("RERANK_ENABLED", "0") == "1"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")


async def rerank(query: str, docs: Sequence[str], *, top_k: int) -> List[int]:
    """Return indices into ``docs`` sorted by descending relevance.

    Falls back to the natural input order (truncated to ``top_k``) when
    reranking is disabled or no docs are supplied.
    """
    if not RERANK_ENABLED or not docs:
        return list(range(min(top_k, len(docs))))

    async with httpx.AsyncClient(timeout=EMBED_TIMEOUT_S) as client:
        r = await client.post(
            f"{EMBED_BACKEND_URL}/rerank",
            json={
                "model": RERANK_MODEL,
                "query": query,
                "documents": list(docs),
                "top_n": top_k,
            },
        )
    if r.status_code != 200:
        raise RuntimeError(f"Infinity rerank failed: {r.status_code} {r.text}")
    results = r.json().get("results", [])
    return [int(item["index"]) for item in results]

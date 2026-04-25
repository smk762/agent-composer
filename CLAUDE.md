# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Stack overview

Three FastAPI services plus sidecars, all wired together in `docker-compose.yml`:

| Service | Port | Role |
|---------|------|------|
| `rag-ingest` | 9050 | Signed ingestion API → chunk → embed → upsert Qdrant |
| `rag-chat` | 9150 | Chat gateway (Ollama + optional RAG), history, UI, pipeline proxy |
| `modernbert` | 7998 | ModernBERT classifier sidecar (lazy-load GPU, zero-shot or fine-tuned) |
| `qdrant` | 6333/6334 | Vector store (internal; written by rag-ingest, read by rag-chat) |
| `postgres` | 5432 | Chat history DB for rag-chat (AsyncSQLAlchemy + Alembic) |
| `infinity` | 7997 | Infinity sidecar serving CodeRankEmbed + bge-reranker (code-audit path) |
| `ollama` | 11434 | Local LLM runtime (internal only) |

## Commands

```bash
# Full stack
docker compose up -d --build
docker compose logs -f rag-chat rag-ingest

# Individual service (faster rebuild)
docker compose up -d --build rag-chat

# Run tests (inside service dir, no container needed)
cd rag-ingest && python -m pytest tests/ -v
cd rag-chat   && python -m pytest tests/ -v

# Single test
python -m pytest tests/test_code_chunker.py::test_python_function -v

# Endpoint smoke test (requires running stack)
python3 scripts/test_endpoints.py

# Pull a model once
docker exec -it ollama ollama pull qwen2.5-coder:32b
```

## Configuration

Copy `env.example` → `.env`. Key variables:

- `INGEST_SHARED_SECRET` — required for rag-ingest HMAC signing
- `QDRANT_URL` — defaults to `http://qdrant:6333` (container name)
- `DATABASE_URL` — set automatically by compose to the internal postgres container
- `RETRIEVAL_PROFILE` — `text` (default, Ollama) or `code` (tree-sitter + Infinity)
- `MODERNBERT_URL` — defaults to `http://modernbert:7998`
- `AUDIT_API_URL` — ai-code-auditor audit API for diff-audit validation between repair iterations (default `http://127.0.0.1:8765`)
- `ECOSYSTEM_CONFIG_PATH` — path to `ai-code-auditor/config/ecosystem.yaml` for repo path resolution (default `/home/smk/ai-code-auditor/config/ecosystem.yaml`)
- `CF_TUNNEL_TOKEN` — only needed when running `--profile cloud`

GPU passthrough requires NVIDIA Container Toolkit on the host. Remove `gpus: all` from compose services to run CPU-only.

## Architecture: two ingest paths

`rag-ingest/app/ingest_pipeline.py` branches per document:

- **Text path** (`RETRIEVAL_PROFILE=text` or `kind: "text"`): slides a char-level window over the document, embeds each chunk via Ollama, upserts into `QDRANT_COLLECTION` using an unnamed dense vector (legacy layout).
- **Code path** (`RETRIEVAL_PROFILE=code` or `kind: "code"`, or inferred from `path`): splits via tree-sitter in `code_chunker.py` (per function/class), embeds via Infinity in `embeddings.py`, upserts into `QDRANT_COLLECTION_CODE` using named vectors (`dense` + optional `bm25`). These two collections are completely separate so both paths can coexist.

`embeddings.py` defines `OllamaEmbeddingClient` and `InfinityEmbeddingClient` both conforming to the same `EmbeddingClient` protocol. `make_client()` selects based on `EMBED_BACKEND`.

## Architecture: rag-chat

`rag-chat/app/main.py` assembles these routers: `chat`, `api_keys`, `ui`, `generation`, `media`, `providers`, `pipeline`, `repair`, `metrics`.

- **Chat** (`routers/chat.py`): streams Ollama responses, optionally injects Qdrant RAG context as a system message, persists conversation + messages to Postgres via async SQLAlchemy.
- **Pipeline proxy** (`routers/pipeline.py`): reverse-proxies `/api/pipeline/*` → ModernBERT sidecar. Exposes classify, load/evict/unload, and labels CRUD.
- **Repair** (`routers/repair.py` + `repair/`): multi-model agentic repair loop. `POST /api/repair/run` accepts a diff or `compare_branch`, runs fixer→critic→adversary iterations, validates via ai-code-auditor audit API, and streams SSE events. `GET /api/repair/jobs/{job_id}` retrieves final state. Key env: `AUDIT_API_URL`, `ECOSYSTEM_CONFIG_PATH`.
- **UI** (`routers/ui.py`): server-rendered HTML pages for chat, history, API keys, image generation, and Pipeline Lab. All pages share `render_page()` / `BASE_CSS`.
- **DB** (`db.py` + `models/orm.py`): async Postgres via `asyncpg` driver. `bootstrap_db.py` auto-creates the DB on startup; `init.sh` runs `alembic upgrade head` before uvicorn.

## Architecture: modernbert sidecar

`modernbert/app/model.py` manages lazy GPU loading with three states: `unloaded → cpu → gpu`. Keep-alive timers (`MODERNBERT_KEEP_ALIVE_GPU`, `MODERNBERT_KEEP_ALIVE_CPU`) evict the model after idle. `/health` always returns 200; check `model_state` for actual readiness.

## Request signing (rag-ingest)

Every `POST /ingest` requires:
- `X-Timestamp`, `X-Nonce`, `X-Signature` headers
- Signature: `HMAC-SHA256(sha256(secret), "{ts}.{nonce}.{sha256(body)}")`

See `scripts/example_tool_calling_client.py` and `scripts/test_endpoints.py` for working Python examples.

## Cloudflare

`cloudflared` is on the `cloud` compose profile and only starts with `docker compose --profile cloud up`. For LAN-only dev, omit the profile. Set bind IPs via `RAG_INGEST_BIND_IP` / `RAG_CHAT_BIND_IP`.

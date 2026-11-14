# Configuration

Every service reads its settings from environment variables, which compose passes through from `.env` (`cp .env.example .env`). This page lists the user-facing ones by service. Defaults are the code defaults; where `docker-compose.yml` sets a different default, the compose value is noted. `.env.example` has the full annotated list.

## Security-relevant

| Variable | Service | Default | Notes |
|---|---|---|---|
| `INGEST_SHARED_SECRET` | rag-ingest | (empty) | **Required.** HMAC signing key, and the source of the Fernet envelope key. |
| `INGEST_REQUIRE_ENCRYPTION` | rag-ingest | `0` | `1` requires a Fernet envelope `{"token": …}` on `/ingest` |
| `INGEST_MAX_SKEW_S` | rag-ingest | `300` | Accepted clock skew for `X-Timestamp` |
| `DEV_AUTH_BYPASS` | rag-chat | `0` | **`1` disables authentication.** Every request is accepted as `X-Dev-User` or `DEV_DEFAULT_USER`. Local development only. |
| `DEV_DEFAULT_USER` | rag-chat | `dev@example.com` | User id when bypassing auth without `X-Dev-User` |
| `PROVIDER_ENCRYPTION_KEY` | rag-chat | (empty) | Passphrase (PBKDF2 → Fernet) that encrypts stored image/video provider API keys. **If unset, provider keys are stored as plain base64**, and only a startup warning is logged. Set it before configuring any provider. Changing it later makes existing stored keys unreadable. |
| `CF_TUNNEL_TOKEN` | cloudflared | (empty) | Only needed with `--profile cloud` |
| `BIND_ADDR`, `*_BIND_IP` | compose | `127.0.0.1` | Host interface for published ports (`RAG_CHAT_BIND_IP`, `RAG_INGEST_BIND_IP`, `OLLAMA_BIND_IP`, `WYOMING_{TTS,MISO,STT}_BIND_IP`, …) |

Some rag-chat APIs are unauthenticated: guard, `/metrics`, and the read-only and generative voice, pipeline and repair calls. Destructive and git actions (clone delete, label replace, repair `apply`/`git_ops`) require auth. See [api.md](api.md#authentication). Binding rag-chat beyond loopback exposes the open routes.

## rag-chat: chat and RAG

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_URL` | `http://ollama:11434` | |
| `OLLAMA_TIMEOUT` | `300` | Seconds per Ollama chat call |
| `OLLAMA_KEEP_ALIVE` | `15m` | Passed to Ollama on each request |
| `CHAT_MODEL` | `llama3.2:3b` | Default model for `/chat`, pipeline helpers and the repair fixer |
| `CHAT_SYSTEM_PROMPT` | `You are a helpful assistant.` | Prepended when the request has no system message |
| `RAG_ENABLED` | `1` | `0` turns off retrieval; `/chat` then only proxies to Ollama |
| `RAG_TOP_K` | `5` | Qdrant hits injected per turn |
| `RAG_MAX_CONTEXT_CHARS` | `4000` | Cap on the injected context |
| `QDRANT_URL` | `http://<STATEFUL_HOST>:6333` | Shared with rag-ingest |
| `QDRANT_COLLECTION` | `audit_docs` | Collection that `/chat` and `/search` retrieve from and the text ingest path writes to |
| `EMBED_MODEL` | `nomic-embed-text` | Must match the model the collection was ingested with (same vector size) |
| `CHAT_DB_URL` / `DATABASE_URL` | `sqlite:////data/chat.db` | Chat history DB. `postgresql://…` is rewritten to `postgresql+asyncpg://`. The DB is created on startup if missing, then Alembic migrates it. |
| `CHAT_HISTORY_MAX_MSGS` | `100` | Each conversation is pruned to its newest N messages after each turn; `0` keeps everything |
| `INFERENCE_WORKER_CONCURRENCY` | `1` | Concurrent media-generation jobs |
| `OOM_WAIT_MAX_SECONDS` / `OOM_WAIT_INTERVAL_SECONDS` | `180` / `5` | Retry window when the GPU reports out-of-memory |
| `LOG_LEVEL` | `INFO` | |

## rag-chat: media generation

| Variable | Default | Notes |
|---|---|---|
| `MEDIA_BACKEND` | `local` | `local` (files under `MEDIA_DIR`) or `minio` |
| `MEDIA_DIR` | `/data/media` | |
| `MEDIA_S3_ENDPOINT_URL`, `MEDIA_S3_BUCKET`, `MEDIA_S3_REGION`, `MEDIA_S3_ACCESS_KEY_ID`, `MEDIA_S3_SECRET_ACCESS_KEY`, `MEDIA_S3_USE_SSL`, `MEDIA_S3_PREFIX` | | Used when `MEDIA_BACKEND=minio` |
| `COMFYUI_URL` | `http://comfyui:8188` | ComfyUI provider endpoint |

## rag-chat: guard

| Variable | Default | Notes |
|---|---|---|
| `GUARD_MODEL` | `hf.co/inclusionAI/SingGuard-4b-GGUF:Q4_K_M` | Runs on the shared Ollama |
| `GUARD_THINKING` | `fast-slow` | `fast`, `fast-slow`, `slow` or `cascade` |
| `GUARD_WORKER_CONCURRENCY` | `1` | Concurrent guard checks |

## rag-chat: voice

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_URL` | `http://whisper:8030` | STT sidecar |
| `XTTS_TTS_URL` | `http://xtts:8031` | Local XTTS engine |
| `MISO_TTS_URL` | (empty = disabled) | Remote Miso engine, e.g. `http://<GPU_HOST>:8034` |
| `TTS_URL` | (empty) | Legacy Qwen engine, registered as `tts` |
| `TTS_DEFAULT_ENGINE` | auto: `miso` if set, else `xtts` | |
| `TTS_FALLBACK_ENGINE` | auto: `xtts` when the default is something else | Used only for requests that don't name an engine |
| `VOICE_TIMEOUT` | `30` | Proxy timeout (seconds) |
| `MISO_SENTENCE_PAUSE_MS` / `MAX_SENTENCE_PAUSE_MS` | `300` / `2000` | Silence inserted between Miso sentences |
| `S3_ENDPOINT_URL`, `S3_PUBLIC_URL`, `S3_BUCKET_NAME`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` | | Shared "voice" MinIO bucket for clones and re-hosted audio |

The sidecar-side settings (`XTTS_PREPROCESS*`, `XTTS_GPT_COND_LEN`, `MISO_KEEP_ALIVE_*`, Wyoming bridge vars) are described in `voice-stack/VOICE_CLONING.md` and `.env.example`.

## rag-chat: sidecars and repair

| Variable | Default | Notes |
|---|---|---|
| `MODERNBERT_URL` | `http://modernbert:7998` | Target of the `/api/pipeline/*` proxy |
| `INFINITY_URL` | `http://infinity:7997` | Target of the `/api/runtime/infinity/*` controls |
| `AUDIT_API_URL` | `http://127.0.0.1:8765` | ai-code-auditor API for diff validation between repair iterations; empty disables validation |
| `AI_AUDIT_PUBLIC_BASE_URL` | falls back to `AUDIT_API_URL` | Base URL for fetching `pack_id` packs |
| `AUDIT_PACK_TIMEOUT_S` | `10` | Per-request timeout for pack fetches |
| `AUDIT_API_ALLOWED_HOSTS` | (empty) | Extra origins a repair request's `audit_api_url` / `audit_api_base` may name: comma-separated `host` (any port) or `host:port`, `*` for any. The configured `AUDIT_API_URL` and `AI_AUDIT_PUBLIC_BASE_URL` origins are always allowed; link-local hosts never are. Others get a 422. |
| `REPAIR_PACK_STATUS_CALLBACK` | `1` | Report `resolved` to the audit API after a clean applied `auto_fix` |
| `AI_AUDIT_API_TOKEN` | (empty) | Sent as `X-API-Token` on audit API writes |
| `ECOSYSTEM_CONFIG_PATH` | `/home/smk/ai-code-auditor/config/ecosystem.yaml` | Resolves repo paths for `apply` and `compare_branch` |

## rag-ingest

| Variable | Default | Notes |
|---|---|---|
| `RETRIEVAL_PROFILE` | `text` | Default ingest path for docs with no `kind` and no recognised `path`: `text` or `code` |
| `QDRANT_COLLECTION` | `audit_docs` | Text-path collection |
| `QDRANT_COLLECTION_CODE` | `code_audit` | Code-path collection (named vectors) |
| `EMBED_BACKEND_URL` | `http://infinity:7997` | Infinity endpoint (code path) |
| `EMBED_MODEL` | `nomic-embed-text` | Text-path embedding model, always served by Ollama. rag-chat embeds queries with the same Ollama model, so set it identically in both services. (`EMBED_BACKEND` is retired: it is ignored, with a warning, if set to anything but `ollama`.) |
| `EMBEDDER_CHECK_TTL_S` | `60` | rag-chat: how often the RAG collection's `embedder` labels are re-checked |
| `EMBED_TASK_PREFIXES` | `none` | Both services: `nomic` embeds chunks as `search_document: …` and queries as `search_query: …`. Changes the vector space, so migrate with `python -m app.reembed` rather than flipping it on a populated collection (see docs/api.md). |
| `EMBEDDER_STRICT` | `0` | Both services: `1` treats points without an `embedder` label as a mismatch |
| `EMBED_MODEL_CODE` | `nomic-ai/CodeRankEmbed` | Code-path embedding model (also preloaded by the `infinity` service) |
| `HYBRID_SPARSE` | `0` | `1` adds a `bm25` sparse vector to code points |
| `INGEST_CHUNK_SIZE` / `INGEST_CHUNK_OVERLAP` | `1000` / `120` | Text-path window, in characters |
| `CODE_CHUNK_MIN_CHARS` / `CODE_CHUNK_MAX_CHARS` | `120` / `3000` | Code-path chunk bounds |
| `EMBED_MAX_CHARS` / `EMBED_MIN_CHARS` | `4000` / `200` | Embedder input bounds |
| `EMBED_TIMEOUT_S` / `EMBED_RETRIES` | `120` / `6` | |
| `INGEST_UPSERT_BATCH_SIZE` | `64` | |
| `INGEST_NONCE_STORE` | `sqlite` | `sqlite` (`INGEST_NONCE_DB`, default `/data/nonces.db`) or `redis` (`INGEST_REDIS_URL`, `INGEST_NONCE_KEY_PREFIX`) |

## modernbert

| Variable | Default | Notes |
|---|---|---|
| `MODERNBERT_MODEL` | `answerdotai/ModernBERT-base` | |
| `MODERNBERT_TASK` | `zero_shot` | Or a fine-tuned classification head |
| `MODERNBERT_DEVICE` | `auto` | |
| `MODERNBERT_KEEP_ALIVE_GPU` / `MODERNBERT_KEEP_ALIVE_CPU` | `300` / `0` | Idle seconds before moving down a stage (`gpu → cpu → unloaded`) |
| `MODERNBERT_PRELOAD` | `0` | Load the model at startup |
| `MODERNBERT_MAX_LENGTH`, `MODERNBERT_TOP_K`, `MODERNBERT_THRESHOLD` | `512`, `10`, `0.05` | Classifier defaults |

## infinity (infinity-manager)

infinity-manager is a small proxy on port 7997. It starts an Infinity child process on the first request and stops it after the idle timeout, which frees VRAM.

| Variable | Default | Notes |
|---|---|---|
| `INFINITY_MODEL_IDS` | `nomic-ai/CodeRankEmbed,BAAI/bge-reranker-v2-m3` | Compose builds it from `EMBED_MODEL_CODE` and `RERANK_MODEL` |
| `INFINITY_IDLE_TIMEOUT` | `900` (compose: `300`) | Seconds idle before the child is stopped |
| `INFINITY_PRELOAD` | `0` | Start the child at boot |
| `INFINITY_STARTUP_TIMEOUT` / `INFINITY_REQUEST_TIMEOUT` | `180` / `300` | |
| `INFINITY_ENGINE` / `INFINITY_DEVICE` | `torch` / `cuda` | |
| `INFINITY_EXTRA_ARGS`, `INFINITY_CHILD_CMD` | (empty) | Extra CLI args, or a full override of the child command |
| `INFINITY_MANAGER_HOST`/`_PORT`, `INFINITY_CHILD_HOST`/`_PORT` | `0.0.0.0:7997`, `127.0.0.1:7998` | Internal wiring; rarely changed |

## Client scripts (`scripts/`)

| Variable | Default | Notes |
|---|---|---|
| `CHAT_URL` | `http://127.0.0.1:9150` | Base URL; scripts append `/chat` and `/health` |
| `INGEST_URL` | `http://127.0.0.1:9050` | Base URL; `test_endpoints.py` appends `/health`. Signed requests go to `${INGEST_URL}/ingest`. |
| `RAG_API_KEY` (or `CHAT_API_KEY`) | (empty) | Bearer key for `/chat`. Required unless the server runs with `DEV_AUTH_BYPASS=1`. |
| `INGEST_SHARED_SECRET` | | For signing ingest requests |

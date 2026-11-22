# Configuration

Every service reads its settings from environment variables, which compose passes through from `.env` (`cp .env.example .env`). This page lists the user-facing ones by service. Defaults are the code defaults; where `docker-compose.yml` sets a different default, the compose value is noted. `.env.example` has the full annotated list.

## Security-relevant

| Variable | Service | Default | Notes |
|---|---|---|---|
| `INGEST_SHARED_SECRET` | rag-ingest | (empty) | **Required.** HMAC signing key, and the source of the Fernet envelope key. |
| `INGEST_REQUIRE_ENCRYPTION` | rag-ingest | `0` | `1` requires a Fernet envelope `{"token": …}` on `/ingest` |
| `INGEST_MAX_SKEW_S` | rag-ingest | `300` | Accepted clock skew for `X-Timestamp` |
| `DEV_AUTH_BYPASS` | rag-chat | `0` | **`1` disables authentication.** Every request is accepted as `DEV_DEFAULT_USER`, or as an `X-Dev-User` listed in `DEV_AUTH_USERS` (any other `X-Dev-User` is a 403). Local development only; rag-chat logs a warning at startup when it is on. |
| `CF_ACCESS_TEAM_DOMAIN` | rag-chat | (empty) | Cloudflare Access team domain (e.g. `myteam.cloudflareaccess.com`). With `CF_ACCESS_AUD`, enables Access identity: the Access JWT is verified against this team's certs. Empty = Access tokens refused. |
| `CF_ACCESS_AUD` | rag-chat | (empty) | Application Audience (AUD) tag of the Access app; the JWT's `aud` must contain it. |
| `DEV_DEFAULT_USER` | rag-chat | `dev@example.com` | User id when bypassing auth without `X-Dev-User` |
| `DEV_AUTH_USERS` | rag-chat | (empty) | Comma-separated extra users the dev bypass may act as via `X-Dev-User`. Empty: only `DEV_DEFAULT_USER` |
| `PROVIDER_ENCRYPTION_KEY` | rag-chat | (empty) | Passphrase (PBKDF2 → Fernet) that encrypts stored image/video provider API keys. **If unset, saving a provider key is refused (503)** and startup logs an error; set but unusable (`cryptography` missing) fails startup. Keys saved by older versions as plain base64 stay readable (re-save them to encrypt). Changing the passphrase later makes encrypted keys unreadable. |
| `ALLOW_PLAINTEXT_PROVIDER_KEYS` | rag-chat | (unset) | `1` lets provider keys be saved without `PROVIDER_ENCRYPTION_KEY`, as `plain:`-prefixed base64. Development only. |
| `CF_TUNNEL_TOKEN` | cloudflared | (empty) | Only needed with `--profile cloud` |
| `BIND_ADDR`, `*_BIND_IP` | compose | `127.0.0.1` | Host interface for published ports (`RAG_CHAT_BIND_IP`, `RAG_INGEST_BIND_IP`, `OLLAMA_BIND_IP`, `WYOMING_{TTS,MISO,STT}_BIND_IP`, …) |

Some rag-chat APIs are unauthenticated: guard, `/metrics`, and the read-only and generative voice, pipeline and repair calls. Destructive and git actions (clone delete, label replace, repair `apply`/`git_ops`) require auth. See [api.md](api.md#authentication). Binding rag-chat beyond loopback exposes the open routes.

## rag-chat: chat and RAG

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_URL` | `http://ollama:11434` | |
| `OLLAMA_TIMEOUT` | `300` | Seconds per Ollama chat call |
| `OLLAMA_KEEP_ALIVE` | `15m` | The `ollama` service's default keep-alive (a duration, seconds, or `-1`; Ollama reads anything else as `5m`). Also the per-request value when `OLLAMA_REQUEST_KEEP_ALIVE` is unset |
| `OLLAMA_REQUEST_KEEP_ALIVE` | `OLLAMA_KEEP_ALIVE` | `keep_alive` rag-chat and rag-ingest send on each Ollama request. `server` (or `none`/`off`) sends none, leaving residency to the daemon. For argus-iris: `OLLAMA_REQUEST_KEEP_ALIVE=server` with `OLLAMA_KEEP_ALIVE` left a duration; iris extends `keep_alive` per its policy and holds pinned models at `-1` |
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

## rag-chat and rag-ingest: argus-iris GPU leases

With `ARGUS_GPU_COORDINATOR=broker`, every Ollama call holds an `ollama:<model>` lease from the argus-iris broker (`app/gpu_lease.py`, the same file in both services). The lease kinds are declared in `gpu-tenant.yaml`, and the manifest is registered with the broker on the first lease. If the grant is on the home node, the call goes to `OLLAMA_URL`; if it's on another node, it goes to that node's Ollama. A refused or timed-out lease is a **503** with `Retry-After`, and `run_with_oom_wait` never retries it as an OOM. A streamed chat or pipeline reports it in the stream instead. If the broker is unreachable, or answers with a 4xx (a bad token, say), the call goes to local Ollama unleased. Metadata calls (`/api/tags`, `/api/show`, `/api/ps`) and `/api/runtime/ollama/*` aren't leased.

| Variable | Default | Notes |
|---|---|---|
| `ARGUS_GPU_COORDINATOR` | `none` | `broker` turns leases on |
| `ARGUS_GPU_BROKER_URL` / `ARGUS_GPU_BROKER_TOKEN` | (empty) | The broker and the suite's caller token |
| `ARGUS_GPU_BROKER_FALLBACK` | `none` | When the broker is unreachable. `lease` (a host flock) would serialise every Ollama call |
| `ARGUS_GPU_HOME_NODE` | `dragon` (compose) | This stack's node: a grant here uses `OLLAMA_URL`, and pinned models stay here |
| `OLLAMA_LEASE_PINNED_MODELS` | `EMBED_MODEL` | Comma list; `none` pins nothing |
| `RAG_CHAT_GPU_LEASE_TIMEOUT_S` / `RAG_INGEST_GPU_LEASE_TIMEOUT_S` | `60` / `300` | Wait for a grant before the 503 (compose maps each to the service's `ARGUS_GPU_LEASE_TIMEOUT_S`) |
| `OLLAMA_LEASE_VRAM_MB` | nomic-embed-text `600` | `model=MiB,…`; anything unlisted is `OLLAMA_LEASE_DEFAULT_VRAM_MB` (`6000`) |
| `OLLAMA_LEASE_MODEL_MB` | qwen3:8b `5300`, nomic-embed-text `320` | `model=MiB,…`: the weights part of the footprint (Ollama `/api/ps` `size_vram`), sent as the lease's `model_mb` so iris doesn't ask for room again while the model is loaded. The rest is the working set; a larger `num_ctx` grows that, so raise `OLLAMA_LEASE_VRAM_MB`. Unlisted models send none (all working set). Keep `gpu-tenant.yaml`'s kinds in step |
| `OLLAMA_NODE_URLS` | (empty) | `node=url,…` when the broker's `base_url` for a node doesn't resolve in the containers |

Priority comes from the call site (`chat`, `embed-query` and `pipeline` are `interactive`; `repair` and `embed-ingest` are `batch`). A request's `X-Argus-Priority` header (`realtime`, `interactive`, `batch`, `background`) overrides it.

## rag-chat: media generation

| Variable | Default | Notes |
|---|---|---|
| `MEDIA_BACKEND` | `local` | `local` (files under `MEDIA_DIR`) or `minio` |
| `MEDIA_DIR` | `/data/media` | |
| `MEDIA_S3_ENDPOINT_URL`, `MEDIA_S3_BUCKET`, `MEDIA_S3_REGION`, `MEDIA_S3_ACCESS_KEY_ID`, `MEDIA_S3_SECRET_ACCESS_KEY`, `MEDIA_S3_USE_SSL`, `MEDIA_S3_PREFIX` | | Used when `MEDIA_BACKEND=minio` |
| `COMFYUI_URL` | `http://comfyui:8188` | ComfyUI provider endpoint |

## rag-chat: guard

`/api/guard/check` is a proxy to argus-retina's SingGuard server (`ghcr.io/smk762/argus-retina-singguard`, container port 8080, host port 8134 in retina's compose). Retina holds the model, queues checks one at a time and takes its own GPU lease, so composer runs no SingGuard and leases nothing for it.

| Variable | Default | Notes |
|---|---|---|
| `GUARD_URL` | (empty) | Retina's SingGuard base URL, e.g. `http://<GPU_HOST>:8134`. Empty: `/api/guard/check` answers 503 "guard not configured" |
| `GUARD_TIMEOUT` | `120` | Seconds to wait for a check (slow and cascade take about 20 s on a 3090, plus queueing); a timeout is a 504 |
| `GUARD_THINKING` | `fast-slow` | `fast`, `fast-slow`, `slow` or `cascade`, sent when a check leaves `thinking` blank (retina's own default is `fast`) |

## rag-chat: voice

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_URL` | `http://whisper:8031` | STT engine (argus-echo) |
| `XTTS_TTS_URL` | `http://xtts:8031` | Local XTTS engine |
| `MISO_TTS_URL` | (empty = disabled) | Remote Miso engine, e.g. `http://<GPU_HOST>:8034` |
| `TTS_URL` | (empty) | Legacy knob: a URL that isn't a named engine is registered as engine `tts` |
| `TTS_DEFAULT_ENGINE` | auto: `miso` if set, else `xtts` | |
| `TTS_FALLBACK_ENGINE` | auto: `xtts` when the default is something else | Used only for requests that don't name an engine |
| `XTTS_NODE_URLS` | (empty) | With `ARGUS_GPU_COORDINATOR=broker`, XTTS synth leases the `xtts` target from argus-iris (#79). A grant on another node calls the URL given here (`node=url,…`), otherwise iris's `base_url` for it. A grant on the home node calls `XTTS_TTS_URL`. A voice clone stays on the home node |
| `XTTS_LEASE_CPU_FALLBACK` | `1` | Lease refused or timed out on the last engine in the chain: `1` sends `device: "cpu"` to the local XTTS, `0` answers 503 with the broker's reason |
| `XTTS_LEASE_VRAM_MB` / `XTTS_LEASE_MODEL_MB` | `2000` / `1800` | Footprint and weights each XTTS lease declares (match `gpu-tenant.yaml`'s `tts-synth`) |
| `VOICE_TIMEOUT` | `30` | Proxy timeout (seconds) |
| `VOICE_MAX_CLIPS` / `VOICE_CLIP_MAX_BYTES` | `16` / `16777216` | Clips per `/api/voice/clone` or `/api/voice/analyze` request, and bytes per clip; more is a 413 |
| `MISO_SENTENCE_PAUSE_MS` / `MAX_SENTENCE_PAUSE_MS` | `300` / `2000` | Silence inserted between Miso sentences |
| `S3_ENDPOINT_URL`, `S3_PUBLIC_URL`, `S3_BUCKET_NAME`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` | | The "voice" MinIO bucket for rag-chat's re-hosted and stitched audio. A clip already served from `S3_PUBLIC_URL`'s origin (any bucket there) isn't re-hosted |

## Voice engines (argus-echo)

The engines run argus-echo's images (`voice.md`, #81). Engine settings are echo's: see its docs (`docs/xtts.md`, `docs/whisper.md`). The compose-level ones:

| Variable | Default | Notes |
|---|---|---|
| `ECHO_TAG` | `0.2.0` | argus-echo release for every engine and bridge image |
| `XTTS_S3_BUCKET` / `XTTS_S3_PUBLIC_URL` | `voice-xtts` / required | XTTS clones and rendered audio; the bucket echo's XTTS on morpheus shares |
| `XTTS_CPU_FALLBACK` / `XTTS_OOM_RETRY_AFTER_S` | `1` / `30` | A GPU promotion that runs out of CUDA memory runs on the CPU (`1`) or answers 503 with the reason |
| `STT_MAX_UPLOAD_MB` | `100` | Largest audio Whisper accepts |
| `STT_AUDIO_URL_ALLOWED_HOSTS` | (empty) | Internal hosts `/transcriptions` may fetch `audio_url` from |
| `WYOMING_STT_MAX_SECONDS` | `120` | Longest utterance the Home Assistant STT bridge buffers |

## rag-chat: sidecars and repair

| Variable | Default | Notes |
|---|---|---|
| `MODERNBERT_URL` | `http://modernbert:7998` | Target of the `/api/pipeline/*` proxy |
| `INFINITY_URL` | `http://infinity:7997` | Target of the `/api/runtime/infinity/*` controls, and the embedder for the repair loop's related-code search |
| `AUDIT_API_URL` | `http://127.0.0.1:8765` | ai-code-auditor API for diff validation between repair iterations; empty disables validation |
| `AI_AUDIT_PUBLIC_BASE_URL` | falls back to `AUDIT_API_URL` | Base URL for fetching `pack_id` packs |
| `AUDIT_PACK_TIMEOUT_S` | `10` | Per-request timeout for pack fetches |
| `AUDIT_API_ALLOWED_HOSTS` | (empty) | Extra origins a repair request's `audit_api_url` / `audit_api_base` may name: comma-separated `host` (any port) or `host:port`. There is no wildcard: `*` is ignored with a warning. The configured `AUDIT_API_URL` and `AI_AUDIT_PUBLIC_BASE_URL` origins are always allowed; link-local hosts never are. Others get a 422. |
| `REPAIR_PACK_STATUS_CALLBACK` | `1` | Report `resolved` to the audit API after a clean applied `auto_fix` |
| `AI_AUDIT_API_TOKEN` | (empty) | Sent as `X-API-Token` on audit API writes |
| `ECOSYSTEM_CONFIG_PATH` | `/home/smk/ai-code-auditor/config/ecosystem.yaml` | Resolves repo paths for `apply` and `compare_branch` |

## rag-ingest

| Variable | Default | Notes |
|---|---|---|
| `RETRIEVAL_PROFILE` | `text` | Default ingest path for docs with no `kind` and no recognised `path`: `text` or `code` |
| `QDRANT_COLLECTION` | `audit_docs` | Text-path collection |
| `QDRANT_COLLECTION_CODE` | `code_audit` | Code-path collection (named vectors). rag-chat's repair loop searches it too, so both services get it from the same variable |
| `EMBED_BACKEND_URL` | `http://infinity:7997` | Infinity endpoint (code path) |
| `EMBED_MODEL` | `nomic-embed-text` | Text-path embedding model, always served by Ollama. rag-chat embeds queries with the same Ollama model, so set it identically in both services. (`EMBED_BACKEND` is retired: it is ignored, with a warning, if set to anything but `ollama`.) |
| `EMBEDDER_CHECK_TTL_S` | `60` | rag-chat: how often the RAG collection's `embedder` labels are re-checked |
| `EMBED_TASK_PREFIXES` | `none` | Both services: `nomic` embeds chunks as `search_document: …` and queries as `search_query: …`. Changes the vector space, so migrate with `python -m app.reembed` rather than flipping it on a populated collection (see docs/api.md). |
| `EMBEDDER_STRICT` | `0` | Both services: `1` treats points without an `embedder` label as a mismatch |
| `EMBED_MODEL_CODE` | `nomic-ai/CodeRankEmbed` | Code-path embedding model (also preloaded by the `infinity` service). rag-chat embeds repair code-search queries with it, so both services share it |
| `EMBED_CODE_QUERY_PREFIX` | CodeRankEmbed's `Represent this query for searching relevant code: ` (empty for other models) | rag-chat: instruction prepended to code-search queries; stored code is unprefixed |
| `HYBRID_SPARSE` | `0` | `1` adds a `bm25` sparse vector to code points |
| `INGEST_CHUNK_SIZE` / `INGEST_CHUNK_OVERLAP` | `1000` / `120` | Text-path window, in characters |
| `CODE_CHUNK_MIN_CHARS` / `CODE_CHUNK_MAX_CHARS` | `120` / `3000` | Code-path chunk bounds |
| `EMBED_MAX_CHARS` / `EMBED_MIN_CHARS` | `4000` / `200` | Embedder input bounds |
| `EMBED_TIMEOUT_S` / `EMBED_RETRIES` | `120` / `6` | |
| `INGEST_UPSERT_BATCH_SIZE` | `64` | |
| `INGEST_NONCE_STORE` | `sqlite` | `sqlite` (`INGEST_NONCE_DB`, default `/data/nonces.db`) or `redis` (`INGEST_REDIS_URL`, `INGEST_NONCE_KEY_PREFIX`). With `redis` unreachable, signed requests and `/health` answer 503 until it is back |
| `INGEST_NONCE_ALLOW_SQLITE_FALLBACK` | `0` | `1`: if Redis is unreachable at startup, fall back to this node's SQLite store (not shared between replicas; logged as an error, `/health` says `sqlite-fallback`) |

## modernbert

| Variable | Default | Notes |
|---|---|---|
| `MODERNBERT_MODEL` | `answerdotai/ModernBERT-base` | |
| `MODERNBERT_TASK` | `zero_shot` | Or a fine-tuned classification head |
| `MODERNBERT_DEVICE` | `auto` | |
| `MODERNBERT_KEEP_ALIVE_GPU` / `MODERNBERT_KEEP_ALIVE_CPU` | `300` / `0` | Idle seconds before moving down a stage (`gpu → cpu → unloaded`). `-1` stays in that stage. The CPU stage follows any demotion, an `/admin/evict` included, so `_GPU=-1` (residency handed to argus-iris) still frees RAM after `_CPU`. The same holds for `XTTS_KEEP_ALIVE_*` |
| `MODERNBERT_PRELOAD` | `0` | Load the model at startup |
| `MODERNBERT_MAX_LENGTH`, `MODERNBERT_TOP_K`, `MODERNBERT_THRESHOLD` | `512`, `10`, `0.05` | Classifier defaults |

## infinity (infinity-manager)

infinity-manager is a small proxy on port 7997. It starts an Infinity child process on the first request and stops it after the idle timeout, which frees VRAM.

| Variable | Default | Notes |
|---|---|---|
| `INFINITY_MODEL_IDS` | `nomic-ai/CodeRankEmbed` (compose: `EMBED_MODEL_CODE`) | Models the child loads, comma-separated. Add `BAAI/bge-reranker-v2-m3` to serve `/rerank` (+1.26 GB VRAM); nothing calls it today |
| `INFINITY_IDLE_TIMEOUT` | `900` (compose: `300`) | Seconds idle before the child is stopped |
| `INFINITY_PRELOAD` | `0` | Start the child at boot |
| `INFINITY_STARTUP_TIMEOUT` / `INFINITY_REQUEST_TIMEOUT` | `180` / `300` | |
| `INFINITY_ENGINE` / `INFINITY_DEVICE` | `torch` / `cuda` | |
| `INFINITY_EXTRA_ARGS` | (empty; compose: `--batch-size 8`) | Extra CLI args for the child. Batch size drives peak VRAM: CodeRankEmbed peaks at 1.3 GB with batch 8 and 3.5 GB with batch 32 on 4000-char chunks, and the peak stays reserved until the idle stop |
| `INFINITY_CHILD_CMD` | (empty) | Full override of the child command |
| `INFINITY_MANAGER_HOST`/`_PORT`, `INFINITY_CHILD_HOST`/`_PORT` | `0.0.0.0:7997`, `127.0.0.1:7998` | Internal wiring; rarely changed |

## Client scripts (`scripts/`)

| Variable | Default | Notes |
|---|---|---|
| `CHAT_URL` | `http://127.0.0.1:9150` | Base URL; scripts append `/chat` and `/health` |
| `INGEST_URL` | `http://127.0.0.1:9050` | Base URL; `test_endpoints.py` appends `/health`. Signed requests go to `${INGEST_URL}/ingest`. |
| `RAG_API_KEY` (or `CHAT_API_KEY`) | (empty) | Bearer key for `/chat`. Required unless the server runs with `DEV_AUTH_BYPASS=1`. |
| `INGEST_SHARED_SECRET` | | For signing ingest requests |

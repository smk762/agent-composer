# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Stack overview

Three FastAPI services plus sidecars, all wired together in `docker-compose.yml`:

| Service | Port | Role |
|---------|------|------|
| `rag-ingest` | 9050 | Signed ingestion API → chunk → embed → upsert Qdrant |
| `rag-chat` | 9150 | Chat gateway (Ollama + optional RAG), history, UI, pipeline proxy |
| `modernbert` | 7998 | ModernBERT classifier sidecar (lazy-load GPU, zero-shot or fine-tuned) |
| `qdrant` | 6333/6334 | Vector store on the stateful node (`<STATEFUL_HOST>`; not in this compose) |
| `postgres` | 5432 | Chat history DB on the stateful node (AsyncSQLAlchemy + Alembic; not in this compose) |
| `infinity` | 7997 | Infinity sidecar serving CodeRankEmbed + bge-reranker (code-audit path) |
| `ollama` | 11434 | Local LLM runtime (internal only) |
| `whisper` | 8030 | faster-whisper STT sidecar (GPU, CTranslate2, large-v3-turbo) |
| `xtts` | 8033→8031 | XTTS-v2 sidecar (GPU, zero-shot cloning + streaming) — local TTS engine |
| `miso` | (remote) | Miso One sidecar (8B, expressive + one-shot clone) on a separate GPU host (e.g. `morpheus:8034`); reached via `MISO_TTS_URL`, not run in this compose |
| `tts` | 8031 | Qwen3-TTS sidecar (GPU, 0.6b Base + CustomVoice) — legacy/rollback |
| `wyoming-xtts` | 10200 | Wyoming TTS bridge → XTTS (Home Assistant `tts` engine, incl. clones) |
| `wyoming-miso` | 10201 | Wyoming TTS bridge → Miso (second HA `tts` engine; disabled until `MISO_TTS_URL` set) |
| `wyoming-whisper` | 10300 | Wyoming STT bridge → Whisper (Home Assistant `stt` engine) |

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

Copy `.env.example` → `.env`. Hosts in this file are placeholders (`<STATEFUL_HOST>`, `<AGENT_HOST>`, …); the real host/IP is in the local `.env`, so check it there. Key variables:

- `INGEST_SHARED_SECRET` — required for rag-ingest HMAC signing
- `QDRANT_URL` — defaults to `http://<STATEFUL_HOST>:6333` (Qdrant, Postgres and MinIO run on the stateful node, not in this compose file; `S3_ENDPOINT_URL` / `S3_PUBLIC_URL` default there too)
- `DATABASE_URL` — Postgres on the stateful node (compose default `postgresql://ragchat:ragchat@<STATEFUL_HOST>:5432/ragchat`)
- `RETRIEVAL_PROFILE` — `text` (default, Ollama) or `code` (tree-sitter + Infinity)
- `MODERNBERT_URL` — defaults to `http://modernbert:7998`
- `AUDIT_API_URL` — ai-code-auditor audit API for diff-audit validation between repair iterations (default `http://127.0.0.1:8765`)
- `ECOSYSTEM_CONFIG_PATH` — path to `ai-code-auditor/config/ecosystem.yaml` for repo path resolution (default `/home/smk/ai-code-auditor/config/ecosystem.yaml`)
- `AI_AUDIT_PUBLIC_BASE_URL` — audit API base for `pack_id`-driven repairs (falls back to `AUDIT_API_URL`, then `http://127.0.0.1:8765`); `AUDIT_PACK_TIMEOUT_S` (default 10) bounds each pack fetch; `REPAIR_PACK_STATUS_CALLBACK` (default on; `0` disables) makes an applied, clean `auto_fix` POST `{status: resolved}` to `/audit/packs/{id}/status` (fails soft; `AI_AUDIT_API_TOKEN` is sent as `X-API-Token`)
- `GUARD_MODEL` — SingGuard tag for content-safety checks (default `hf.co/inclusionAI/SingGuard-4b-GGUF:Q4_K_M`, runs on shared Ollama). `GUARD_THINKING` is `fast-slow` (label, triggered rule, and a written reason), `fast` (label and rule), `slow` (reason first, then the verdict), or `cascade` (`fast`, then `slow` only when fast flags the content). `GUARD_WORKER_CONCURRENCY` (default 1) is how many of those checks run at once.
- TTS engine registry — TTS is a registry of interchangeable engines (all speak the XTTS sidecar contract), selectable per request. `XTTS_TTS_URL` (default `http://xtts:8031`, local GPU), `MISO_TTS_URL` (remote Miso host, e.g. `http://<morpheus-ip>:8034`, empty = disabled), `TTS_DEFAULT_ENGINE` (`xtts`|`miso`|`tts`, default `xtts`). Legacy `TTS_URL` is still honoured as a fallback engine for Qwen rollback. `WHISPER_URL` for STT. `VOICE_TIMEOUT` for proxy timeouts.
- `CF_TUNNEL_TOKEN` — only needed when running `--profile cloud`

GPU passthrough requires NVIDIA Container Toolkit on the host. Remove `gpus: all` from compose services to run CPU-only.

## Architecture: two ingest paths

`rag-ingest/app/ingest_pipeline.py` branches per document:

- **Text path** (`RETRIEVAL_PROFILE=text` or `kind: "text"`): slides a char-level window over the document, embeds each chunk via Ollama, upserts into `QDRANT_COLLECTION` using an unnamed dense vector (legacy layout). A request may name another text `collection`, but only one listed in `INGEST_ALLOWED_COLLECTIONS` (anything else is a 400, never a silent redirect); ai-code-auditor sends `audit_docs`.
- **Code path** (`RETRIEVAL_PROFILE=code` or `kind: "code"`, or inferred from `path`): splits via tree-sitter in `code_chunker.py` (per function/class), embeds via Infinity in `embeddings.py`, upserts into `QDRANT_COLLECTION_CODE` using named vectors (`dense` + optional `bm25`). These two collections are completely separate so both paths can coexist.
- **Replace on re-send:** a doc with an `id` (code: else `repo`+`path`) is identified by that plus `source`. Points are upserted only after the whole request is embedded; then `_prune_replaced` deletes the identity's older points (`ingest_seq` below the request's) that the request didn't write (`pruned` in the response). A failed request writes and deletes nothing; id-less docs stay append-only. One identity twice in a request is a 400 (`IngestRequestError`). Requests touching the same identity are serialised by per-identity asyncio locks (`_identity_locks`, in-process: rag-ingest runs one uvicorn worker). Points carry `ingested_at` and `ingest_seq`. Signed `POST /delete {doc_ids?, paths?+repo (code), source, collection?, kind?}` retires docs (`delete_docs`).

`embeddings.py` defines `OllamaEmbeddingClient` and `InfinityEmbeddingClient`, both conforming to the `EmbeddingClient` protocol. Each path has exactly one: text always uses Ollama (the backend rag-chat embeds queries with, so both share one vector space) and code always uses Infinity. `EMBED_BACKEND` is retired (ignored with a warning). Every point carries `embedder` (`TEXT_EMBEDDER` = `ollama:<EMBED_MODEL>`, `CODE_EMBEDDER` = `infinity:<EMBED_MODEL_CODE>`); `_check_embedder` makes `/ingest` answer 409 (`EmbedderMismatchError`) rather than add to a collection labelled by another embedder. Unlabelled (older) points are accepted.

## Architecture: rag-chat

`rag-chat/app/main.py` assembles these routers: `chat`, `api_keys`, `ui`, `generation`, `media`, `providers`, `pipeline`, `repair`, `guard`, `voice`, `search`, `metrics`.

- **Chat** (`routers/chat.py`): streams Ollama responses, optionally injects Qdrant RAG context as a system message, persists conversation + messages to Postgres via async SQLAlchemy.
- **Retrieval** (`retrieval.py`, `routers/search.py`): both the chat's RAG context and `POST /search` (auth) query `QDRANT_COLLECTION`, default `audit_docs` (ai-code-auditor's wiki sections and agent packs). `/search` takes `query`, `top_k` (≤20) and tag filters (`repo` → `repo:<name>`, `kind` → `kind:<kind>`, `tags`; a hit must carry all of them) and returns hits with title, url, tags and text. Chat context heads each passage with its title and link. `retrieval.embedder_status()` checks the collection's `embedder` labels against rag-chat's `TEXT_EMBEDDER` (startup, then cached `EMBEDDER_CHECK_TTL_S`): on a mismatch `/search` is a 409 and chat skips RAG context; `/health` shows the last check.
- **Pipeline proxy** (`routers/pipeline.py`): reverse-proxies `/api/pipeline/*` → ModernBERT sidecar. Exposes classify, load/evict/unload, and labels CRUD.
- **Repair** (`routers/repair.py` + `repair/`): multi-model agentic repair loop. `POST /api/repair/run` accepts a diff or `compare_branch`, runs fixer→critic→adversary iterations, validates via ai-code-auditor audit API, and streams SSE events. `GET /api/repair/jobs/{job_id}` retrieves job state. The job runs in its own asyncio task and the SSE response only relays its queue, so a client disconnect never strands a half-done apply; the 200-job store evicts only finished jobs (503 when all are running). Only a patch no validation rejected is applied or reported as resolved (`patch_validated` on `complete`). Key env: `AUDIT_API_URL`, `ECOSYSTEM_CONFIG_PATH`. **Pack-driven:** send `pack_id` (optional `audit_api_base`) and the router fetches `GET /audit/packs/{id}` + `/brief`, injects the pack (title, location, acceptance criteria, suggested tests, constraints, brief) at the top of every iteration's context via `repair/context_builder.py`, and streams `pack_loaded` / `warning` SSE events. `repo` may be omitted (taken from the pack); on a repo mismatch read-only runs use the pack's repo with a warning, while `apply`/`git_ops` runs are rejected. Pack fetches time out (`AUDIT_PACK_TIMEOUT_S`) and fail soft — repair continues without pack context. Without `diff`/`compare_branch`, a pack alone is accepted. Example: `curl -N -X POST http://127.0.0.1:9150/api/repair/run -H 'Content-Type: application/json' -d '{"pack_id":"ext-retry-core-001","mode":"review"}'`. **Auth:** runs with `apply: true` or `git_ops.enabled` go through `require_user` (401 before any work); read-only runs stay open. Likewise `DELETE /api/voice/clones/{id}` and `PUT /api/pipeline/labels` require auth — destructive/git actions only.

  <!-- ai-audit:repair-pack-notes:start -->
  **Audit API base & pack_id repair.** Packs come from ai-code-auditor (set `AI_AUDIT_PUBLIC_BASE_URL` to `http://<AUDIT_HOST>:8765`; the real host/IP is in the local `.env`, see `.env.example`; resolution order: request `audit_api_base` → `AI_AUDIT_PUBLIC_BASE_URL` → `AUDIT_API_URL` → `http://127.0.0.1:8765`). The CLI equivalent is `ai-audit repair --pack-id <id> [--audit-api-base URL]`. Consuming-agent flow: `GET /audit/packs/bundle?repo=<name>` → `POST /audit/packs/{id}/status {"status":"claimed"}` → repair with `pack_id` → `POST /audit/validate_patch` with `pack_id` (pass ⇒ `resolved`). See `AGENTS.md` and `.cursor/rules/agent-coding-packs.mdc`.
  **SSRF note:** `audit_api_base` is client-supplied and the server issues `GET {audit_api_base}/audit/packs/{id}[/brief]` (and the resolved status callback) to it. Only expose `/api/repair/run` to trusted callers. Only `http(s)` URLs are accepted and fetches time out (`AUDIT_PACK_TIMEOUT_S`), but there is no host allowlist yet — **TODO:** restrict `audit_api_base` to an allowlist (e.g. `AUDIT_API_ALLOWED_HOSTS`) and ignore the field otherwise.
  <!-- ai-audit:repair-pack-notes:end -->
- **Guard** (`routers/guard.py` + `guard_policy.py` + `guard_image.py`): SingGuard checks via shared Ollama. The policy is natural-language request input (`policy`, default taxonomy when omitted). `POST /api/guard/check` returns the safe/unsafe label, the triggered rule title, and why that rule applies. `thinking` is `fast`, `fast-slow`, `slow` (reason first), or `cascade` (fast, then slow on flags; the response `thinking` is the pass that stood). Text is checked with a raw prompt; `image_base64` (raw base64 or a data URL, with the prompt as `input`) is judged as pixels via Ollama `/api/chat`, attached to the user turn. Images are re-encoded to RGB JPEG (or PNG with alpha) before Ollama sees them: its vision path is pickier than Pillow, and cull sends frames cortex left as-is when they were already small. Checks queue on `GUARD_WORKER_CONCURRENCY` (default 1) so cull scans and darkroom pre-flights share one model. The generate page's "Check image" button sends the picture plus the prompt that produced it. `GET /api/guard/categories` and `GET /api/guard/policy` expose the default rules and the supported thinking modes. Key env: `GUARD_MODEL`, `GUARD_THINKING`, `GUARD_WORKER_CONCURRENCY`.
- **Voice** (`routers/voice.py`): proxies STT/TTS requests to the whisper and TTS containers. `POST /api/voice/transcribe` (multipart audio → text), `POST /api/voice/synthesise` (text → audio URL), `POST /api/voice/synthesise/stream` (SSE sentence-chunked TTS, one WAV/sentence), `POST /api/voice/synthesise/pcm` (raw 24 kHz float32 PCM passthrough for live browser playback; XTTS only), `POST /api/voice/analyze` (multipart clips → per-clip quality/consistency score, recommended subset + best-first ordering; no clone created), `POST /api/voice/clone` (multipart, one or more reference clips → voice clone; XTTS averages the speaker embedding across clips), `GET /api/voice/clones` (list stored clones), `GET /api/voice/speakers`, `GET /api/voice/health`. Voice cloning UI lives at `/ui/voice-clone`; cloned voices are stored in MinIO at `voice_clones/{companion_id}/` (`prompt.pt` latents + `meta.json`) and reusable via `voice_clone_id`. The XTTS sidecar standardises each reference clip before computing latents (mono → resample → cascaded high-pass → Silero VAD split into speech+gap regions → SNR-gated stationary denoise of HVAC hum, seeded from the VAD gaps via `noisereduce` → LUFS normalise) and returns a per-clip report (kept seconds, SNR, noise floor, gain, denoise applied). Toggle/tune via `XTTS_PREPROCESS`, `XTTS_PREPROCESS_VAD`, `XTTS_PREPROCESS_HIGHPASS_HZ`/`_ORDER`, `XTTS_PREPROCESS_DENOISE`/`_PROP`/`_SNR_DB`, `XTTS_PREPROCESS_LUFS`, `XTTS_PREPROCESS_MAX_SECONDS`. Conditioning-latent windows are tuned via `XTTS_GPT_COND_LEN` (default 30; XTTS's own default of 6 under-uses multi-clip sets), `XTTS_GPT_COND_CHUNK_LEN`, `XTTS_MAX_REF_LEN`. `POST /analyze-clips` scores candidate clips (SNR, clipping, speech seconds, and speaker-embedding consistency vs the centroid) and returns a recommended subset (threshold `XTTS_ANALYZE_THRESHOLD`) plus the best-first ordering used to fill the GPT conditioning window; the `/ui/voice-clone` page surfaces this as per-clip scores + checkboxes. **TTS is a multi-engine registry** (`config.TTS_ENGINES`, `tts_engine_url()`): every TTS endpoint accepts an `engine` selector (JSON field on synth requests, `?engine=` query param elsewhere) and `GET /api/voice/engines` lists the configured engines + the default; `/ui/voice` and `/ui/voice-clone` render an engine picker. Engines all speak the same sidecar contract — `xtts` (local XTTS-v2, ~2 GB VRAM, RTF ~0.2, multi-clip averaging) and `miso` (remote Miso One on `MISO_TTS_URL`, expressive + one-shot cloning) coexist; `tts` (legacy Qwen on `TTS_URL`) is the rollback. **Miso idle lifecycle** (set on the Miso host, not this compose): like the other sidecars it runs `unloaded → cpu → gpu`; `MISO_KEEP_ALIVE_GPU` (default 300 s) is idle time on GPU, `MISO_KEEP_ALIVE_CPU` (default `0`) is idle time parked on CPU before a full unload — `0` skips the CPU stage (the 8B model holds ~17 GB RSS on CPU, enough to trip `earlyoom` on a shared host), `-1` pins either stage. A CUDA OOM on load keeps no partial copy and returns 503 with `Retry-After` (`MISO_OOM_RETRY_AFTER_S`); `/health` reports `model_state`, `rss_mb` and VRAM. `/health` fans out to every engine. Default chosen by `TTS_DEFAULT_ENGINE`; `tts_engine_chain()` appends `TTS_FALLBACK_ENGINE` (auto = local xtts) so default-path synth/stream/pcm requests transparently fall back when the preferred engine's host is unreachable (an explicitly selected engine is never silently switched; clone/list ops are engine-specific and don't fall back). Optimal clone input per engine is documented in `voice-stack/VOICE_CLONING.md`. Key env: `WHISPER_URL`, `XTTS_TTS_URL`, `MISO_TTS_URL`, `TTS_DEFAULT_ENGINE`, `TTS_FALLBACK_ENGINE`, `TTS_URL`.
- **Wyoming bridges** (`voice-stack/wyoming_bridge/`): CPU-only protocol adapters that expose the voice stack to Home Assistant as native engines. `wyoming-xtts` (port `10200`) translates Wyoming `synthesize` → XTTS `POST /synthesise/stream`, converting the 24 kHz float32 PCM to signed 16-bit AudioChunks; it advertises the `XTTS_DEFAULT_SPEAKER` plus every stored clone as `clone:<companion_id>` (set `WYOMING_TTS_LIST_SPEAKERS=1` to also enumerate all built-in studio speakers). Selecting a clone resolves to XTTS's full `voice_clone_id` S3 key (`voice_clones/<id>/prompt.pt`) — passing the bare id returns 422. `wyoming-whisper` (port `10300`) buffers the HA audio stream, resamples to 16 kHz mono, and POSTs to Whisper `POST /transcribe`, returning a Wyoming `transcript`. Both run from the shared `Dockerfile.wyoming` (entrypoint chosen via compose `command`) and reach the sidecars over the internal docker network. The TTS bridge module (`xtts_tts.py`) is engine-agnostic — it targets whatever `TTS_URL` (or legacy `XTTS_URL`) points at, with a configurable program name/attribution — so the same module also powers **`wyoming-miso`** (port `10201`, advertised as program `miso`, pointed at `MISO_TTS_URL`); HA can add XTTS and Miso as independent `tts` engines. Add them in HA via **Settings → Devices & Services → Wyoming Protocol** (host = this stack's LAN IP, ports `10200`/`10201`/`10300`). Key env: `WYOMING_XTTS_URL`, `WYOMING_MISO_URL`, `WYOMING_WHISPER_URL`, `XTTS_DEFAULT_SPEAKER`, `WYOMING_TTS_PROGRAM_NAME`/`_DESC`/`_ATTRIBUTION_{NAME,URL}`, `WYOMING_TTS_DEFAULT_VOICE`, `WYOMING_{TTS,STT,MISO}_BIND_IP`/`_PORT`.
- **UI** (`routers/ui.py`): server-rendered HTML pages for chat, history, API keys, image generation, Pipeline Lab, and Guard tester. All pages share `render_page()` / `BASE_CSS`.
- **DB** (`db.py` + `models/orm.py`): async Postgres via `asyncpg` driver. `bootstrap_db.py` auto-creates the DB on startup; `init.sh` runs `alembic upgrade head` before uvicorn.

## Architecture: modernbert sidecar

`modernbert/app/model.py` manages lazy GPU loading with three states: `unloaded → cpu → gpu`. Keep-alive timers (`MODERNBERT_KEEP_ALIVE_GPU`, `MODERNBERT_KEEP_ALIVE_CPU`) evict the model after idle. `/health` always returns 200; check `model_state` for actual readiness.

## Request signing (rag-ingest)

Every `POST /ingest` requires:
- `X-Timestamp`, `X-Nonce`, `X-Signature` headers
- Signature: `HMAC-SHA256(sha256(secret), "{ts}.{nonce}.{sha256(body)}")`

See `scripts/example_tool_calling_client.py` and `scripts/test_endpoints.py` for working Python examples.

## Cloudflare

`cloudflared` is on the `cloud` compose profile and only starts with `docker compose --profile cloud up`. For LAN-only dev, omit the profile. Published ports bind to loopback by default; set `BIND_ADDR` (all services) or a per-service `RAG_INGEST_BIND_IP` / `RAG_CHAT_BIND_IP` / `OLLAMA_BIND_IP` / `WYOMING_{TTS,MISO,STT}_BIND_IP` to expose them on the LAN.

## Branches

`dev` is the default branch and the base for every PR. Cut feature branches from `origin/dev` and bring them up to date by merging `origin/dev`, never `main`. Never push work to `main` or open a PR against it, except the release PR from `dev`: `main` must never be ahead of `dev`, with one exception. A release fast-forwards `main` to a `dev` commit that passed CI (`git push origin origin/dev:main`), then tags it. If `main` is protected, the release is a reviewed PR from `dev` into `main`, and merging it leaves a merge commit on `main` alone. That merge commit is the one exception to the rule: it leaves `main` one commit ahead of `dev` with an identical tree. Leave it there and don't merge `main` back into `dev`. Anything else on `main` that `dev` lacks is a mistake (`git diff origin/dev origin/main` must be empty). Once `main` carries such a commit it can't be fast-forwarded, so later releases are PRs too.

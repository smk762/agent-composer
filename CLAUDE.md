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

Copy `env.example` → `.env`. Key variables:

- `INGEST_SHARED_SECRET` — required for rag-ingest HMAC signing
- `QDRANT_URL` — defaults to `http://qdrant:6333` (container name)
- `DATABASE_URL` — set automatically by compose to the internal postgres container
- `RETRIEVAL_PROFILE` — `text` (default, Ollama) or `code` (tree-sitter + Infinity)
- `MODERNBERT_URL` — defaults to `http://modernbert:7998`
- `AUDIT_API_URL` — ai-code-auditor audit API for diff-audit validation between repair iterations (default `http://127.0.0.1:8765`)
- `ECOSYSTEM_CONFIG_PATH` — path to `ai-code-auditor/config/ecosystem.yaml` for repo path resolution (default `/home/smk/ai-code-auditor/config/ecosystem.yaml`)
- `GUARD_MODEL` — LlamaGuard model tag for content-safety checks (default `llama-guard3:8b`, runs on shared Ollama)
- TTS engine registry — TTS is a registry of interchangeable engines (all speak the XTTS sidecar contract), selectable per request. `XTTS_TTS_URL` (default `http://xtts:8031`, local GPU), `MISO_TTS_URL` (remote Miso host, e.g. `http://<morpheus-ip>:8034`, empty = disabled), `TTS_DEFAULT_ENGINE` (`xtts`|`miso`|`tts`, default `xtts`). Legacy `TTS_URL` is still honoured as a fallback engine for Qwen rollback. `WHISPER_URL` for STT. `VOICE_TIMEOUT` for proxy timeouts.
- `CF_TUNNEL_TOKEN` — only needed when running `--profile cloud`

GPU passthrough requires NVIDIA Container Toolkit on the host. Remove `gpus: all` from compose services to run CPU-only.

## Architecture: two ingest paths

`rag-ingest/app/ingest_pipeline.py` branches per document:

- **Text path** (`RETRIEVAL_PROFILE=text` or `kind: "text"`): slides a char-level window over the document, embeds each chunk via Ollama, upserts into `QDRANT_COLLECTION` using an unnamed dense vector (legacy layout).
- **Code path** (`RETRIEVAL_PROFILE=code` or `kind: "code"`, or inferred from `path`): splits via tree-sitter in `code_chunker.py` (per function/class), embeds via Infinity in `embeddings.py`, upserts into `QDRANT_COLLECTION_CODE` using named vectors (`dense` + optional `bm25`). These two collections are completely separate so both paths can coexist.

`embeddings.py` defines `OllamaEmbeddingClient` and `InfinityEmbeddingClient` both conforming to the same `EmbeddingClient` protocol. `make_client()` selects based on `EMBED_BACKEND`.

## Architecture: rag-chat

`rag-chat/app/main.py` assembles these routers: `chat`, `api_keys`, `ui`, `generation`, `media`, `providers`, `pipeline`, `repair`, `guard`, `voice`, `metrics`.

- **Chat** (`routers/chat.py`): streams Ollama responses, optionally injects Qdrant RAG context as a system message, persists conversation + messages to Postgres via async SQLAlchemy.
- **Pipeline proxy** (`routers/pipeline.py`): reverse-proxies `/api/pipeline/*` → ModernBERT sidecar. Exposes classify, load/evict/unload, and labels CRUD.
- **Repair** (`routers/repair.py` + `repair/`): multi-model agentic repair loop. `POST /api/repair/run` accepts a diff or `compare_branch`, runs fixer→critic→adversary iterations, validates via ai-code-auditor audit API, and streams SSE events. `GET /api/repair/jobs/{job_id}` retrieves final state. Key env: `AUDIT_API_URL`, `ECOSYSTEM_CONFIG_PATH`.
- **Guard** (`routers/guard.py`): LlamaGuard content-safety checks via shared Ollama. `POST /api/guard/check` accepts input text + optional conversation context, returns safe/unsafe verdict with flagged category codes. `GET /api/guard/categories` returns the taxonomy. Key env: `GUARD_MODEL`.
- **Voice** (`routers/voice.py`): proxies STT/TTS requests to the whisper and TTS containers. `POST /api/voice/transcribe` (multipart audio → text), `POST /api/voice/synthesise` (text → audio URL), `POST /api/voice/synthesise/stream` (SSE sentence-chunked TTS, one WAV/sentence), `POST /api/voice/synthesise/pcm` (raw 24 kHz float32 PCM passthrough for live browser playback; XTTS only), `POST /api/voice/analyze` (multipart clips → per-clip quality/consistency score, recommended subset + best-first ordering; no clone created), `POST /api/voice/clone` (multipart, one or more reference clips → voice clone; XTTS averages the speaker embedding across clips), `GET /api/voice/clones` (list stored clones), `GET /api/voice/speakers`, `GET /api/voice/health`. Voice cloning UI lives at `/ui/voice-clone`; cloned voices are stored in MinIO at `voice_clones/{companion_id}/` (`prompt.pt` latents + `meta.json`) and reusable via `voice_clone_id`. The XTTS sidecar standardises each reference clip before computing latents (mono → resample → cascaded high-pass → Silero VAD split into speech+gap regions → SNR-gated stationary denoise of HVAC hum, seeded from the VAD gaps via `noisereduce` → LUFS normalise) and returns a per-clip report (kept seconds, SNR, noise floor, gain, denoise applied). Toggle/tune via `XTTS_PREPROCESS`, `XTTS_PREPROCESS_VAD`, `XTTS_PREPROCESS_HIGHPASS_HZ`/`_ORDER`, `XTTS_PREPROCESS_DENOISE`/`_PROP`/`_SNR_DB`, `XTTS_PREPROCESS_LUFS`, `XTTS_PREPROCESS_MAX_SECONDS`. Conditioning-latent windows are tuned via `XTTS_GPT_COND_LEN` (default 30; XTTS's own default of 6 under-uses multi-clip sets), `XTTS_GPT_COND_CHUNK_LEN`, `XTTS_MAX_REF_LEN`. `POST /analyze-clips` scores candidate clips (SNR, clipping, speech seconds, and speaker-embedding consistency vs the centroid) and returns a recommended subset (threshold `XTTS_ANALYZE_THRESHOLD`) plus the best-first ordering used to fill the GPT conditioning window; the `/ui/voice-clone` page surfaces this as per-clip scores + checkboxes. **TTS is a multi-engine registry** (`config.TTS_ENGINES`, `tts_engine_url()`): every TTS endpoint accepts an `engine` selector (JSON field on synth requests, `?engine=` query param elsewhere) and `GET /api/voice/engines` lists the configured engines + the default; `/ui/voice` and `/ui/voice-clone` render an engine picker. Engines all speak the same sidecar contract — `xtts` (local XTTS-v2, ~2 GB VRAM, RTF ~0.2, multi-clip averaging) and `miso` (remote Miso One on `MISO_TTS_URL`, expressive + one-shot cloning) coexist; `tts` (legacy Qwen on `TTS_URL`) is the rollback. `/health` fans out to every engine. Default chosen by `TTS_DEFAULT_ENGINE`. Optimal clone input per engine is documented in `voice-stack/VOICE_CLONING.md`. Key env: `WHISPER_URL`, `XTTS_TTS_URL`, `MISO_TTS_URL`, `TTS_DEFAULT_ENGINE`, `TTS_URL`.
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

`cloudflared` is on the `cloud` compose profile and only starts with `docker compose --profile cloud up`. For LAN-only dev, omit the profile. Set bind IPs via `RAG_INGEST_BIND_IP` / `RAG_CHAT_BIND_IP`.

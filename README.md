# Ollama + External Qdrant (RAG) behind Cloudflare Zero Trust — with Ingestion + Chat APIs

Run a private AI stack that feels production-grade on day one: chat with your local LLM, ingest knowledge into vector search, and publish only hardened API edges through Cloudflare Zero Trust.

## What this does
- Runs **Ollama** locally and connects to an **external Qdrant** endpoint.
- Provides **`rag-chat`** (FastAPI) as your chat/API gateway, with optional retrieval from Qdrant.
- Provides **`rag-ingest`** (FastAPI) to sign, verify, and safely ingest docs into your vector index.
- Uses **`cloudflared` + Cloudflare Access** so humans use SSO/MFA and machines use service tokens.
- Adds GPU sidecars that sleep when idle, plus the `rag-chat` features that use them:
  - **`modernbert`**: ModernBERT classifier. `/api/pipeline/*` and Pipeline Lab use it.
  - **`infinity`** (`infinity-manager/`): code embeddings and a reranker for the code-ingest profile.
  - **Voice stack** (`voice-stack/`): Whisper STT, XTTS-v2 / Miso / legacy Qwen TTS with voice cloning, and Wyoming bridges for Home Assistant. Used by `/api/voice/*`.
  - **Guard**: SingGuard content-safety checks on the shared Ollama (`/api/guard/*`).
  - **Media generation**: image and video through pluggable providers (OpenAI, Stability, Replicate, Together, ComfyUI).
  - **Repair**: a multi-model repair loop that ai-code-auditor validates (`/api/repair/*`).

| Service | Port | Role |
|---|---|---|
| `rag-ingest` | 9050 | Signed ingestion → chunk → embed → upsert to Qdrant |
| `rag-chat` | 9150 | Chat gateway (Ollama + optional RAG), history, UI, guard/voice/pipeline/repair/generation APIs |
| `ollama` | 11434 | LLM runtime (internal by default) |
| `infinity` | 7997 | Infinity embeddings and reranker, behind an idle-evicting manager (loopback only) |
| `modernbert` | 7998 | ModernBERT classifier sidecar |
| `whisper` | 8030 | faster-whisper STT |
| `xtts` / `tts` | 8031 | XTTS-v2 TTS / legacy Qwen3-TTS |
| `wyoming-xtts` / `wyoming-miso` / `wyoming-whisper` | 10200 / 10201 / 10300 | Home Assistant Wyoming bridges |
| `cloudflared` | n/a | Tunnel; `--profile cloud` only |

Qdrant, Postgres and MinIO run outside this compose file (`QDRANT_URL`, `DATABASE_URL`, `S3_*`). A Miso TTS host is optional and also external (`MISO_TTS_URL`).

## Why this is awesome
- **Secure by default**: your LLM and vector DB stay private; only intentional endpoints are exposed.
- **Fast to ship**: one Compose stack gets you chat, ingestion, retrieval, and auth-ready routing.
- **Practical for real teams**: replay protection, request signing, optional encryption, and easy API integration.
- **Flexible deployment path**: run localhost-only for dev, LAN for trusted setups, or internet-facing behind Zero Trust.

## What this stack does (today)
- **Network posture**:
  - Ollama stays on an internal Docker network only.
  - Qdrant is expected to be reachable via `QDRANT_URL`.
  - APIs bind to `127.0.0.1` on the host, so they’re not reachable from your LAN.
  - `cloudflared` is the only component intended to accept inbound traffic (via Cloudflare’s edge).
- **`rag-chat`**:
  - Proxies `POST /chat` to Ollama `POST /api/chat`.
  - Injects a default system prompt if you didn’t provide one.
  - Optional RAG: embeds the last user message, searches Qdrant, and injects retrieved context as an additional system message.
- **`rag-ingest`**:
  - Requires a shared secret.
  - Verifies an HMAC signature over the request body + timestamp + nonce.
  - Persists nonces (SQLite) and rejects replays within the timestamp window.
  - Optionally requires an encrypted envelope (Fernet) derived from the same shared secret.
  - Has two ingest paths, chosen per document:
    - **Text** (default): char-window chunking → Ollama embeddings → `QDRANT_COLLECTION`.
    - **Code** (`RETRIEVAL_PROFILE=code`, `kind: "code"`, or a source-file `path`): tree-sitter chunks, one per function or class → Infinity `EMBED_MODEL_CODE` → `QDRANT_COLLECTION_CODE` (default `code_audit`), with optional BM25 (`HYBRID_SPARSE=1`). Code docs can carry `path`, `language`, `repo` and `commit_sha`. See [docs/api.md](docs/api.md#rag-ingest-port-9050).

## Design goals
- Keep **Qdrant protected** (private/LAN-only or behind trusted network controls)
- Keep **Ollama entirely unexposed externally**
- Expose only:
  - `rag-ingest` (for indexing; **service-token protected** with Cloudflare Access)
  - `rag-chat` (for humans/agents; **SSO/MFA** or service-token protected with Cloudflare Access)

## Repo layout
```
.
├─ docker-compose.yml            # full stack (Qdrant/Postgres/MinIO are external)
├─ docker-compose.lan-ollama.yml # optional overlay exposing Ollama on the LAN
├─ .env.example
├─ docs/                         # api.md, configuration.md, integration-guide.md
├─ rag-ingest/app/               # main.py (signing, /ingest), ingest_pipeline.py, code_chunker.py, embeddings.py
├─ rag-chat/app/                 # main.py wires routers/
│  ├─ routers/                   # chat, api_keys, ui, generation, media, providers, pipeline, repair, runtime, guard, voice
│  ├─ providers/                 # image/video provider adapters
│  ├─ repair/                    # repair loop (fixer → critic → adversary)
│  ├─ streaming.py               # /chat SSE
│  └─ guard_policy.py, guard_image.py
├─ modernbert/app/               # classifier sidecar (lazy GPU load, keep-alive eviction)
├─ infinity-manager/manager.py   # idle-evicting proxy in front of Infinity
├─ voice-stack/                  # whisper/xtts/qwen-tts servers, wyoming_bridge/, VOICE_CLONING.md
└─ scripts/                      # smoke tests, tool-calling/streaming clients, recommend_model.py
```

## Prereqs
- Docker Engine + Docker Compose v2 (`docker compose`)
- A Cloudflare account with Zero Trust enabled (only needed if you plan to expose the APIs over the internet)
- A host firewall/security group that does **not** expose your Docker ports publicly

## Configuration

### `.env`
Copy the example and fill in values:

```bash
cp .env.example .env
```

Required:
- `CF_TUNNEL_TOKEN`: Cloudflared tunnel token (from Cloudflare Zero Trust)
- `INGEST_SHARED_SECRET`: shared secret used for ingestion HMAC (and for optional envelope encryption)
- `QDRANT_URL`: external Qdrant HTTP endpoint (example: `http://<INFRA_HOST>:6333`)

Optional:
- `INGEST_REQUIRE_ENCRYPTION`: set to `1` to require encrypted envelopes for ingestion
- `CHAT_MODEL`: default Ollama model for chat (example: `llama3.2:3b`)
- `CHAT_SYSTEM_PROMPT`: default system prompt
- `QDRANT_COLLECTION`: collection that `rag-ingest` writes text docs to and `rag-chat` retrieves from for `/chat` and `/search` (default `audit_docs`, which holds ai-code-auditor's wiki sections and agent packs). `project_docs`, the old default, is deprecated. Code docs go to `QDRANT_COLLECTION_CODE`.
- `INGEST_ALLOWED_COLLECTIONS`: comma-separated extra text collections a signed `/ingest` request may target with its optional `collection` field `QDRANT_COLLECTION` is always allowed; any other collection is refused with 400, never redirected. `rag-chat` retrieves only from `QDRANT_COLLECTION`.
- `RAG_ENABLED`: `1` (default) injects Qdrant context into `/chat`; `0` turns retrieval off. Tune with `RAG_TOP_K` and `RAG_MAX_CONTEXT_CHARS`.
- `CHAT_HISTORY_MAX_MSGS`: messages kept per conversation (default `100`; older ones are pruned; `0` keeps everything).
- `DEV_AUTH_BYPASS`: **`1` disables rag-chat authentication** (requests run as `X-Dev-User` or `DEV_DEFAULT_USER`). Local development only. Never enable it on an exposed instance.
- `PROVIDER_ENCRYPTION_KEY`: passphrase that encrypts stored image/video provider API keys. **If unset, provider keys are stored as plain base64.**
- `RETRIEVAL_PROFILE`: `text` (default) or `code`, the default ingest path (see above).
- `OLLAMA_KEEP_ALIVE`: Ollama keep-alive setting (example: `15m`)
- `INFINITY_IDLE_TIMEOUT`: seconds before the managed Infinity child is evicted after idle; next inference request starts it again
- `CHAT_DB_URL` / `DATABASE_URL`: chat database DSN. `postgresql://...` is accepted and automatically normalized to async SQLAlchemy driver usage.
  - On startup, `rag-chat` auto-creates the Postgres database if it does not exist, then runs Alembic migrations.
- `INGEST_NONCE_STORE`: `sqlite` (default) or `redis` for replay-protection nonce claims.
- `INGEST_REDIS_URL`: Redis URL used when `INGEST_NONCE_STORE=redis`.
- `MEDIA_BACKEND`: `local` (default) or `minio` for generated media storage.
- `MEDIA_S3_*`: MinIO/S3 settings used when `MEDIA_BACKEND=minio`.
- Sidecar URLs: `MODERNBERT_URL`, `INFINITY_URL`, `WHISPER_URL`, `XTTS_TTS_URL` / `MISO_TTS_URL` / `TTS_URL`.

Every variable, grouped by service: [docs/configuration.md](docs/configuration.md).

Single-user homelab QA profile (quality-first):
- `CHAT_MODEL=Qwen2.5:7b`
- `EMBED_MODEL=mxbai-embed-large`
- Rationale: stronger code-review/reasoning and better retrieval fidelity are usually worth the extra latency in a one-user QA setup.
- If latency/memory pressure is too high, fall back to lighter defaults (e.g., `llama3.2:3b` + `nomic-embed-text`).

## Quick start

1) Start the stack:

```bash
docker compose up -d --build
docker logs -f cloudflared
```

### External Qdrant migration checklist
- Confirm Qdrant API is reachable from this host/container network: `curl -s http://<QDRANT_HOST>:6333/readyz`
- Point `QDRANT_COLLECTION` at the collection you retrieve from (default `audit_docs`).
- Keep `EMBED_MODEL` unchanged (or same vector dimension), otherwise upserts/search can fail due to collection vector-size mismatch.
- Ensure your external Qdrant has persistent storage configured (snapshot/volume policy handled in `test_dbs`).
- Ensure host firewall rules allow this app host to reach `6333` (and `6334` only if you later use gRPC clients).
- Confirm both `rag-chat` and `rag-ingest` use the same `QDRANT_URL` + `QDRANT_COLLECTION`.

2) Pull a model (once):

```bash
docker exec -it ollama ollama pull llama3.2:3b
```

Optional: pick better defaults for your machine (host-side helper):

```bash
python3 scripts/recommend_model.py
```

3) Local health checks (on the host):

```bash
curl -s http://127.0.0.1:9050/health
curl -s http://127.0.0.1:9150/health
```

### LAN-only / no Cloudflare (dev)
- Defaults stay bound to `127.0.0.1` (safe for single-host dev).
- Ports are now parametric; set bind IP/port via env vars instead of adding extra `ports` entries (avoids double-binding the defaults):

```
services:
  rag-ingest:
    environment:
      - RAG_INGEST_BIND_IP=<LAN_HOST>
      - RAG_INGEST_PORT=9050
  rag-chat:
    environment:
      - RAG_CHAT_BIND_IP=<LAN_HOST>
      - RAG_CHAT_PORT=9150
  cloudflared:
    profiles: ["cloud"]  # starts only when you opt into the profile
```

- Hot reload is the default (`APP_RELOAD=1`) for local/LAN. Set `APP_RELOAD=0` when you run with the `cloud` profile / prod to disable `uvicorn --reload`.

- Bring up the stack without Cloudflare: `docker compose up -d --build` (cloudflared is skipped because it’s on the `cloud` profile).
- If/when you want Cloudflare, opt in: `docker compose --profile cloud up -d --build`.
- Test from another LAN host (replace `<LAN_IP>` with your machine): `curl http://<LAN_IP>:9150/health`
- Keep `INGEST_SHARED_SECRET` strong and prefer `INGEST_REQUIRE_ENCRYPTION=1` if you allow LAN access. Use host firewalls to restrict which LAN clients can reach `9050/9150`.

#### Optional: expose Ollama on LAN for a trusted service
- Default behavior keeps `ollama` internal-only. If an external LAN service must speak the native Ollama API, use the overlay file: `docker compose -f docker-compose.yml -f docker-compose.lan-ollama.yml up -d`
- Set `OLLAMA_BIND_IP=<LAN_IP>` and optionally `OLLAMA_PORT=11434` in `.env` so Ollama only binds to your LAN interface, not all host interfaces.
- Existing internal container access is unchanged: other services in this stack should keep using `http://ollama:11434`.
- Restrict source IPs with the host firewall; Docker Compose does not provide source-IP allowlists for published ports.

## Cloudflare Zero Trust setup

### 1) Create a Tunnel
In Cloudflare Zero Trust:
- Networks → Tunnels → Create
- Choose Docker
- Copy the **Tunnel Token** into `.env` as `CF_TUNNEL_TOKEN`

### 2) Add Public Hostnames (routes) in the Tunnel
In the Tunnel configuration, add ONLY:
- `your-chat-hostname.example.com` → `http://rag-chat:9150`
- `your-ingest-hostname.example.com` → `http://rag-ingest:9050`

Do **not** add hostnames for Qdrant or Ollama.

### 3) Add Cloudflare Access apps / policies
Create Access apps for the two hostnames:

**`rag-chat` (humans)**
- Policy: SSO + MFA (your account(s))
- Optional: device posture checks

**`rag-ingest` (machines)**
- Policy: **Service Token only**
- Create a Service Token per client (laptop/CI/agent)
- Rotate tokens as needed

## Simple UI

### `rag-chat`
<img width="1165" height="1367" alt="image" src="https://github.com/user-attachments/assets/b35185ae-1ce9-4914-84cd-19c70a4f9b4c" />


Basic chat interface to query the agent and view responses.

### `rag-ingest`
<img width="1053" height="955" alt="image" src="https://github.com/user-attachments/assets/a23376ea-a3b9-49ed-9d11-fb832fea87d0" />

Simple form to submit chunks for embedding and upserting to qdrant.

## APIs

- **[docs/api.md](docs/api.md)**: every route, with its auth requirement and key request fields, including the full `/chat` schema, streaming and tool calling.
- **[docs/integration-guide.md](docs/integration-guide.md)**: calling the stack from another host (LAN or Cloudflare Access), with ready-to-copy client snippets.

### `rag-chat`
**Auth:** chat, history, API-key, generation, provider and runtime routes need `Authorization: Bearer <key>` (create one at `/ui/api-keys`), a Cloudflare Access identity, or `DEV_AUTH_BYPASS=1`. Destructive and git actions also need auth: `DELETE /api/voice/clones/{id}`, `PUT /api/pipeline/labels`, and `/api/repair/run` with `apply` or `git_ops.enabled`. The rest of the guard, voice, pipeline and repair APIs and `/metrics` are **unauthenticated**, so keep rag-chat on loopback or a trusted network, or behind Access.

- `GET /health`, `GET /models`, `GET /metrics`
- `POST /chat`: OpenAI-style chat with `model`, `conversation_id`, `tools`, `temperature`, `max_tokens` and `stream` (SSE ending in `[DONE]`). Returns `choices`, `usage` and `conversation_id`, and persists the conversation.
- `/conversations` (list/create/get/delete), `/api-keys` (list/create/revoke)
- `/api/generate/image`, `/api/edit/image`, `/api/generate/video`, `/api/generations`, `/api/media/{id}`, `/api/providers/*`
- `/api/guard/check`, `/api/guard/categories`, `/api/guard/policy`
- `/api/voice/*`: transcribe, synthesise (plain/stream/pcm), analyze, clone, clones, speakers, engines, health
- `/api/pipeline/*`: ModernBERT status/load/evict/unload, labels, classify, plus overlay/brain/prose helpers
- `/api/repair/run` (SSE) and `/api/repair/jobs/{id}`
- `/api/runtime/{ollama,infinity}/{status,wake,evict}`
- UI: `/` and `/ui` show an index page linking every app page and service dashboard. The pages are `/ui/chat` (chat form), `/ui/history`, `/ui/api-keys`, `/ui/generate`, `/ui/pipeline`, `/ui/guard`, `/ui/voice` and `/ui/voice-clone`.

Example curl (local default port 9150):

```bash
# with API key (recommended; set one at /ui/api-keys)
API_KEY="paste-api-key-here"
curl -s http://127.0.0.1:9150/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, can you summarize what this API does?"}
    ]
  }'
```

Example request body:

```json
{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}
```

### `rag-ingest`
- `GET /health`
- `POST /ingest` → verifies signature + replay-protects nonce, optionally decrypts, then chunks → embeds → upserts into Qdrant
- UI: `GET /ui/ingest` for a minimal in-browser form that signs requests client-side (enter your shared secret locally; keep access limited). Works only when `INGEST_REQUIRE_ENCRYPTION=0`; for encrypted envelopes you must call the API with your own client.

Example curl (local default port 9050 — replace headers with a real signature):

```bash
TS=$(date +%s)
NONCE=$(openssl rand -hex 16)
BODY='{"docs":[{"id":"doc-1","text":"hello world","meta":{"source":"demo"}}],"source":"demo","tags":["example"]}'
# Replace SIG with an HMAC computed exactly as described below
SIG="replace-with-hex-hmac"

curl -s http://127.0.0.1:9050/ingest \
  -H "Content-Type: application/json" \
  -H "X-Timestamp: ${TS}" \
  -H "X-Nonce: ${NONCE}" \
  -H "X-Signature: ${SIG}" \
  -d "${BODY}"
```

### Endpoint smoke test script
- From repo root: `python3 scripts/test_endpoints.py`
- Overrides: set `CHAT_URL` and/or `INGEST_URL`. Both are base URLs (defaults `http://127.0.0.1:9150` and `http://127.0.0.1:9050`).
- Auth: `/chat` needs a key. Set `RAG_API_KEY` (or `CHAT_API_KEY`) to a key from `/ui/api-keys`, unless the server runs with `DEV_AUTH_BYPASS=1`.
- Checks `rag-chat` health, a simple `/chat` call, and `rag-ingest` health; exits non-zero on failure.

#### Request signing (required)
`rag-ingest` requires these headers:
- `X-Timestamp`: unix epoch seconds
- `X-Nonce`: random nonce (unique per request)
- `X-Signature`: hex HMAC-SHA256 over: `sha256(body)` plus timestamp + nonce

Signature logic (matches `rag-ingest/app/main.py`):
- `key = sha256(INGEST_SHARED_SECRET)`
- `msg = "{ts}.{nonce}.{sha256(body)}"` (as bytes, with literal dots; `sha256(body)` is hex)
- `signature = hmac_sha256(key, msg).hexdigest()`

#### Python example client (sign + send)
This is the easiest way to generate the exact signature format:

```python
import hashlib, hmac, json, os, secrets, time
import requests

INGEST_SHARED_SECRET = os.environ["INGEST_SHARED_SECRET"]
INGEST_URL = os.environ.get("INGEST_URL", "http://127.0.0.1:9050")  # base URL

def hmac_key(secret: str) -> bytes:
    return hashlib.sha256(secret.encode("utf-8")).digest()

def sign_body(ts: str, nonce: str, body_bytes: bytes) -> str:
    body_hash = hashlib.sha256(body_bytes).hexdigest().encode("ascii")
    msg = ts.encode("ascii") + b"." + nonce.encode("ascii") + b"." + body_hash
    return hmac.new(hmac_key(INGEST_SHARED_SECRET), msg, hashlib.sha256).hexdigest()

payload = {
    "docs": [{"id": "doc-1", "text": "hello world", "meta": {"source": "demo"}}],
    "source": "demo",
    "tags": ["example"],
}

body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
ts = str(int(time.time()))
nonce = secrets.token_hex(16)
sig = sign_body(ts, nonce, body)

r = requests.post(
    f"{INGEST_URL}/ingest",
    data=body,
    headers={
        "Content-Type": "application/json",
        "X-Timestamp": ts,
        "X-Nonce": nonce,
        "X-Signature": sig,
    },
    timeout=30,
)
print(r.status_code, r.text)
```

#### Optional encrypted envelope (Fernet)
If you set `INGEST_REQUIRE_ENCRYPTION=1`, the request body must be:

```json
{"token":"..."}
```

Where `token` is a Fernet token produced using a key derived from `INGEST_SHARED_SECRET`:
- `fernet_key = base64.urlsafe_b64encode(sha256(INGEST_SHARED_SECRET))`

Python example for envelope encryption:

```python
import base64, hashlib, json, os
from cryptography.fernet import Fernet

secret = os.environ["INGEST_SHARED_SECRET"]
fernet_key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
f = Fernet(fernet_key)

inner = {"docs": [{"id": "doc-1", "text": "hello"}]}
token = f.encrypt(json.dumps(inner).encode("utf-8")).decode("utf-8")
outer = {"token": token}
print(json.dumps(outer))
```

## Model recommendation script
If you’re not sure which Ollama model to run on your host, use:

```bash
python3 scripts/recommend_model.py
```

What it does:
- **CPU/RAM**: reads CPU cores and total RAM (Linux: `/proc/meminfo`)
- **GPU (optional)**: if `nvidia-smi` exists, reads NVIDIA GPU name + VRAM
- Prints suggested defaults:
  - `CHAT_MODEL=...`
  - `EMBED_MODEL=...`
  - plus matching `ollama pull ...` commands

How to apply:
- Add/override in your `.env`:
  - `CHAT_MODEL=...`
  - `EMBED_MODEL=...`
- Pull the models (from the host, or via the container):

```bash
docker exec -it ollama ollama pull <CHAT_MODEL>
docker exec -it ollama ollama pull <EMBED_MODEL>
```

## Extending for “exact answers” (SQL)
For questions like totals/averages/counts, do not rely on embeddings. Instead:
- use SQL to compute exact values (SQLite/Postgres/etc.)
- use the LLM only to explain/format results

This avoids “LLM made up a number” failure modes.

## Obvious missing pieces (recommended next upgrades)
- **Rate limiting / abuse control**: add per-client rate limits (Cloudflare + app-level).
- **Better chunking + parsing**: handle PDFs/HTML/markdown, sentence-aware chunking, dedupe, and content-type specific extractors.
- **Metadata filters / multi-tenant**: per-tenant collections or payload filters; enforce tenant separation server-side.
- **Backups**: document how to back up/restore `ollama`, `rag_ingest_data`, and your external Qdrant storage/snapshots.
- **Secret management**: consider Docker secrets / an external secret manager instead of `.env` on disk.

## Blindspots / gotchas
- Binding ports to `0.0.0.0` on the host exposes services to LAN and can bypass Access. This compose file binds host ports to `127.0.0.1` intentionally.
- Never embed secrets / tokens / keys.
- Treat retrieved content as untrusted reference text (prompt injection is real).

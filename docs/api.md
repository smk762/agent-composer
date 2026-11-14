# API reference

Routes served by each service in this stack, with their auth requirements and key request fields. For connection setup and client snippets see [integration-guide.md](integration-guide.md); for environment variables see [configuration.md](configuration.md).

## Authentication

### rag-chat (`require_user`)

Routes marked **auth** below resolve a user in this order (`rag-chat/app/auth.py`):

1. `Authorization: Bearer <key>`: an API key created at `/ui/api-keys` or via `POST /api-keys`. An unknown or revoked key returns `401 Invalid API key`.
2. `Cf-Access-Authenticated-User-Email`: the header Cloudflare Access sets on authenticated requests. The email becomes the user id.
3. `DEV_AUTH_BYPASS=1`: no credentials needed. The user id is the `X-Dev-User` header, else `DEV_DEFAULT_USER` (default `dev@example.com`). **Use this for local development only.** With it on, anyone who can reach the port is authenticated.

Otherwise the route returns `401 Authentication required`. `GET /health` reports which mode is active (`"auth": "dev-bypass"` or `"access/api-key"`).

> **Open routes.** Routes marked **open** have no auth dependency: the guard API, read-only and generative voice, pipeline and repair calls, `/models`, `/metrics` and the HTML UI pages. Anyone who can reach the port can call them, so keep rag-chat on loopback or a trusted LAN, or put it behind Cloudflare Access. Destructive and git actions always need auth: deleting a voice clone, replacing the pipeline label set, and repair runs with `apply` or `git_ops.enabled`.

### rag-ingest

`POST /ingest` requires HMAC request signing (`X-Timestamp`, `X-Nonce`, `X-Signature`) with replay protection. Envelope encryption is optional. See [Request signing](../README.md#request-signing-required) in the README.

---

## rag-chat (port 9150)

### Chat and history (`routers/chat.py`)

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/health` | open | `{ok, model, auth}` |
| GET | `/models` | open | Ollama chat models (embedding models filtered out); names are tag-qualified, e.g. `llama3.2:3b` |
| GET | `/conversations` | auth | The caller's conversations |
| POST | `/conversations` | auth | Create an empty conversation |
| GET | `/conversations/{id}` | auth | Conversation plus its messages |
| DELETE | `/conversations/{id}` | auth | |
| POST | `/chat` | auth | OpenAI-style chat; see below |
| POST | `/search` | auth | Retrieval over `QDRANT_COLLECTION` as JSON (`routers/search.py`); see below |

#### `POST /chat`

Request (`ChatRequest` in `app/models/schemas.py`):

| Field | Type | Notes |
|---|---|---|
| `messages` | list | Required. `{role: system\|user\|assistant\|tool, content, tool_calls?, tool_call_id?}` |
| `model` | string | Defaults to `CHAT_MODEL`. `foo` and `foo:latest` are both accepted and resolved against Ollama. |
| `conversation_id` | string | Continue a stored conversation. If omitted, a new one is created and its id is returned. |
| `tools` | list | OpenAI-format function tools: `{type: "function", function: {name, description, parameters}}` |
| `temperature` | float | Forwarded to Ollama `options.temperature` |
| `max_tokens` | int | Forwarded to Ollama `options.num_predict` |
| `stream` | bool | Default `false`. When `true`, the response is Server-Sent Events. |

Processing:
- If the first message is not a `system` message, `CHAT_SYSTEM_PROMPT` is prepended.
- If `RAG_ENABLED=1` (the default), the last user message is embedded with `EMBED_MODEL`, the top `RAG_TOP_K` hits are retrieved from `QDRANT_COLLECTION` and injected as a second system message, capped at `RAG_MAX_CONTEXT_CHARS`. Each passage is headed by its title and link (`meta.title`, `meta.url`), and the model is asked to name and link the passages it uses. A failed Qdrant search is logged and skipped. A failed embedding returns 502.
- Messages are persisted to Postgres. Each conversation keeps the newest `CHAT_HISTORY_MAX_MSGS` (default 100; `0` keeps all).

Non-streaming response:

```json
{
  "id": "chatcmpl-…",
  "created": 1731500000,
  "model": "llama3.2:3b",
  "choices": [
    {"message": {"role": "assistant", "content": "…", "tool_calls": null}, "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46},
  "conversation_id": "…"
}
```

When the model calls a tool, `message.tool_calls` holds `[{id, type: "function", function: {name, arguments}}]` (`arguments` is a JSON string). Run the tool, then send its result back as `{"role": "tool", "tool_call_id": "<id>", "content": "<result>"}` with the same `conversation_id`.

Streaming response (`stream: true`), with each event on one `data:` line:

```
data: {"id":"chatcmpl-…","created":…,"model":"…","choices":[{"index":0,"delta":{"role":"assistant"}}],"conversation_id":"…"}
data: {"id":"chatcmpl-…","created":…,"model":"…","choices":[{"index":0,"delta":{"content":"Hel"}}]}
data: {"id":"…","created":…,"model":"…","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

Tool calls arrive as `delta.tool_calls`. Errors arrive as `data: {"error": "…"}`, followed by `data: [DONE]`.

```bash
curl -N http://127.0.0.1:9150/chat \
  -H "Authorization: Bearer ${CHAT_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Count to 5"}],"stream":true}'
```

Working clients: `scripts/test_streaming.py`, `scripts/test_tool_calling.py` and `scripts/example_tool_calling_client.py`.

#### `POST /search`

The passages `/chat` would inject, as JSON, for agents and tools that want to read them directly. Same auth as `/chat`.

| Field | Type | Notes |
|---|---|---|
| `query` | string | Required, 1–2000 characters. Embedded with `EMBED_MODEL`. |
| `top_k` | int | Default 5, at most 20. |
| `repo` | string | Shorthand for the tag `repo:<name>`. |
| `kind` | string | Shorthand for the tag `kind:<kind>`: ai-code-auditor sends `kind:wiki` (one doc per wiki `##` section), `kind:agent_coding_pack`, `kind:mithrandir_latest` and its report kinds. |
| `tags` | list | Further tags; a hit must carry every requested tag. |

Response: `{collection, tags, hits: [{score, doc_id, chunk_index, title, url, path, source, tags, text}]}`, best first. A Qdrant failure is a 502, as is a failed embedding.

```bash
curl -s http://127.0.0.1:9150/search -H "Authorization: Bearer ${CHAT_API_KEY}" -H "Content-Type: application/json" \
  -d '{"query":"how are uploads validated","repo":"argus-mint","kind":"wiki","top_k":3}'
```

### API keys (`routers/api_keys.py`)

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/api-keys` | auth | The caller's keys (prefix and timestamps only) |
| POST | `/api-keys` | auth | `{name?}` → returns the secret **once** |
| DELETE | `/api-keys/{key_id}` | auth | Revoke |

You need some credential to create the first key: Cloudflare Access in front of `/ui/api-keys`, or `DEV_AUTH_BYPASS=1` on a trusted host.

### Media generation (`routers/generation.py`, `routers/media.py`, `routers/providers.py`)

Image and video generation go through pluggable providers (`app/providers/`: `openai`, `stability`, `replicate`, `together`, `comfyui`). Each user configures their own provider API keys. Keys are encrypted at rest only when `PROVIDER_ENCRYPTION_KEY` is set (see [configuration.md](configuration.md#security-relevant)).

| Method | Path | Auth | Key fields |
|---|---|---|---|
| POST | `/api/generate/image` | auth | JSON `{prompt, negative_prompt?, provider?, model?, size="1024x1024", n=1, style?}` |
| POST | `/api/edit/image` | auth | multipart `prompt`, `image`, `mask?`, `provider?`, `model?`, `size?` |
| POST | `/api/generate/video` | auth | multipart `prompt`, `image?`, `provider?`, `model?`, `duration?`, `aspect_ratio?` |
| GET | `/api/generations` | auth | The caller's generations |
| GET | `/api/generations/{id}` | auth | Status plus media list |
| GET | `/api/media/{asset_id}` | auth | Asset bytes (local disk or MinIO, per `MEDIA_BACKEND`) |
| GET | `/api/providers` | auth | Registered providers and whether each is configured |
| GET | `/api/providers/{name}` | auth | Stored config: `has_api_key`, `settings`, `enabled` (the key itself is never returned) |
| POST | `/api/providers/{name}/configure` | auth | Store the API key and settings |
| POST | `/api/providers/{name}/test` | auth | Connectivity check |
| DELETE | `/api/providers/{name}` | auth | Remove the config |

### Guard (`routers/guard.py`)

SingGuard content-safety checks on the shared Ollama.

| Method | Path | Auth | Notes |
|---|---|---|---|
| POST | `/api/guard/check` | open | See below |
| GET | `/api/guard/categories` | open | Default rule taxonomy |
| GET | `/api/guard/policy` | open | Default policy text and supported `thinking` modes |

`POST /api/guard/check` fields: `input` (the text, or the prompt that goes with an image), `image_base64` (raw base64 or a data URL; optional), `conversation` (`[{role: user|assistant, content}]` turns before `input`), `input_role` (`user` or `assistant`), `policy` (natural-language rules; omit it to use the default taxonomy), `thinking` (`fast`, `fast-slow`, `slow` or `cascade`; default `GUARD_THINKING`), `model` (default `GUARD_MODEL`). The response contains `safe`, `verdict`, `rule`, `rule_text`, `explanation`, `categories`, `raw`, `model`, `thinking` (the pass whose verdict stood) and `elapsed_ms`.

### Voice (`routers/voice.py`)

Proxies to the Whisper STT sidecar and a registry of TTS engines (`xtts`, `miso`, legacy `tts`). Synth requests take an `engine` field; the other routes take an `?engine=` query parameter. If you leave it out, the request uses `TTS_DEFAULT_ENGINE` and, on synth, falls back to `TTS_FALLBACK_ENGINE`. For clone input guidance, see `voice-stack/VOICE_CLONING.md`.

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/api/voice/engines` | open | Configured engines and the default |
| GET | `/api/voice/health` | open | Fans out to Whisper and every engine |
| POST | `/api/voice/transcribe` | open | multipart `audio`, `?language=` → text |
| POST | `/api/voice/synthesise` | open | JSON `{text, engine?, voice_clone_id?, speaker?, emotion="neutral", language="en", speed?, pause_ms?}` → audio URL |
| POST | `/api/voice/synthesise/stream` | open | Same fields → SSE, one WAV per sentence |
| POST | `/api/voice/synthesise/pcm` | open | Raw 24 kHz float32 PCM stream (XTTS only) |
| GET | `/api/voice/speakers` | open | Built-in speakers |
| POST | `/api/voice/analyze` | open | multipart `audio` (one or more clips) → per-clip scores and a recommended subset |
| POST | `/api/voice/clone` | open | multipart `audio` (one or more), `voice_name`, `companion_id?`, `reference_text?` |
| GET | `/api/voice/clones` | open | Stored clones |
| POST | `/api/voice/clones/import` | open | multipart `archive`, `voice_name?`, `companion_id?` |
| GET | `/api/voice/clones/{companion_id}/download` | open | Clone archive |
| DELETE | `/api/voice/clones/{companion_id}` | auth | Permanent. The `/ui/voice-clone` page sends the key saved at `/ui/api-keys`. |

### Pipeline (`routers/pipeline.py`)

Reverse proxy to the ModernBERT sidecar, plus LLM helpers for character-dialogue pipelines.

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/api/pipeline/status` | open | Sidecar `model_state` |
| POST | `/api/pipeline/load` \| `/evict` \| `/unload` | open | Move the model between `gpu`, `cpu` and `unloaded` |
| GET | `/api/pipeline/labels` | open | Current zero-shot label set |
| PUT | `/api/pipeline/labels` | auth | Replace the label set: `{labels: [...]}` |
| POST | `/api/pipeline/classify` | open | `{text, context?, top_k?, threshold?}` |
| POST | `/api/pipeline/overlay` | open | `{character_name, character_persona, character_state?, labels, scores, turns?, model?}` → a style directive |
| POST | `/api/pipeline/brain` | open | `{character_name, character_persona, style_directive, labels, turns?, message, model?}` → a reply plan |
| POST | `/api/pipeline/prose` | open | `{character_name, style_directive, skeleton, turns?, message, model?}` → the final reply |

### Repair (`routers/repair.py`)

Runs a multi-model agentic repair loop (fixer → critic → adversary), validated between iterations by the ai-code-auditor audit API.

| Method | Path | Auth | Notes |
|---|---|---|---|
| POST | `/api/repair/run` | open / auth | SSE stream; events: `started`, `pack_loaded`, `iteration`, `warning`, `error`, `complete`. **Auth is required when `apply: true` or `git_ops.enabled`**, and the call fails with 401 before any work starts. Review, suggest and temp-copy `auto_fix` runs stay open. |
| GET | `/api/repair/jobs/{job_id}` | open | Final job state |

Key fields (`app/repair/models.py`): `repo`, `diff` or `compare_branch`, `pack_id` (an ai-code-auditor pack; `repo` may then be omitted), `audit_api_base`, `errors`, `mode` (`auto_fix`, `review` or `suggest`), `apply`, `git_ops`, `fix_model`, `critique_model`, `adversarial_model`, `max_iterations` (1–10, default 3), `validate_tests`.

> **SSRF:** `audit_api_base` is supplied by the client, and the server issues GET requests to it, plus a status-callback POST. Only `http(s)` URLs are accepted, but no host allowlist exists yet, and read-only runs are unauthenticated. Expose this route only to trusted callers. `ai-audit repair` sends its token as a Bearer key, so set it to a rag-chat API key for `apply` runs.

### Runtime controls (`routers/runtime.py`)

All routes require **auth**.

| Method | Path |
|---|---|
| GET | `/api/runtime/ollama/status` |
| POST | `/api/runtime/ollama/wake` \| `/api/runtime/ollama/evict` |
| GET | `/api/runtime/infinity/status` |
| POST | `/api/runtime/infinity/wake` \| `/api/runtime/infinity/evict` |

### Metrics

`GET /metrics` (open) returns Prometheus request counters and latency histograms (`app/metrics.py`).

### UI pages (`routers/ui.py`)

These are server-rendered HTML pages. The pages themselves are open, and the APIs they call apply their own auth.

| Path | Page |
|---|---|
| `/`, `/ui` | Index: links to every app page and service dashboard |
| `/ui/chat` | Chat form |
| `/ui/history` | Browse and open conversations |
| `/ui/api-keys` | Create, list and revoke API keys |
| `/ui/generate` | Image and video generation, with a "Check image" guard button |
| `/ui/pipeline` | Pipeline Lab (ModernBERT classify, labels, model state) |
| `/ui/guard` | Guard tester |
| `/ui/voice` | TTS/STT playground with an engine picker |
| `/ui/voice-clone` | Clone creation with per-clip analysis |

---

## rag-ingest (port 9050)

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/health` | open | |
| GET | `/ui/ingest` | open | Browser form that signs requests client-side. It only works when `INGEST_REQUIRE_ENCRYPTION=0`. |
| POST | `/ingest` | HMAC | `{docs: [...], source?, tags?}` |

Each doc accepts:
- **Text:** one of `text`, `content`, `page_content` or `body`, plus optional `id`, `doc_id`, `title`, `url` and `source`, which are stored as point payload. A doc with none of the text fields is indexed as its JSON.
- **Code:** `kind` (`code` or `text`, forces the path), `path` (repo-relative; drives language detection), `language` (tree-sitter id), `repo`, `commit_sha`.

Routing works per doc. An explicit `kind` wins. Otherwise a `path` with a known source extension selects the code path. Otherwise `RETRIEVAL_PROFILE` decides (`text` by default). The two paths write to separate collections:

| | Text path | Code path |
|---|---|---|
| Chunking | Sliding char window (`INGEST_CHUNK_SIZE` / `INGEST_CHUNK_OVERLAP`) | tree-sitter, per function or class (`CODE_CHUNK_MIN_CHARS` / `CODE_CHUNK_MAX_CHARS`) |
| Embeddings | Ollama `EMBED_MODEL` | Infinity `EMBED_MODEL_CODE` (default `nomic-ai/CodeRankEmbed`) |
| Collection | `QDRANT_COLLECTION` (unnamed dense vector) | `QDRANT_COLLECTION_CODE` (default `code_audit`; named `dense` vector, plus `bm25` when `HYBRID_SPARSE=1`) |

rag-chat's `/chat` RAG context and `/search` read only `QDRANT_COLLECTION`. The code collection is consumed by ai-code-auditor.

The response contains `ok`, `chunks`, `upserted`, `skipped_chunks`, `truncated_chunks`, `collection`, `embed_model`, `retrieval_profile`, `hybrid_sparse`, `code_docs` and `text_docs`.

---

## Sidecars

These are internal to the compose network unless noted. rag-chat proxies the ones a client normally needs.

| Service | Port | Routes |
|---|---|---|
| `modernbert` | 7998 | `GET /health` (always 200; check `model_state`), `GET /model/status`, `POST /model/load\|evict\|unload`, `GET\|PUT /labels`, `POST /classify`, `POST /embed` |
| `infinity` (infinity-manager) | 7997 (loopback) | Proxies the Infinity embeddings/rerank API to a child process that starts on demand and is evicted after `INFINITY_IDLE_TIMEOUT`; `GET /health`, `GET /admin/status`, `POST /admin/wake\|evict` |
| `whisper` | 8030 | `POST /transcribe` |
| `xtts` / `miso` / `tts` | 8031–8034 | Shared TTS sidecar contract: `/synthesise`, `/synthesise/stream`, `/clone-voice`, `/clones`, `/speakers`, `/analyze-clips`, `/health` |
| `wyoming-xtts` / `wyoming-miso` / `wyoming-whisper` | 10200 / 10201 / 10300 | Wyoming protocol bridges for Home Assistant |

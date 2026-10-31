# `rag-ingest` Standalone Repo Specification

## Goal
Create a standalone repo to deploy `rag-ingest` as an independent service on `192.168.1.198`, with external dependencies:
- `QDRANT_URL=http://192.168.1.198:6333`
- `OLLAMA_URL=http://192.168.1.198:11434`
- Optional Redis nonce store on `192.168.1.121:6380`
- Required Ollama embed model pre-pulled on `192.168.1.198` (QA profile: `mxbai-embed-large`)

## Non-goals
- No `rag-chat` service in this repo.
- No bundled Qdrant/Ollama containers.
- No Cloudflare config automation (just env + docs hooks).

---

## Functional Requirements

- Expose:
  - `GET /health`
  - `POST /ingest`
  - `GET /ui/ingest` (works only when `INGEST_REQUIRE_ENCRYPTION=0`)
- Request auth:
  - Required headers: `X-Timestamp`, `X-Nonce`, `X-Signature`
  - HMAC verification logic compatible with current implementation
- Replay protection:
  - `INGEST_NONCE_STORE=sqlite|redis`
  - Redis mode uses TTL + atomic claim (`SET NX EX`)
- Ingestion pipeline:
  - Chunk -> embed via Ollama -> upsert into Qdrant collection
- Config-driven behavior:
  - Strict `.env` support with safe defaults and fail-fast for required secrets

---

## Required Deliverables

1. **Repo structure**
   - `docker-compose.yml` (single service: `rag-ingest`)
   - `rag-ingest/` app code + `Dockerfile`
   - `env.example`
   - `README.md`
   - `scripts/smoke_test.sh` (or python equivalent)

2. **Startup behavior**
   - `init.sh` pattern (same style as current repos)
   - `APP_RELOAD` toggle for dev/prod behavior

3. **Docs**
   - API contract
   - Signature generation examples
  - Deployment guide for the rag-ingest host (192.168.1.198)
   - Troubleshooting section (Qdrant/Ollama unreachable, signature mismatch, replay detection)

4. **Verification**
   - `docker compose config --services` passes
   - smoke tests pass against real endpoints
   - clear "first run" checklist

---

## Environment Contract

Must support at minimum:

- `INGEST_SHARED_SECRET` (required)
- `INGEST_REQUIRE_ENCRYPTION` (`0|1`)
- `INGEST_MAX_SKEW_S` (default `300`)
- `INGEST_NONCE_STORE` (`sqlite|redis`, default `sqlite`)
- `INGEST_NONCE_DB` (sqlite path, default `/data/nonces.db`)
- `INGEST_REDIS_URL` (required when nonce store is redis)
- `INGEST_NONCE_KEY_PREFIX` (default `ingest:nonce:`)
- `QDRANT_URL` (required; set to `http://192.168.1.198:6333`)
- `QDRANT_COLLECTION` (default `project_docs`)
- `OLLAMA_URL` (required; set to `http://192.168.1.198:11434`)
- `EMBED_MODEL` (QA profile: `mxbai-embed-large`; must be pulled on Ollama host)
- `INGEST_CHUNK_SIZE`, `INGEST_CHUNK_OVERLAP`, `INGEST_UPSERT_BATCH_SIZE`
- `APP_PORT` (default `9050`)
- `APP_RELOAD` (`0|1`)

---

## Task List for Agent

1. **Bootstrap repo**
   - Create minimal FastAPI app with `/health`, `/ingest`, `/ui/ingest`.
   - Add requirements and Dockerfile.

2. **Implement security**
   - Add HMAC signature verification.
   - Add timestamp skew validation.
   - Add nonce replay guard with pluggable sqlite/redis backend.

3. **Implement ingestion pipeline**
   - Chunking logic.
   - Ollama embedding client with retry/truncation handling.
   - Qdrant create-if-missing collection and upsert flow.

4. **Containerization**
   - Add `init.sh`.
   - Compose with only `rag-ingest` service and persistent volume for sqlite nonce DB.
   - No bundled infra services.

5. **Configuration + docs**
   - `env.example` with comments and production notes.
   - README with curl/python examples and signature reference.

6. **Validation tooling**
   - Add smoke test script:
     - health check
     - signed ingest test
   - Add basic CI or local check commands in README.

7. **Hardening pass**
   - Fail-fast when required env vars missing.
   - Add sensible timeouts.
   - Ensure logs do not leak secret material.
   - Add healthcheck in compose.

8. **Acceptance verification**
   - Run against:
     - Qdrant at `192.168.1.198:6333`
     - Ollama at `192.168.1.198:11434`
     - Ensure Ollama has the embedding model: `ollama pull mxbai-embed-large`
   - Confirm successful ingest/upsert and replay rejection.

### QA profile rationale
- For single-user homelab QA, prefer quality over raw throughput.
- Recommended pairing:
  - `CHAT_MODEL=Qwen2.5:7b`
  - `EMBED_MODEL=mxbai-embed-large`
- This gives stronger reasoning/review quality and better retrieval fidelity; switch to lighter models only if latency or memory pressure becomes a blocker.

---

## Definition of Done

- `docker compose up -d --build` starts service cleanly.
- Valid signed request ingests to Qdrant.
- Duplicate nonce is rejected.
- Works in both nonce modes (`sqlite`, `redis`).
- README is complete enough for a fresh operator to deploy in under 15 minutes.

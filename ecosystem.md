## Repos

| Label | Repo | Role |
|-------|------|------|
| Core API | [kimini](https://github.com/smk762/kimini-api) | FastAPI platform backend — auth, chat, companions, economy, marketplace, generation proxies, WebSocket |
| Familiar UI | [somnus](https://github.com/smk762/somnus) | Frontend/operator app for familiar workflows, HITL review/rating, and training/evaluation operations |
| Image | [imogen](https://github.com/smk762/imogen) | Self-hosted GPU image generation (FLUX.1-dev + SDXL) behind async gateway |
| Post-process | [rubedo](https://github.com/smk762/rubedo) | Image post-processing — HTTP callbacks from imogen after generation jobs complete (`imogen` → `rubedo` in `config/architecture_rules.yaml`) |
| Video | [vidita](https://github.com/smk762/vidita) | Self-hosted GPU video generation (Wan 2.2) behind async gateway |
| Voice | [tss-stack](https://github.com/smk762/tss-stack) | TTS (XTTS) + STT (Whisper) voice gateway with async job API, presigned audio URLs |
| Ollama Chat | [agent-composer](https://github.com/smk762/agent-composer) | Ollama-backed chat service (RAG chat endpoint) behind Cloudflare Zero Trust |
| RAG Ingest | [mimiri](https://github.com/smk762/mimiri) | Standalone `rag-ingest` service with signed ingestion API (`/ingest`) for Qdrant upserts via external Ollama embeddings |
| Orchestrator | [gothmog](https://github.com/smk762/gothmog) | LangGraph/LangChain workflow orchestrator — multi-step pipelines across stacks |
| ComfyUI | [voluptas](https://github.com/smk762/voluptas) | ComfyUI inference node — docker-compose wrapper with GPU passthrough, workflow API at :8188 |
| Infra | [test_dbs](https://github.com/smk762/test_dbs) | PostgreSQL, Redis, MinIO, Qdrant — shared data layer |
| Observability | [sauron](https://github.com/smk762/sauron) | Prometheus + Grafana + Loki + Promtail + Alertmanager for cross-stack metrics, logs, and alerting |

---

## Ecosystem Flow Diagram

```
                              ┌─────────────────────────────────────────────┐
                              │             CLIENTS / FRONTENDS             │
                              │  (somnus UI, web app, mobile, SDK, admin)   │
                              └────────────────────┬────────────────────────┘
                                                   │
                                        HTTP / WebSocket
                                                   │
                         ┌─────────────────────────▼──────────────────────────┐
                         │                                                    │
                         │            KIMINI  —  Core API  :8000              │
                         │         (fastapi-prod-skeleton)                    │
                         │                                                    │
                         │  ┌──────────┬──────────┬──────────┬─────────────┐  │
                         │  │ Auth     │ Chat     │ Economy  │ Marketplace │  │
                         │  │ Users    │ Convos   │ Gems     │ Creators    │  │
                         │  │ Personas │ Stories  │ Subs     │ Discover    │  │
                         │  ├──────────┴──────────┴──────────┴─────────────┤  │
                         │  │           Generation Domains                 │  │
                         │  │  imagegen │ videogen │ voicegen              │  │
                         │  ├──────────────────────────────────────────────┤  │
                         │  │  Orchestrate proxy  │  WebSocket (realtime)  │  │
                         │  ├──────────────────────────────────────────────┤  │
                         │  │  TaskIQ workers (high / default / low)       │  │
                         │  └──────────────────────────────────────────────┘  │
                         │                                                    │
                         └──┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬───────┘
                            │      │      │      │      │      │      │      │
            ┌───────────────┘      │      │      │      │      │      │      └──────────────────┐
            │                ┌─────┘      │      │      │      └─────┐│                         │
            ▼                ▼            ▼      ▼      ▼            ▼▼                         ▼
  ┌─────────────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  │  GOTHMOG :8030  │ │ IMOGEN   │ │   RUBEDO     │ │  VIDITA      │ │ TSS-STACK      │ │ AGENT-COMPOSER   │ │ VOLUPTAS :8188   │
  │  Orchestrator   │ │ :8003    │ │ post-process │ │  :8000       │ │ :9001 gateway  │ │ Ollama chat      │
  │                 │ │ img-gw   │ │ (callbacks   │ │  vid-gw      │ │                │ │                  │
  │ LangGraph       │ │          │ │  from imogen)│ │              │ │ TTS (XTTS)     │ │ rag-chat  :9150  │
  │ workflows:      │ │ POST     │ │              │ │ POST         │ │ STT (Whisper)  │ │ mimiri    :9050* │
  │ • img_generate  │ │ /images/ │ │ HTTP in      │ │ /videos/     │ │                │ │                  │
  │ • img_to_video  │ │ generate │ │ from imogen  │ │ generate     │ │ /voices        │ │ Ollama (remote)  │
  │ • char_create   │ │          │ │ after jobs   │ │              │ │ /tts/jobs      │ │ :11434 @ .138    │
  │ • style_xfer    │ │ GET      │ │              │ │ GET          │ │ /stt/jobs      │ │                  │
  │ • batch_gen     │ │ /images/ │ │              │ │ /videos/     │ │ /tts/jobs/{id} │ │ Cloudflare       │
  │                 │ │ jobs/{id}│ │              │ │ jobs/{id}    │ │ /stt/jobs/{id} │ │ Zero Trust       │
  │ LLM calls:      │ │          │ │              │ │              │ │                │ └──────────────────┘
  │ prompt expand,  │ │ GPU:     │ │              │ │ GPU:         │ │ Services:      │
  │ style analysis  │ │ flux     │ │              │ │ wan22 :8010  │ │ xtts (engine)  │ ┌─────────────────┐
  │                 │ │ :8001    │ │              │ │ (internal)   │ │ tts-worker     │ │ LORALINE :8010  │
  │ Tools call ─────┤►│ sdxl     │ │◄─ HTTP from  │ │              │ │ whisper-worker │ │ LoRA + Proxy    │
  │ img/vid/voice   │ │ :8002    │ │   imogen jobs│ │              │ │ redis, minio   │ │                 │
  └────────┬────────┘ └──────────┘ └──────────────┘ └──────────────┘ └───────┬────────┘ │ LoRA training   │
           │               ▲             ▲             ▲              ▲   │          │ Image proxy ──►─┤─► imogen
           │               │             │             │              │   │          │ Video proxy ──►─┤─► vidita
           │               │             │             │              │   │          │ Chat  (Ollama)  │
           └───────────────┴─────────────┴─────────────┴──────────────┘   │          │ Voice (tss) ──►─┤─► tss-stack
             gothmog also calls img/vid/voice stacks (not rubedo)        │          │                 │
             directly via LangChain tools                                │          │ POST/GET        │
                                                            │            │ /v1/lora/jobs   │
           kimini voicegen calls tss-stack ─────────────────┘            │ /v1/image/jobs  │
                                                                       │ /v1/video/jobs  │
                                                                       │ /v1/chat/jobs   │
                                                                       └─────────────────┘

  ────────────────────────────────────────────────────────────────────────────
                         SHARED DATA LAYER  (test_dbs)
  ────────────────────────────────────────────────────────────────────────────

  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  ┌──────────────────┐
  │  PostgreSQL :5432│  │  Redis :6379     │  │  MinIO (S3) :9000/:9001  │    │  Qdrant :6333    │
  │                  │  │                  │  │                          │  │                  │
  │  kimini tables   │  │  rate limits     │  │  uploads bucket          │  │  project_docs    │
  │  gothmog.runs    │  │  TaskIQ broker   │  │  generated/images/       │  │  RAG vectors     │
  │  loraline jobs   │  │  WS pub/sub      │  │  generated/videos/       │  │  mimiri upserts  │
  │                  │  │  imgq:*, vidq:*  │  │  generated/audio/        │  │  rag-chat reads  │
  │                  │  │  orcq:*, orcr:*  │  │  lora artifacts          │  │                  │
  │                  │  │  tss job queues  │  │  comfyui/workflows/      │  │                  │
  │                  │  │                  │  │  tts/stt audio artifacts │  │                  │
  └──────────────────┘  └──────────────────┘  └──────────────────────────┘  └──────────────────┘

  ────────────────────────────────────────────────────────────────────────────
                           OBSERVABILITY  (sauron)
  ────────────────────────────────────────────────────────────────────────────

  ┌──────────────────────────────────────────────────────────────────────────┐
  │ SAURON                                                                   │
  │ prometheus:9090  grafana:3001  loki:3100  alertmanager:9093              │
  │ promtail ships Docker logs -> Loki                                       │
  │ Prometheus scrapes primary host services + remote GPU exporters          │
  │ (stack remains healthy if one GPU host is offline)                       │
  └──────────────────────────────────────────────────────────────────────────┘
```

RAG ingestion is served by `mimiri` (`rag-ingest`) on `192.168.1.128:9050`, colocated with Qdrant on `192.168.1.128:6333`, and using Ollama on `192.168.1.138:11434` (`/ui/ingest` only when `INGEST_REQUIRE_ENCRYPTION=0`).

---

## Deployment Topology

### Recommended Host Allocation (homelab test/QA, no failover)

| Host | Profile | Recommended services | Why this fit is optimal |
|------|---------|----------------------|--------------------------|
| `192.168.1.109` | Linux desktop, RTX 3090 24 GB, 64 GB RAM, Ryzen 5 7600 | `imogen`, `vidita`, `loraline`, `rubedo` | Best single-job GPU capacity. Keep the heaviest CUDA workloads, LoRA training, and imogen-adjacent post-processing (`rubedo`) on the strongest GPU host for predictable latency in test runs. |
| `192.168.1.138` | VM, RTX 4060 8 GB (passthrough), 16 GB RAM, 8 vCPU | `tss-stack`, `agent-composer` (Ollama runtime/chat + Qdrant + rag-ingest) | Good fit for moderate GPU services (Whisper/XTTS + small/medium Ollama). Keeps conversational workloads off the control-plane host. Qdrant and rag-ingest colocated here with agent-composer. |
| `192.168.1.121` | DS923+ NAS | PostgreSQL (`:5433`), Redis (`:6380`), MinIO (`:9000`) | Stateful data layer — shared across stacks. Postgres and Redis migrated here from `.128`; MinIO was already here. |
| `192.168.1.128` | VM, RTX 4060 8 GB (passthrough), 64 GB RAM, 4 vCPU | `kimini`, `somnus`, `gothmog`, `sauron` | Central control/observability node. PostgreSQL/Redis/MinIO on NAS (`.121`); Qdrant/rag-ingest moved to `.138` colocated with `agent-composer`. `mimiri` retired — superseded by `rag-ingest` in `agent-composer`. |

### Repo-to-Host Quick Matrix (target state)

| Repo | Primary host |
|------|--------------|
| `kimini-api` | `192.168.1.128` |
| `somnus` | `192.168.1.128` |
| `gothmog` | `192.168.1.86` |
| Qdrant | `192.168.1.138` (inside `agent-composer`) |
| PostgreSQL (`:5433`) / Redis (`:6380`) / MinIO (`:9000`) | `192.168.1.121` (NAS) |
| `sauron` | `192.168.1.128` |
| ~~`mimiri`~~ | retired — `rag-ingest` lives in `agent-composer` |
| `imogen` | `192.168.1.109` |
| `rubedo` | `192.168.1.109` |
| `vidita` | `192.168.1.109` |
| `loraline` | `192.168.1.109` |
| `voluptas` | `192.168.1.109` |
| `tss-stack` | `192.168.1.138` |
| `agent-composer` (+ Qdrant + rag-ingest) | `192.168.1.138` |

### Validation Commands (2-3 minute smoke check)

Run these from any host with LAN reachability:

```bash
# NAS data layer (.121)
docker run --rm -e PGPASSWORD=testpass postgres:16 \
  psql -h 192.168.1.121 -p 5433 -U testuser -d testdb -c "SELECT 1" >/dev/null || echo "postgres (NAS) down"
redis-cli -h 192.168.1.121 -p 6380 -a testpass ping || echo "redis (NAS) down"
curl -fsS http://192.168.1.121:9000/minio/health/live || echo "minio (NAS) down"

# Control plane + observability (.128)
curl -fsS http://192.168.1.128:8000/health || echo "kimini down"
curl -fsS http://192.168.1.86:8030/health || echo "gothmog down"
curl -fsS http://192.168.1.128:9090/-/healthy || echo "prometheus down"

# GPU generation node (.109)
curl -fsS http://192.168.1.109:8003/health || echo "imogen down"
# rubedo: port/path from that service’s deploy (not fixed here)
curl -fsS http://192.168.1.109:8000/videos/health || echo "vidita down"
curl -fsS http://192.168.1.109:8010/health || echo "loraline down"
curl -fsS http://192.168.1.109:8188/system_stats || echo "voluptas (comfyui) down"

# Voice/chat node (.138)
curl -fsS http://192.168.1.138:9001/health || echo "tss-stack down"
curl -fsS http://192.168.1.138:9150/health || echo "agent-composer-rag down"
curl -fsS http://192.168.1.138:9050/health || echo "rag-ingest down"
curl -fsS http://192.168.1.138:6333/collections >/dev/null || echo "qdrant down"
curl -fsS http://192.168.1.138:11434/api/tags >/dev/null || echo "ollama down"

# Optional observability targets (expected to fail if intentionally not deployed)
curl -fsS http://192.168.1.138:9100/metrics >/dev/null || echo "node-host-b exporter down"
curl -fsS http://192.168.1.109:9091/-/ready || echo "pushgateway down"
```

Expected baseline for this homelab profile:

- Core checks (`kimini`, `gothmog`, `imogen`, `rubedo` if deployed, `vidita`, `loraline`, `tss-stack`, `agent-composer-rag`, `rag-ingest`, `qdrant`, `ollama`) should pass.
- `node-host-b exporter` and `pushgateway` may fail if not intentionally deployed yet.
- `somnus` should be reachable on its configured frontend route on `192.168.1.128` when the familiar UI is deployed.

### Practical notes (single-job capacity first)

- Keep `kimini` workers (`imagegen`, `videogen`, `voicegen`, `loragen`) on `192.168.1.128`; call GPU services remotely over LAN.
- Run `somnus` on `192.168.1.128` alongside `kimini` for low-latency operator UI/API interactions.
- `gothmog` runs on `192.168.1.86:8030`.
- Use `192.168.1.109` as the primary inference/training target for `imogen`, `rubedo` (paired with imogen callbacks), `vidita`, and `loraline`.
- Use `192.168.1.138` as the default voice/chat node (`tss-stack`, `agent-composer`) with local Ollama, Qdrant, and `rag-ingest`. `mimiri` is retired.
- No automatic failover needed in this profile; optimize for deterministic single-job behavior and simpler operations.
- PostgreSQL (`:5433`), Redis (`:6380`), and MinIO (`:9000`) run on the DS923+ NAS (`192.168.1.121`) — stateful infra is decoupled from the control-plane VM.
- Qdrant is colocated with `agent-composer` on `192.168.1.138`; `rag-ingest` uses `QDRANT_URL=http://qdrant:6333` (container-local) and `OLLAMA_URL=http://ollama:11434` (container-local).

---

### Cutover Checklist (status-aware)

Status legend: `[x] done`, `[ ] remaining`, `[~] partial`.

Based on latest probe report from this host:

- [x] `rag-ingest` and Qdrant colocated in `agent-composer` on `192.168.1.138` — `mimiri` retired.
- [x] Core control-plane/data services are reachable with metrics (`kimini`, `gothmog`, `redis-exporter`, `postgres-exporter`, `node-host-a`, `node-host-c`).
- [x] `tss-stack` and `loraline` are reachable with metrics.
- [~] `imogen` and `vidita` are reachable but do not expose `/metrics` (health only).
- [ ] Confirm all ingest callers/scrape targets now point to `agent-composer` host at `192.168.1.138:9050`.
- [ ] Update Prometheus scrape config in `sauron` — move Qdrant target from `192.168.1.128:6333` to `192.168.1.138:6333`; update gothmog target to `192.168.1.86:8030`.
- [ ] Update Redis scrape target port in `sauron` from `6379` to `6380`.
- [ ] `node-host-b` exporter (`192.168.1.138:9100`) is still unreachable.
- [ ] `pushgateway` scrape target is still unreachable from this vantage point.
- [ ] Ensure `kimini` production env routes image/video/lora calls to `192.168.1.109` and orchestrator calls to `192.168.1.86:8030`.

For this homelab profile, no failover setup is required; close the remaining items only to improve observability and endpoint correctness.

---

## Request Flow: Direct Generation

```
Client
  │
  ├─► POST /v1/images/generate ──► Kimini ──► TaskIQ worker ──► Imogen gateway
  │                                                                 │
  │                                               flux/sdxl ◄───────┘
  │                                                  │
  │                                            GPU inference
  │                                                  │
  │                    optional HTTP callback ───────►│ Rubedo (post-process)
  │                                                  │
  │                                            MinIO upload
  │                                                  │
  │   poll GET /v1/images/{job_id} ◄── Kimini ◄──── Redis job status
  │   or subscribe WS job:{job_id}
  │
  ├─► POST /v1/videos/generate ──► Kimini ──► TaskIQ worker ──► Vidita gateway
  │                                                                 │
  │                                                wan22 ◄──────────┘
  │                                                  │
  │                                            GPU inference → MinIO
  │
  ├─► POST /v1/voice/tts ──► Kimini ──► TaskIQ worker ──► tss-stack gateway :9001
  │                                                           │
  │                                              tts-worker ◄─┘──► xtts engine
  │                                                  │
  │                                            MinIO upload (presigned URL)
  │
  ├─► POST /v1/voice/stt ──► Kimini ──► TaskIQ worker ──► tss-stack gateway :9001
  │                                                           │
  │                                          whisper-worker ◄─┘──► Whisper model
  │                                                  │
  │                                            MinIO upload (transcript)
  │
  └─► POST /v1/loras ──► Kimini ──► TaskIQ worker ──► LoraLine ──► trainer subprocess
```

---

## Request Flow: ComfyUI (somnus → voluptas direct)

```
Operator (browser)
  │
  │  WebSocket ws://NEXT_PUBLIC_COMFYUI_WS_URL/ws?clientId=X
  │  ───────────────────────────────────────────────────────► Voluptas :8188 (direct, LAN-only)
  │                                                            (WEB_ENABLE_AUTH=false)
  │
  ├─► POST /api/comfyui/prompt ──► somnus (server) ──proxy──► Voluptas :8188
  │         (submit workflow)                                   │
  │                                                             │  GPU inference (ComfyUI)
  │                                                             │
  │  WS message: executing / progress ◄──────────────────────────┘
  │  WS message: node=null → run complete
  │
  ├─► GET /api/comfyui/history/{prompt_id} ──proxy──► Voluptas /history/{id}
  │         (fetch output images)
  │
  ├─► GET /api/comfyui/view?filename=...&type=output ──proxy──► Voluptas /view
  │         (stream output image bytes)
  │
  └─► GET/POST/PUT/DELETE /api/comfyui-workflows[/{id}]
            (workflow CRUD — reads/writes MinIO uploads/comfyui/workflows/{id}.json)
            No Kimini in this path — somnus owns the workflow library directly.
```

Note: `COMFYUI_API_URL` (server-side proxy target) and `NEXT_PUBLIC_COMFYUI_WS_URL` (browser WS) are
separate env vars. The HTTP proxy uses the server-side var; the WebSocket connects directly from the
browser (no server-side WS proxy needed since ComfyUI is LAN-gated).

---

## Request Flow: Orchestrated Pipeline

```
Client
  │
  POST /v1/orchestrate/run { workflow: "character_create", input: {...} }
  │
  ▼
Kimini ──proxy──► Gothmog
                    │
                    ▼
              LangGraph executes workflow DAG:
              ┌──────────────────────────────────────────────────────────┐
              │                                                          │
              │  1. expand_character  ──► LLM (Ollama / OpenAI / etc.)   │
              │         │                                                │
              │         ▼                                                │
              │  2. generate_portraits ──► Imogen (3× txt2img)           │
              │         │                                                │
              │         ▼                                                │
              │  3. moderate_portraits ──► (placeholder / NudeNet)       │
              │         │                                                │
              │         ▼                                                │
              │  4. generate_video_intro ──► Vidita (img2vid)            │
              │                                                          │
              └──────────────────────────────────────────────────────────┘
                    │
                    ▼
              Result stored in Redis + Postgres
              │
              ▼
Client polls GET /v1/orchestrate/runs/{run_id}
  or streams via POST /v1/orchestrate/run/stream (SSE)
```

---

## Async Job Pattern (universal)

Every generation domain follows the same contract:

```
  Client                Kimini                  Backend Gateway
    │                     │                          │
    │  POST /v1/{domain}  │                          │
    │────────────────────►│  enqueue (TaskIQ/Redis)  │
    │  ◄── 202 {job_id}   │                          │
    │                     │  worker picks up job     │
    │                     │─────────────────────────►│  POST /{domain}/generate
    │                     │  ◄── 202 {backend_id}    │
    │                     │                          │  GPU inference...
    │                     │  poll loop               │
    │                     │─────────────────────────►│  GET /{domain}/jobs/{id}
    │                     │  ◄── {status, result}    │
    │                     │                          │
    │  poll or WS sub     │  update job in DB        │
    │────────────────────►│                          │
    │  ◄── {completed}    │                          │
```

Domains using this pattern: `imagegen`, `videogen`, `voicegen`, `loragen`

---

## Observability Flow (sauron)

```
Runtime services + infra exporters
  │
  ├─ metrics endpoints (/metrics) ───────────────► Prometheus (:9090)
  │                                                │
  │                                                ├─ alert rules ─► Alertmanager (:9093)
  │                                                └─ datasource ──► Grafana (:3001)
  │
  └─ Docker container logs ─► Promtail ───────────► Loki (:3100) ──► Grafana Explore

Remote host metrics scrapes are best-effort/opportunistic. In this homelab profile, degraded visibility is acceptable when a non-critical exporter is down.
```
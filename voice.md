# Voice Stack

faster-whisper STT (port 8030, GPU) and Qwen3-TTS (port 8031, GPU), integrated
into the main `docker-compose.yml`. Both services share the GPU — inference is
bursty, not concurrent-heavy.

## Prerequisites

- NVIDIA Container Toolkit on the Docker host
- GPU with >= 6 GB VRAM (RTX 4060 8 GB confirmed)

## Step 1: Build and start

```bash
cd ~/agent-composer

# Build voice services (first time downloads models — ~15 min total)
docker compose build whisper tts

# Start everything (or just the voice services)
docker compose up -d
```

## Step 3: Verify

```bash
# Health checks
curl -s http://localhost:8030/health | python3 -m json.tool
curl -s http://localhost:8031/health | python3 -m json.tool

# Test Whisper (needs an audio file)
curl -X POST http://localhost:8030/transcribe \
  -F "audio=@test.webm" | python3 -m json.tool

# Test TTS
curl -X POST http://localhost:8031/synthesise \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test.", "speaker": "Chelsie"}' \
  | python3 -m json.tool

# Voice UI
open http://localhost:9150/ui/voice
```

## Architecture

```
voice-stack/
├── Dockerfile.whisper     # faster-whisper + large-v3-turbo (self-contained)
├── Dockerfile.tts         # Qwen3-TTS 0.6b (self-contained, no compance-base)
├── whisper_server.py      # FastAPI STT server (faster-whisper backend)
└── tts_server.py          # Enhanced TTS server (evolved from playact-engine)
```

Both services are defined in the main `docker-compose.yml` alongside the rest of
the stack. Build context is `./voice-stack`. Both Dockerfiles are self-contained.

This stack replaces [tss-stack](https://github.com/smk762/tss-stack/tree/dev),
which required 6+ containers (gateway, redis, minio, whisper-worker, tts-worker,
xtts, xtts-glue) to wrap two synchronous GPU processes. voice-stack eliminates
the redundant job queue and artifact storage — consumers call the inference
servers directly.

- Snapcast delivery and voice-preset selection → absorbed by mithrandir
- STT/TTS inference and voice registry → owned by voice-stack (this repo)

Consumers: rag-chat (proxied via `/api/voice/*`), playact-engine (direct),
mithrandir (direct), kimini-api (direct via `/transcriptions`).

## Configuration

### Whisper (STT)

| Env var | Default | Description |
|---------|---------|-------------|
| `STT_MODEL` | `large-v3-turbo` | CTranslate2 model id |
| `STT_DEVICE` | `cuda` | `cuda` or `cpu` |
| `STT_COMPUTE_TYPE` | `float16` | `float16`, `int8`, `int8_float16`, `float32` |
| `STT_VAD_FILTER` | `1` | Silero VAD pre-filter (faster on audio with silence) |
| `STT_BEAM_SIZE` | `5` | Beam search width |

### TTS

| Env var | Default | Description |
|---------|---------|-------------|
| `TTS_MODEL_SIZE` | `0.6b` | `0.6b` or `1.7b` |
| `TTS_DEVICE` | `cuda:0` | `cuda:0` or `cpu` |
| `TTS_LOAD_MODELS` | `both` | `both`, `custom_voice`, or `base` |
| `S3_ENDPOINT_URL` | `http://192.168.1.109:9000` | MinIO/S3 for audio storage |

### rag-chat voice proxy

| Env var | Default | Description |
|---------|---------|-------------|
| `WHISPER_URL` | `http://whisper:8030` | Internal Docker URL (set empty to disable) |
| `TTS_URL` | `http://tts:8031` | Internal Docker URL (set empty to disable) |
| `VOICE_TIMEOUT` | `30` | Proxy timeout in seconds |

## GPU error handling

Both servers catch CUDA OOM at the request level and return 503 with structured
error details — no process crashes. GPU memory state is logged on failure.

Future: integrate with gothmog orchestrator for GPU semaphore / scheduling across
services (whisper, TTS, imogen, ollama, infinity, modernbert).

## Resource budget

| Service | CPU | RAM | VRAM |
|---------|-----|-----|------|
| faster-whisper large-v3-turbo (fp16, GPU) | Low | ~1.5 GB | ~2.5 GB |
| TTS 0.6b both models (bfloat16, CUDA) | Low | ~2 GB | ~2.5 GB |
| **Total peak** | Bursty | ~4 GB | ~5.5 GB |
| **Available (RTX 4060)** | — | — | 8 GB |

~2.5 GB VRAM headroom. INT8 quantization for Whisper would free another ~1 GB if
needed later.

## Standalone deployment (alternative)

For deploying voice services on a separate host without the full stack, a
standalone `voice-stack/compose.yml` is also provided. Set `WHISPER_URL` and
`TTS_URL` in the main `.env` to point at the remote host's LAN IP.

## API Endpoints

### Whisper STT

| Endpoint | Format | Consumer | Description |
|----------|--------|----------|-------------|
| `POST /transcribe` | multipart (`audio` field) | playact-engine, rag-chat | Returns `{text, language, duration}` |
| `POST /transcriptions` | JSON (`audio_url` or `audio_base64`) | kimini-api | Returns `{text, language, duration_seconds, confidence}` |
| `GET /health` | — | all | Readiness check |

The `/transcriptions` endpoint accepts either a URL to download or base64-encoded
audio. Exactly one of `audio_url` or `audio_base64` must be present. This replaces
the tss-stack gateway + Redis + whisper-worker chain with a single synchronous call.

### Qwen3-TTS

| Endpoint | Format | Consumer | Description |
|----------|--------|----------|-------------|
| `POST /synthesise` | JSON | all | Returns `{audio_url, duration, duration_seconds, format}` |
| `GET /speakers` | — | all | Structured list: `[{id, name, language, languages}]` |
| `POST /clone-voice` | multipart | playact-engine | Voice cloning from reference audio |
| `GET /health` | — | all | Returns `{status, engine, model, device}` |

## Consumers

### playact-engine

Add to playact `.env`:

```
WHISPER_SERVER_URL=http://<host>:8032
TTS_SERVER_URL=http://<host>:8031
```

### kimini-api

Add to kimini `.env`:

```
VOICE_STT_BACKEND=direct
WHISPER_API_BASE_URL=http://<host>:8032
WHISPER_TRANSCRIBE_PATH=/transcriptions
```

kimini's TaskIQ worker calls `/transcriptions` synchronously — no intermediate
job queue needed. This replaces the tss-stack gateway (6+ containers) with a
single HTTP call to the whisper container.

### mithrandir

Mithrandir calls `POST /synthesise` with a `speaker` from its persona YAML
(`voice.voice_id`), downloads the audio from `audio_url`, transcodes to FLAC,
and pipes to Snapcast. It also probes `GET /speakers` to validate configured
voice IDs and `GET /health` for readiness.

```yaml
# mithrandir persona config (e.g. config/persona/holly.yaml)
voice:
  voice_id: Chelsie    # must match a speaker id from GET /speakers
  speed: 1.0
  language: en
```

# Voice engines

> Hosts in this doc are placeholders (`<AGENT_HOST>`, `<STATEFUL_HOST>`, …). The real host/IP is in the local `.env` (see `.env.example`); check it there.

The speech engines are [argus-echo](https://github.com/smk762/argus-echo)'s (#81). This stack only deploys them from
`docker-compose.yml`, pinned by `ECHO_TAG`. The engines themselves, their settings and their tests live in echo:
[docs/api.md](https://github.com/smk762/argus-echo/blob/main/docs/api.md) (the contracts),
[docs/xtts.md](https://github.com/smk762/argus-echo/blob/main/docs/xtts.md),
[docs/whisper.md](https://github.com/smk762/argus-echo/blob/main/docs/whisper.md) and
[docs/voice-cloning.md](https://github.com/smk762/argus-echo/blob/main/docs/voice-cloning.md).

| Service | Image | Host port | Role |
|---|---|---|---|
| `whisper` | `argus-echo-whisper` | `127.0.0.1:8032` | Whisper large-v3-turbo: `POST /transcribe`, `POST /transcriptions`, `WS /transcribe/stream` |
| `xtts` | `argus-echo-xtts` | `127.0.0.1:8033` | XTTS-v2, rag-chat's default TTS engine |
| `wyoming-xtts` / `wyoming-miso` | `argus-echo-wyoming` | `10200` / `10201` | Home Assistant `tts` engines (XTTS, remote Miso) |
| `wyoming-whisper` | `argus-echo-wyoming` (`asr_bridge.py`) | `10300` | Home Assistant `stt` engine |

Every engine listens on `8031` inside the compose network (`http://xtts:8031`, `http://whisper:8031`). Each serves
argus-cortex's resident contract (`/health`, `/admin/{evict,unload,wake}`), which argus-iris reads and drives through
rag-chat's gateway (`/api/runtime/residents/<name>`).

Qwen TTS isn't run here: it is echo's `argus-echo-qwen`, its own iris resident, and apps that need it (argus-mask)
call echo directly (#82). Miso is a remote engine (`MISO_TTS_URL`).

## Storage

Each echo engine keeps its clones in its own bucket, because the engines share keys but not formats. XTTS uses
`XTTS_S3_BUCKET` (default `voice-xtts`) and serves its rendered audio from `XTTS_S3_PUBLIC_URL`. echo's XTTS on morpheus
uses the same `voice-xtts` bucket, so a clone made on either XTTS loads on both. rag-chat's own re-hosted and stitched
audio stays in `S3_BUCKET_NAME` (`voice`). `minio-setup` creates both buckets and makes each one's `generated/` prefix
publicly readable.

## Upgrading

Bump `ECHO_TAG` (in `.env` or compose's default), then
`docker compose pull whisper xtts wyoming-xtts wyoming-miso wyoming-whisper && docker compose up -d whisper xtts wyoming-xtts wyoming-miso wyoming-whisper`.

Check an engine against the contract (read-only):

```sh
uvx argus-echo check http://127.0.0.1:8033                     # XTTS
uvx argus-echo check http://127.0.0.1:8032 --transcribe        # Whisper
```

## Consumers

- **rag-chat** proxies `/api/voice/*` (engine registry, iris leases for XTTS; see CLAUDE.md).
- **kimini-api** calls Whisper's `POST /transcriptions` (`VOICE_STT_BACKEND=direct`,
  `WHISPER_API_BASE_URL=http://<host>:8032`, `WHISPER_TRANSCRIBE_PATH=/transcriptions`). `audio_url` is fetched under
  echo's SSRF policy; trusted internal hosts go in `STT_AUDIO_URL_ALLOWED_HOSTS`.
- **mithrandir** calls `POST /synthesise` with a speaker from its persona (`voice.voice_id`, which must match one in
  `GET /speakers`) and probes `GET /health`.
- **Home Assistant** adds the Wyoming bridges under Settings → Devices & Services → Wyoming Protocol (host = this
  stack's LAN IP, ports `10200`, `10201`, `10300`).

# Voice

> Hosts in this doc are placeholders (`<AGENT_HOST>`, `<STATEFUL_HOST>`, …). The real host/IP is in the local `.env` (see `.env.example`); check it there.

agent-composer runs no speech engines (#81). They belong to [argus-echo](https://github.com/smk762/argus-echo), which
runs Whisper, XTTS-v2, Qwen3-TTS and the Home Assistant Wyoming bridges from its own compose. On dragon that is
`~/argus-echo`, published on `ECHO_BIND`. Echo owns their settings, buckets, residency (argus-iris reaches its residents
directly, with echo's admin token), upgrades and tests:
[docs/api.md](https://github.com/smk762/argus-echo/blob/main/docs/api.md) (the contracts),
[docs/xtts.md](https://github.com/smk762/argus-echo/blob/main/docs/xtts.md),
[docs/whisper.md](https://github.com/smk762/argus-echo/blob/main/docs/whisper.md),
[docs/voice-cloning.md](https://github.com/smk762/argus-echo/blob/main/docs/voice-cloning.md).

Composer keeps its voice pathways and proxies them into echo:

| Composer | Points at | Echo's default port |
|---|---|---|
| `/api/voice/transcribe` | `WHISPER_URL` | `8144` (Whisper) |
| `/api/voice/synthesise{,/stream,/pcm}`, clones, speakers (engine `xtts`) | `XTTS_TTS_URL` | `8141` (XTTS) |
| engine `miso` | `MISO_TTS_URL` | `8140` on echo's Miso host (or voluptas's `8034`) |
| engine `tts` (optional custom) | `TTS_URL` | e.g. `8142` (Qwen) |
| `/ui/voice`, `/ui/voice-clone` | the engines above | |

Every XTTS synth still takes an argus-iris lease on the `xtts` target first (#79; `gpu-tenant.yaml` kind `tts-synth`).
A grant on dragon calls `XTTS_TTS_URL`, and a grant elsewhere calls that node's XTTS. A refusal falls back to the next
engine, else to XTTS on its CPU.

Audio echo renders into its own buckets on the same MinIO (`voice-xtts`, …) is served as is. Audio from any other host
is re-hosted into rag-chat's `S3_BUCKET_NAME` (`voice`), as is stitched, paced audio.

Check an engine against echo's contract (read-only): `uvx argus-echo check http://<AGENT_HOST>:8141`.

Other consumers (kimini-api's `/transcriptions`, mithrandir, argus-mask, Home Assistant) call echo directly: see echo's
docs.

# Voice cloning — getting the best results

This stack clones voices through two interchangeable TTS engines. They have
different cloning strategies, so the *optimal reference input differs per
engine*. Pick the engine in the **Engine** dropdown on `/ui/voice-clone`; the
page shows a one-line hint and routes scoring/cloning/testing to that engine.

| | **XTTS-v2** (`xtts`, local) | **Miso One** (`miso`, remote) |
|---|---|---|
| Strategy | Multi-clip — averages a speaker embedding across clips | One-shot — conditions on a single reference clip |
| Ideal input | 6–60 s total across **several** varied clips | **one** clean, expressive ~6–15 s clip |
| More clips help? | Yes (up to the conditioning window) | No — extra clips are not averaged; the best one is used |
| Strength | Robust, noise-averaged timbre | Expressive prosody + emotion, very low latency |

## Universal recording guidance (both engines)

Quality of the reference dominates quality of the clone. Aim for:

- **Clean signal.** Quiet room, no music/TV, minimal reverb. A close mic (15–30 cm)
  beats a distant one. Avoid built-in laptop mics where possible.
- **One speaker only.** No overlapping voices, no background chatter.
- **Natural, expressive delivery.** Speak the way you want the clone to sound —
  conversational pace, normal pitch range. Monotone in → monotone out.
- **Consistent capture.** Same mic, distance, and room across clips. Don't mix a
  phone clip with a studio clip.
- **Good level, no clipping.** Loud enough to sit well above the noise floor, but
  not so hot that peaks distort. The scorer flags clipping.
- **Format.** WAV/FLAC preferred (lossless); MP3/M4A work. Mono or stereo both
  fine — clips are downmixed to mono. Any sample rate (resampled internally).

## XTTS-v2 (`xtts`) — multi-clip

- Provide **3–8 short clips**, ~3–10 s each, totalling **6–60 s**.
- **Vary the content** (different sentences/phonemes) but keep the *voice and
  capture conditions identical*. Variety in words → better phoneme coverage;
  consistency in mic/room → cleaner average.
- Hit **Score clips** before creating. The analyzer ranks each clip by SNR,
  clipping, speech seconds, and speaker-embedding consistency vs the set
  centroid, auto-deselects low scorers, and fixes a best-first order used to
  fill the GPT conditioning window (`XTTS_GPT_COND_LEN`, default 30 s).
- Each clip is auto-standardised before latents are computed: mono → resample →
  high-pass → Silero VAD speech/gap split → SNR-gated HVAC-hum denoise (seeded
  from VAD gaps) → LUFS loudness normalise. The create step returns a per-clip
  report (kept seconds, SNR, gain, denoise applied).

## Miso One (`miso`) — one-shot

- Provide **one** clip, ~6–15 s, of clean, **expressive** speech. This is the
  single most important input — there is no averaging to hide a bad take.
- Prefer a clip that demonstrates the *prosody you want* (warmth, energy,
  cadence). Miso reproduces emotion well, so give it emotion to copy.
- You can still upload several clips and **Score** them — the highest-ranked
  clip is used as the one-shot reference (consistency is reported as `null`
  since Miso has no embedding-averaging step).
- Avoid long silences at the head/tail; trim to mostly-speech.

## After cloning

- Clones are stored per engine in MinIO under `voice_clones/{companion_id}/`
  (`prompt.pt` + `meta.json`) and reused via `voice_clone_id`.
- Test in the **Test a voice** panel (normal or live/streaming playback).
- **Download** a clone as a `.zip` to back it up or move it; **Import** restores
  it into the selected engine.
- Clones are engine-specific: a voice cloned on `xtts` lives on the XTTS sidecar,
  one cloned on `miso` lives on the Miso host. Switch the engine dropdown to see
  the clones stored on that engine.

## Home Assistant

Both engines are exposed to HA over Wyoming as independent `tts` engines:
`wyoming-xtts` (port `10200`) and `wyoming-miso` (port `10201`, enabled once
`MISO_TTS_URL` is set). Each advertises its stored clones as `clone:<companion_id>`.
Add via **Settings → Devices & Services → Wyoming Protocol** using this host's
LAN IP and the relevant port.

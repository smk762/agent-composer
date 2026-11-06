"""Wyoming protocol bridges fronting the XTTS and Whisper HTTP sidecars.

These let Home Assistant (or any Wyoming client) use the agent-composer voice
stack as native `tts` / `asr` engines — including XTTS cloned voices — without
HA ever speaking the bespoke HTTP APIs directly.
"""

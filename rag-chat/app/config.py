import logging
import os
from datetime import timezone
from typing import Optional

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s")
log = logging.getLogger("rag-chat")
for _quiet in ("aiosqlite", "sqlalchemy.engine", "httpcore", "httpx", "hpack"):
    logging.getLogger(_quiet).setLevel(logging.WARNING)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "15m").strip()
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
# SingGuard 4B. The 4B is the image-safety sweet spot in the model card;
# the 8B does not beat it there. Q4_K_M is the default pull; Q8_0 and F16
# are on the same repo. Override with GUARD_MODEL to pick another size.
GUARD_MODEL = os.getenv(
    "GUARD_MODEL",
    "hf.co/inclusionAI/SingGuard-4b-GGUF:Q4_K_M",
)
# fast-slow asks the model to write why the rule fired. fast is the label only.
# slow reasons first (no first-line verdict). cascade is fast, then slow if flagged.
# Keep in sync with guard_policy.THINKING_MODES.
_GUARD_THINKING = os.getenv("GUARD_THINKING", "fast-slow").strip().lower()
GUARD_THINKING = (
    _GUARD_THINKING
    if _GUARD_THINKING in ("fast", "fast-slow", "slow", "cascade")
    else "fast-slow"
)
SYSTEM_PROMPT = os.getenv("CHAT_SYSTEM_PROMPT", "You are a helpful assistant.")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "project_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
RAG_ENABLED = os.getenv("RAG_ENABLED", "1") == "1"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000"))
DEV_AUTH_BYPASS = os.getenv("DEV_AUTH_BYPASS", "0") == "1"
DEV_DEFAULT_USER = os.getenv("DEV_DEFAULT_USER", "dev@example.com")
CHAT_HISTORY_MAX_MSGS = int(os.getenv("CHAT_HISTORY_MAX_MSGS", "100"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
OOM_WAIT_MAX_SECONDS = _env_int("OOM_WAIT_MAX_SECONDS", 180)
OOM_WAIT_INTERVAL_SECONDS = _env_int("OOM_WAIT_INTERVAL_SECONDS", 5)
INFERENCE_WORKER_CONCURRENCY = _env_int("INFERENCE_WORKER_CONCURRENCY", 1)
# One SingGuard pass at a time: cull scans and darkroom pre-flights share Ollama.
GUARD_WORKER_CONCURRENCY = _env_int("GUARD_WORKER_CONCURRENCY", 1)

MEDIA_DIR = os.getenv("MEDIA_DIR", "/data/media")
MEDIA_BACKEND = os.getenv("MEDIA_BACKEND", "local").strip().lower()
MEDIA_S3_ENDPOINT_URL = os.getenv("MEDIA_S3_ENDPOINT_URL", "").strip()
MEDIA_S3_BUCKET = os.getenv("MEDIA_S3_BUCKET", "").strip()
MEDIA_S3_REGION = os.getenv("MEDIA_S3_REGION", "us-east-1").strip()
MEDIA_S3_ACCESS_KEY_ID = os.getenv("MEDIA_S3_ACCESS_KEY_ID", "").strip()
MEDIA_S3_SECRET_ACCESS_KEY = os.getenv("MEDIA_S3_SECRET_ACCESS_KEY", "").strip()
MEDIA_S3_USE_SSL = os.getenv("MEDIA_S3_USE_SSL", "0") == "1"
MEDIA_S3_PREFIX = os.getenv("MEDIA_S3_PREFIX", "rag-chat-media").strip().strip("/")

PROVIDER_ENCRYPTION_KEY = os.getenv("PROVIDER_ENCRYPTION_KEY", "")

MODERNBERT_URL = os.getenv("MODERNBERT_URL", "http://modernbert:7998")
INFINITY_URL = os.getenv("INFINITY_URL", "http://infinity:7997").rstrip("/")

# ── Voice services ────────────────────────────────────────────────────────────
# STT (Whisper) is a single sidecar. TTS is a *registry* of one or more engines
# that all speak the XTTS sidecar HTTP contract (/synthesise, /synthesise/stream,
# /clone-voice, /clones, /speakers, /health, /analyze-clips). Each engine is
# selectable per request and one is the default — so the local XTTS GPU sidecar
# and a remote Miso host (e.g. morpheus) can coexist and be picked from the UI,
# the API, or the Wyoming/Home Assistant bridges.
WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper:8030").rstrip("/")
VOICE_TIMEOUT = _env_int("VOICE_TIMEOUT", 30)


def _clean_url(value: str) -> str:
    return (value or "").strip().rstrip("/")


# Per-engine base URLs. Empty → that engine is not available.
XTTS_TTS_URL = _clean_url(os.getenv("XTTS_TTS_URL", "http://xtts:8031"))
MISO_TTS_URL = _clean_url(os.getenv("MISO_TTS_URL", ""))
# Legacy single-engine knob (back-compat / rollback to the Qwen sidecar).
_LEGACY_TTS_URL = _clean_url(os.getenv("TTS_URL", ""))

# Human-friendly labels shown in the UI engine picker.
TTS_ENGINE_LABELS = {
    "miso": "Miso One",
    "xtts": "XTTS-v2",
    "tts": "Qwen3-TTS (legacy)",
}

# Build the registry in a stable, human-meaningful order.
TTS_ENGINES: dict[str, str] = {}
for _name, _url in (("xtts", XTTS_TTS_URL), ("miso", MISO_TTS_URL)):
    if _url:
        TTS_ENGINES[_name] = _url
# A custom TTS_URL that isn't already one of the named engines is registered as
# its own "tts" engine (the legacy Qwen sidecar) so rollback keeps working.
if _LEGACY_TTS_URL and _LEGACY_TTS_URL not in TTS_ENGINES.values():
    TTS_ENGINES["tts"] = _LEGACY_TTS_URL

# Resolve the default engine: explicit env wins, else prefer miso, then xtts,
# then whatever was configured first.
_requested_default = os.getenv("TTS_DEFAULT_ENGINE", "").strip().lower()
if _requested_default and _requested_default in TTS_ENGINES:
    TTS_DEFAULT_ENGINE = _requested_default
elif "miso" in TTS_ENGINES:
    TTS_DEFAULT_ENGINE = "miso"
elif "xtts" in TTS_ENGINES:
    TTS_DEFAULT_ENGINE = "xtts"
else:
    TTS_DEFAULT_ENGINE = next(iter(TTS_ENGINES), "")

# Back-compat export: many call sites import TTS_URL directly. Point it at the
# default engine's URL so they transparently use the chosen default.
TTS_URL = TTS_ENGINES.get(TTS_DEFAULT_ENGINE, _LEGACY_TTS_URL or XTTS_TTS_URL)

# Fallback engine used when the *default* engine is unreachable (e.g. a remote
# Miso host is down). Only applies to requests that don't name an engine — an
# explicitly selected engine is always honoured, never silently switched.
# Explicit env wins; otherwise auto-fall back to local XTTS when the default is
# something else (so "prefer Miso, fall back to XTTS" works out of the box).
_requested_fallback = os.getenv("TTS_FALLBACK_ENGINE", "").strip().lower()
if _requested_fallback in TTS_ENGINES and _requested_fallback != TTS_DEFAULT_ENGINE:
    TTS_FALLBACK_ENGINE = _requested_fallback
elif not _requested_fallback and TTS_DEFAULT_ENGINE != "xtts" and "xtts" in TTS_ENGINES:
    TTS_FALLBACK_ENGINE = "xtts"
else:
    TTS_FALLBACK_ENGINE = ""

# Rendered-audio re-hosting. Some engines (e.g. a remote Miso host) return an
# ``audio_url`` on their *own* host, which LAN browsers on a different subnet may
# not be able to reach. When the shared "voice" MinIO bucket is configured here,
# the voice proxy downloads such audio and re-uploads it under the same
# bucket/prefix the XTTS sidecar uses, returning the browser-reachable
# S3_PUBLIC_URL so playback always comes from one origin. Reuses the same S3_*
# vars the sidecars use. Empty S3_PUBLIC_URL/S3_ENDPOINT_URL disables re-hosting
# (URLs are passed through untouched).
VOICE_S3_ENDPOINT_URL = _clean_url(os.getenv("S3_ENDPOINT_URL", ""))
VOICE_S3_BUCKET = os.getenv("S3_BUCKET_NAME", "voice").strip()
VOICE_S3_PUBLIC_URL = _clean_url(os.getenv("S3_PUBLIC_URL", ""))
VOICE_S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "").strip()
VOICE_S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "").strip()
VOICE_S3_REHOST = bool(VOICE_S3_ENDPOINT_URL and VOICE_S3_PUBLIC_URL and VOICE_S3_BUCKET)

# Sentence pacing. Miso (CSM-style) reads multi-sentence text as one fast
# generation with no breath between sentences, and ignores emotion/speed. When a
# request doesn't set an explicit pause, the proxy inserts this much silence
# between sentences for the given engine (0 = off). Miso defaults on; other
# engines off. Any request can override via ``pause_ms`` (capped at MAX).
MISO_SENTENCE_PAUSE_MS = _env_int("MISO_SENTENCE_PAUSE_MS", 300)
MAX_SENTENCE_PAUSE_MS = _env_int("MAX_SENTENCE_PAUSE_MS", 2000)


def default_sentence_pause_ms(engine: Optional[str]) -> int:
    """Per-engine default inter-sentence silence when a request omits pause_ms."""
    return MISO_SENTENCE_PAUSE_MS if (engine or "").strip().lower() == "miso" else 0


def tts_engine_url(engine: Optional[str] = None) -> str:
    """Resolve a TTS engine name to its base URL.

    An unknown or empty engine falls back to the default engine. Returns "" only
    when no TTS engine is configured at all.
    """
    if engine:
        url = TTS_ENGINES.get(engine.strip().lower())
        if url:
            return url
    return TTS_URL


def tts_engine_chain(engine: Optional[str] = None) -> list:
    """Ordered ``(name, url)`` candidates to try for a request.

    A request that explicitly names a known engine uses *only* that engine (no
    surprise fallback). A request relying on the default gets the default engine
    followed by the configured fallback engine (if any), so callers can retry
    the next candidate when the preferred host is unreachable.
    """
    if engine:
        name = engine.strip().lower()
        if name in TTS_ENGINES:
            return [(name, TTS_ENGINES[name])]
        # Unknown engine name → treat as "use the default chain".
    chain = []
    if TTS_DEFAULT_ENGINE in TTS_ENGINES:
        chain.append((TTS_DEFAULT_ENGINE, TTS_ENGINES[TTS_DEFAULT_ENGINE]))
    if (
        TTS_FALLBACK_ENGINE
        and TTS_FALLBACK_ENGINE in TTS_ENGINES
        and TTS_FALLBACK_ENGINE != TTS_DEFAULT_ENGINE
    ):
        chain.append((TTS_FALLBACK_ENGINE, TTS_ENGINES[TTS_FALLBACK_ENGINE]))
    return chain

# ── Agentic repair pipeline ────────────────────────────────────────────────────
# URL of the ai-code-auditor audit API — used for diff-audit validation between
# repair iterations.  Leave empty to disable validation.
AUDIT_API_URL = os.getenv("AUDIT_API_URL", "http://127.0.0.1:8765")
# Path to the ai-code-auditor ecosystem.yaml — used to resolve repo paths for
# apply-in-place and compare-branch operations.
ECOSYSTEM_CONFIG_PATH = os.getenv(
    "ECOSYSTEM_CONFIG_PATH",
    "/home/smk/ai-code-auditor/config/ecosystem.yaml",
)

UTC = timezone.utc


def _normalize_db_url(url: str) -> str:
    if url.startswith("sqlite:") and "+aiosqlite" not in url:
        return url.replace("sqlite:", "sqlite+aiosqlite:", 1)
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://") and "+" not in url.split("://", 1)[0]:
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


_raw_db_url = (
    os.getenv("CHAT_DB_URL")
    or os.getenv("DATABASE_URL")
    or "sqlite:////data/chat.db"
)
CHAT_DB_URL = _normalize_db_url(_raw_db_url)

log.info(
    "config: ollama=%s model=%s rag=%s qdrant=%s log_level=%s dev_auth_bypass=%s timeout=%ds",
    OLLAMA_URL, CHAT_MODEL, RAG_ENABLED, QDRANT_URL, LOG_LEVEL, DEV_AUTH_BYPASS, OLLAMA_TIMEOUT,
)
log.info(
    "config: tts_engines=%s default=%s fallback=%s whisper=%s",
    list(TTS_ENGINES) or "none", TTS_DEFAULT_ENGINE or "none",
    TTS_FALLBACK_ENGINE or "none", WHISPER_URL or "none",
)

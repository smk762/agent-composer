"""Environment-driven configuration for rag-chat: Ollama/chat model, the guard upstream, Qdrant RAG,
auth, media storage, ModernBERT/Infinity/Whisper URLs, audit API, and the TTS engine
registry (TTS_ENGINES, tts_engine_url(), tts_engine_chain()) with default/fallback engines.
"""

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
# keep_alive sent on every Ollama request: OLLAMA_REQUEST_KEEP_ALIVE, else
# OLLAMA_KEEP_ALIVE (which compose also hands the Ollama daemon as its default, #53).
# "server" (or "none"/"off", or an empty OLLAMA_KEEP_ALIVE) sends none, so the Ollama
# server's own keep-alive governs residency: set OLLAMA_REQUEST_KEEP_ALIVE=server when
# argus-iris owns idle/keepalive policy, or each request resets the timer.
_KEEP_ALIVE_RAW = (os.getenv("OLLAMA_REQUEST_KEEP_ALIVE", "").strip()
                   or os.getenv("OLLAMA_KEEP_ALIVE", "15m").strip())
OLLAMA_KEEP_ALIVE = "" if _KEEP_ALIVE_RAW.lower() in ("", "server", "none", "off") else _KEEP_ALIVE_RAW


def ollama_keep_alive() -> dict:
    """``{"keep_alive": ...}`` to splat into an Ollama payload, or ``{}``."""
    return {"keep_alive": OLLAMA_KEEP_ALIVE} if OLLAMA_KEEP_ALIVE else {}


CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
# SingGuard content checks are argus-retina's SingGuard server (ghcr.io/smk762/argus-retina-singguard);
# /api/guard/check proxies to it. Empty: the route answers 503 "guard not configured".
GUARD_URL = os.getenv("GUARD_URL", "").strip().rstrip("/")
# Slow and cascade checks take about 20 s on a 3090, and retina queues one check at a time.
GUARD_TIMEOUT = float(os.getenv("GUARD_TIMEOUT", "120"))
# fast-slow asks the model to write why the rule fired. fast is the label only.
# slow reasons first (no first-line verdict). cascade is fast, then slow if flagged.
# Sent when a check leaves thinking blank (retina's own default is fast).
# Keep in sync with guard_policy.THINKING_MODES.
_GUARD_THINKING = os.getenv("GUARD_THINKING", "fast-slow").strip().lower()
GUARD_THINKING = (
    _GUARD_THINKING
    if _GUARD_THINKING in ("fast", "fast-slow", "slow", "cascade")
    else "fast-slow"
)
SYSTEM_PROMPT = os.getenv("CHAT_SYSTEM_PROMPT", "You are a helpful assistant.")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "audit_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
# Task prefixes (#32), the same setting rag-ingest reads: ``nomic`` embeds retrieval queries
# as "search_query: <query>" (rag-ingest embeds chunks as "search_document: <chunk>").
_QUERY_PREFIXES = {"none": "", "nomic": "search_query: "}
EMBED_TASK_PREFIXES = (os.getenv("EMBED_TASK_PREFIXES", "none").strip().lower() or "none")
if EMBED_TASK_PREFIXES not in _QUERY_PREFIXES:
    raise RuntimeError(f"EMBED_TASK_PREFIXES must be one of {sorted(_QUERY_PREFIXES)}, not {EMBED_TASK_PREFIXES!r}")
QUERY_PREFIX = _QUERY_PREFIXES[EMBED_TASK_PREFIXES]
# rag-chat embeds queries with Ollama + EMBED_MODEL (+ prefix scheme). rag-ingest stamps text
# points with the same label (``embedder`` payload), so retrieval can refuse a collection whose
# vectors came from another embedder.
TEXT_EMBEDDER = f"ollama:{EMBED_MODEL}" + ("" if EMBED_TASK_PREFIXES == "none" else f"+{EMBED_TASK_PREFIXES}-prefix")
EMBEDDER_CHECK_TTL_S = float(os.getenv("EMBEDDER_CHECK_TTL_S", "60"))
# Code collection (rag-ingest's code path: Infinity + EMBED_MODEL_CODE, label CODE_EMBEDDER).
# The repair loop searches it for related code; queries must use the same embedder (#35).
QDRANT_COLLECTION_CODE = os.getenv("QDRANT_COLLECTION_CODE", "code_audit")
EMBED_MODEL_CODE = os.getenv("EMBED_MODEL_CODE", "nomic-ai/CodeRankEmbed")
CODE_EMBEDDER = f"infinity:{EMBED_MODEL_CODE}"
# Query instruction for the code embedder (CodeRankEmbed's model card); stored code is unprefixed.
_DEFAULT_CODE_QUERY_PREFIX = (
    "Represent this query for searching relevant code: " if "coderankembed" in EMBED_MODEL_CODE.lower() else ""
)
CODE_QUERY_PREFIX = os.getenv("EMBED_CODE_QUERY_PREFIX", _DEFAULT_CODE_QUERY_PREFIX)
# Strict: unlabelled points (written before labels existed) count as a mismatch too.
EMBEDDER_STRICT = os.getenv("EMBEDDER_STRICT", "0").strip().lower() in ("1", "true", "yes")
RAG_ENABLED = os.getenv("RAG_ENABLED", "1") == "1"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000"))
DEV_AUTH_BYPASS = os.getenv("DEV_AUTH_BYPASS", "0") == "1"
DEV_DEFAULT_USER = os.getenv("DEV_DEFAULT_USER", "dev@example.com")
# Other users the dev bypass may act as via X-Dev-User (comma-separated). Empty: only DEV_DEFAULT_USER,
# so a caller can't name an arbitrary (real) user just by sending the header.
DEV_AUTH_USERS = frozenset(u.strip() for u in os.getenv("DEV_AUTH_USERS", "").split(",") if u.strip())
# Cloudflare Access: identity comes only from a verified Cf-Access-Jwt-Assertion
# (or CF_Authorization cookie). Both must be set or Access identity is refused.
CF_ACCESS_TEAM_DOMAIN = os.getenv("CF_ACCESS_TEAM_DOMAIN", "").strip().removeprefix("https://").rstrip("/")
CF_ACCESS_AUD = os.getenv("CF_ACCESS_AUD", "").strip()
CHAT_HISTORY_MAX_MSGS = int(os.getenv("CHAT_HISTORY_MAX_MSGS", "100"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
OOM_WAIT_MAX_SECONDS = _env_int("OOM_WAIT_MAX_SECONDS", 180)
OOM_WAIT_INTERVAL_SECONDS = _env_int("OOM_WAIT_INTERVAL_SECONDS", 5)
INFERENCE_WORKER_CONCURRENCY = _env_int("INFERENCE_WORKER_CONCURRENCY", 1)
# Largest image (or mask) /api/edit/image and /api/generate/video accept; bigger is a 413.
GENERATION_UPLOAD_MAX_BYTES = _env_int("GENERATION_UPLOAD_MAX_BYTES", 25 * 1024 * 1024)

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
# Bearer token the GPU sidecars require on /admin/* when set (argus-iris resident
# contract); rag-chat forwards it when it proxies admin calls.
ARGUS_ADMIN_TOKEN = os.getenv("ARGUS_ADMIN_TOKEN", "").strip()
RESIDENT_ADMIN_TIMEOUT = float(os.getenv("RESIDENT_ADMIN_TIMEOUT", "300"))


def _parse_residents(raw: str) -> dict[str, str]:
    """``name=url,name=url`` → {name: url}; blank or malformed entries are skipped (logged)."""
    out: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, url = item.partition("=")
        name, url = name.strip().lower(), url.strip()
        if not sep or not name or not url.startswith(("http://", "https://")):
            logging.getLogger("rag-chat").warning("RUNTIME_EXTRA_RESIDENTS: ignoring %r (want name=http(s)://...)", item)
            continue
        out[name] = url
    return out


# GPU sidecars the /api/runtime/residents gateway serves that aren't rag-chat's own
# backends, e.g. the Qwen tts sidecar argus-mask calls directly. Gateway only: they
# don't become TTS engines. A name rag-chat already uses (an engine) wins.
RUNTIME_EXTRA_RESIDENTS = _parse_residents(os.getenv("RUNTIME_EXTRA_RESIDENTS", ""))

# ── Voice services ────────────────────────────────────────────────────────────
# STT (Whisper) is a single sidecar. TTS is a *registry* of one or more engines
# that all speak the XTTS sidecar HTTP contract (/synthesise, /synthesise/stream,
# /clone-voice, /clones, /speakers, /health, /analyze-clips). Each engine is
# selectable per request and one is the default — so the local XTTS GPU sidecar
# and a remote Miso host (e.g. morpheus) can coexist and be picked from the UI,
# the API, or the Wyoming/Home Assistant bridges.
WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper:8030").rstrip("/")
VOICE_TIMEOUT = _env_int("VOICE_TIMEOUT", 30)
# Reference clips /api/voice/clone and /api/voice/analyze accept per request, and the size of each
# (bytes): they are buffered in memory before forwarding, so more is a 413.
VOICE_MAX_CLIPS = _env_int("VOICE_MAX_CLIPS", 16)
VOICE_CLIP_MAX_BYTES = _env_int("VOICE_CLIP_MAX_BYTES", 16 * 1024 * 1024)


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

# argus-iris leases for the local XTTS engine (#79). With ARGUS_GPU_COORDINATOR=broker
# every XTTS synth runs inside a lease on the `xtts` service target, so XTTS no longer
# loads onto a card iris has promised to someone else. A grant on ARGUS_GPU_HOME_NODE
# calls XTTS_TTS_URL; one on another node calls XTTS_NODE_URLS[node] (`node=url,...`),
# else the broker's base_url for it. A request naming a voice clone stays home (the
# clone is stored for this XTTS). When the lease is refused or times out, a default-path
# request moves to the next engine in the chain; on the last engine,
# XTTS_LEASE_CPU_FALLBACK=1 (default) asks the local XTTS to run on its CPU, 0 answers
# 503 with the broker's reason.
XTTS_LEASE_KIND = "tts-synth"
XTTS_LEASE_VRAM_MB = _env_int("XTTS_LEASE_VRAM_MB", 2000)
XTTS_LEASE_MODEL_MB = _env_int("XTTS_LEASE_MODEL_MB", 1800)
XTTS_NODE_URLS: dict[str, str] = {}
for _item in os.getenv("XTTS_NODE_URLS", "").split(","):
    _node, _sep, _url = _item.strip().partition("=")
    if _sep and _node.strip() and _clean_url(_url):
        XTTS_NODE_URLS[_node.strip()] = _clean_url(_url)
XTTS_LEASE_CPU_FALLBACK = os.getenv("XTTS_LEASE_CPU_FALLBACK", "1").strip().lower() not in ("0", "false", "no", "off")

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

# Agent coding packs (ai-code-auditor ``GET /audit/packs/{id}``).  Default base for
# ``pack_id``-driven repairs: AI_AUDIT_PUBLIC_BASE_URL, then AUDIT_API_URL, then localhost.
AUDIT_PACK_BASE_URL = (
    os.getenv("AI_AUDIT_PUBLIC_BASE_URL")
    or os.getenv("AUDIT_API_URL")
    or "http://127.0.0.1:8765"
)
AUDIT_PACK_TIMEOUT_S = float(os.getenv("AUDIT_PACK_TIMEOUT_S", "10"))
# Extra origins a repair request's audit_api_url / audit_api_base may name, comma-separated
# ``host`` (any port) or ``host:port``; ``*`` allows any. The configured AUDIT_API_URL and
# AI_AUDIT_PUBLIC_BASE_URL origins are always allowed; link-local hosts never are.
# See app/repair/url_policy.py.
AUDIT_API_ALLOWED_HOSTS = os.getenv("AUDIT_API_ALLOWED_HOSTS", "")
# After a clean, applied auto_fix, POST /audit/packs/{id}/status {"status": "resolved"}.
# Fails soft (404/405 from older audit APIs is ignored).  Set to 0 to disable.
REPAIR_PACK_STATUS_CALLBACK = os.getenv("REPAIR_PACK_STATUS_CALLBACK", "1").lower() in ("1", "true", "yes")
# Sent as X-API-Token when the audit API gates mutating routes (AI_AUDIT_API_TOKEN there).
AUDIT_API_TOKEN = os.getenv("AI_AUDIT_API_TOKEN", "")

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

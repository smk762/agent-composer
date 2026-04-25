import logging
import os
from datetime import timezone

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
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
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

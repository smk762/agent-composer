import logging
import os
from datetime import timezone

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

MEDIA_DIR = os.getenv("MEDIA_DIR", "/data/media")

PROVIDER_ENCRYPTION_KEY = os.getenv("PROVIDER_ENCRYPTION_KEY", "")

UTC = timezone.utc


def _normalize_db_url(url: str) -> str:
    if url.startswith("sqlite:") and "+aiosqlite" not in url:
        return url.replace("sqlite:", "sqlite+aiosqlite:", 1)
    return url


CHAT_DB_URL = _normalize_db_url(os.getenv("CHAT_DB_URL", "sqlite+aiosqlite:///./chat.db"))

log.info(
    "config: ollama=%s model=%s rag=%s qdrant=%s log_level=%s dev_auth_bypass=%s timeout=%ds",
    OLLAMA_URL, CHAT_MODEL, RAG_ENABLED, QDRANT_URL, LOG_LEVEL, DEV_AUTH_BYPASS, OLLAMA_TIMEOUT,
)

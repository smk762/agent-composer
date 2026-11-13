"""rag-chat: chat gateway (Ollama + optional Qdrant RAG), history, UI and service proxies.

``main`` builds the FastAPI app from the routers in ``routers``; settings live in
``config``. The guard, voice, pipeline, repair and generation features each have
their own router plus supporting modules.
"""

__all__ = ["auth", "config", "db", "main", "metrics", "storage", "streaming"]

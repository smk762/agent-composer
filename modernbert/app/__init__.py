"""ModernBERT classifier sidecar: FastAPI app with a lazily loaded GPU/CPU model.

Submodules: ``main`` (HTTP routes), ``model`` (unloaded → cpu → gpu lifecycle and
keep-alive eviction), ``classifier`` (zero-shot / fine-tuned scoring) and
``schemas`` (request/response models).
"""

__all__ = ["classifier", "main", "model", "schemas"]

"""Symmetric encryption helpers for provider API keys stored at rest.

Uses Fernet (AES-128-CBC + HMAC-SHA256) via the ``cryptography`` library.
The encryption key is derived from PROVIDER_ENCRYPTION_KEY env var using
PBKDF2 so operators can use a human-readable passphrase.
"""

import base64
import hashlib
import logging
import os
from typing import Optional

log = logging.getLogger("rag-chat.crypto")

_FERNET = None


def _get_fernet():
    global _FERNET
    if _FERNET is not None:
        return _FERNET

    raw = os.getenv("PROVIDER_ENCRYPTION_KEY", "")
    if not raw:
        log.warning("PROVIDER_ENCRYPTION_KEY not set; provider API keys will be stored as plaintext base64")
        return None

    try:
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"rag-chat-provider-keys",
            iterations=480_000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(raw.encode()))
        _FERNET = Fernet(key)
        return _FERNET
    except ImportError:
        log.warning("cryptography package not installed; falling back to plaintext base64")
        return None


def encrypt_value(plaintext: str) -> str:
    """Encrypt a string value, returning a storable token."""
    f = _get_fernet()
    if f is None:
        return base64.urlsafe_b64encode(plaintext.encode()).decode()
    return f.encrypt(plaintext.encode()).decode()


def decrypt_value(token: str) -> Optional[str]:
    """Decrypt a previously encrypted token back to plaintext."""
    if not token:
        return None
    f = _get_fernet()
    if f is None:
        try:
            return base64.urlsafe_b64decode(token.encode()).decode()
        except Exception:
            return token
    try:
        return f.decrypt(token.encode()).decode()
    except Exception:
        log.error("Failed to decrypt provider API key (wrong PROVIDER_ENCRYPTION_KEY?)")
        return None

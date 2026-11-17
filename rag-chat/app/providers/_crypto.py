"""Symmetric encryption helpers for provider API keys stored at rest.

Uses Fernet (AES-128-CBC + HMAC-SHA256) via the ``cryptography`` library.
The encryption key is derived from PROVIDER_ENCRYPTION_KEY env var using
PBKDF2 so operators can use a human-readable passphrase.

Fails closed: with no PROVIDER_ENCRYPTION_KEY a key is refused rather than
stored as reversible base64, unless ALLOW_PLAINTEXT_PROVIDER_KEYS=1 opts in
(dev only), in which case it is stored with a ``plain:`` prefix so it can be
found and re-saved once a key is set. A key that is set but can't be used
(``cryptography`` missing) is an error, never a silent downgrade.

Reading still accepts values written before this (un-prefixed base64), so
existing deployments keep working; re-save those keys to encrypt them.
"""

import base64
import binascii
import logging
import os
from typing import Optional

log = logging.getLogger("rag-chat.crypto")

PLAIN_PREFIX = "plain:"
# Every Fernet token starts with version byte 0x80, i.e. "gAAAAA" in urlsafe base64. The old
# format was base64 of ASCII text, which can never start with "g" (a byte >= 0x80).
_FERNET_PREFIX = "gAAAAA"

_FERNET = None
_FERNET_FOR: Optional[str] = None
_warned: set[str] = set()


class ProviderKeyEncryptionError(RuntimeError):
    """A provider key can't be stored (or read) encrypted, and plaintext isn't allowed."""


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(msg)


def _plaintext_allowed() -> bool:
    return os.getenv("ALLOW_PLAINTEXT_PROVIDER_KEYS", "").strip().lower() in ("1", "true", "yes")


def _get_fernet():
    """The Fernet for PROVIDER_ENCRYPTION_KEY, cached; None when no key is set.

    A key that is set but unusable raises: the operator asked for encryption."""
    global _FERNET, _FERNET_FOR
    raw = os.getenv("PROVIDER_ENCRYPTION_KEY", "")
    if not raw:
        return None
    if _FERNET is not None and _FERNET_FOR == raw:
        return _FERNET

    try:
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except ImportError as exc:
        raise ProviderKeyEncryptionError(
            "PROVIDER_ENCRYPTION_KEY is set but the cryptography package is not installed; "
            "refusing to handle provider keys unencrypted"
        ) from exc

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"rag-chat-provider-keys",
        iterations=480_000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(raw.encode()))
    _FERNET, _FERNET_FOR = Fernet(key), raw
    return _FERNET


def check_provider_encryption() -> None:
    """Startup check: raise if a configured key can't be used; log loudly if keys can't be saved."""
    if _get_fernet() is not None:
        return
    if _plaintext_allowed():
        log.warning(
            "ALLOW_PLAINTEXT_PROVIDER_KEYS=1 and no PROVIDER_ENCRYPTION_KEY: provider API keys are "
            "stored as reversible base64. Development only."
        )
    else:
        log.error(
            "PROVIDER_ENCRYPTION_KEY is not set: saving a provider API key will be refused. Set it "
            "(or ALLOW_PLAINTEXT_PROVIDER_KEYS=1 for development). Keys already stored stay readable."
        )


def encrypt_value(plaintext: str) -> str:
    """Encrypt a string value, returning a storable token.

    Raises ProviderKeyEncryptionError when it can't be encrypted and plaintext isn't allowed."""
    f = _get_fernet()
    if f is not None:
        return f.encrypt(plaintext.encode()).decode()
    if not _plaintext_allowed():
        raise ProviderKeyEncryptionError(
            "PROVIDER_ENCRYPTION_KEY is not set, so provider API keys can't be stored encrypted. "
            "Set it (or ALLOW_PLAINTEXT_PROVIDER_KEYS=1 for development)."
        )
    _warn_once("plain-write", "storing a provider API key unencrypted (ALLOW_PLAINTEXT_PROVIDER_KEYS=1)")
    return PLAIN_PREFIX + base64.urlsafe_b64encode(plaintext.encode()).decode()


def _b64_text(value: str) -> str:
    try:
        return base64.urlsafe_b64decode(value.encode()).decode()
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return value


def decrypt_value(token: str) -> Optional[str]:
    """Decrypt a previously encrypted token back to plaintext."""
    if not token:
        return None
    if token.startswith(PLAIN_PREFIX):
        _warn_once("plain-read", "a provider API key is stored unencrypted; re-save it once "
                   "PROVIDER_ENCRYPTION_KEY is set")
        return _b64_text(token[len(PLAIN_PREFIX):])
    if token.startswith(_FERNET_PREFIX):
        f = _get_fernet()
        if f is None:
            log.error("A provider API key is encrypted but PROVIDER_ENCRYPTION_KEY is not set")
            return None
        try:
            return f.decrypt(token.encode()).decode()
        except Exception:
            log.error("Failed to decrypt provider API key (wrong PROVIDER_ENCRYPTION_KEY?)")
            return None
    # Written before storage failed closed: un-prefixed base64. Still read, so existing
    # deployments keep working; re-saving the key encrypts it.
    _warn_once("legacy-read", "a provider API key is stored in the old unencrypted format; "
               "re-save it to encrypt it")
    return _b64_text(token)

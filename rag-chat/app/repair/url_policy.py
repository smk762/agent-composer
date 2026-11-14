"""Which audit API origins a repair request may point rag-chat at (issue #28).

``POST /api/repair/run`` accepts two client-supplied base URLs, ``audit_api_url`` and
``audit_api_base``, and the server sends requests to them: the generated patch for
validation, the pack fetch, git ops, and the resolved callback that carries
``AI_AUDIT_API_TOKEN``. A URL is accepted only when:

* it is ``http``/``https`` with a host, and
* its host is not link-local (169.254.0.0/16, fe80::/10, where cloud metadata services
  live), whatever the allowlist says, and
* its ``host:port`` equals a configured audit URL's (``AUDIT_API_URL``,
  ``AI_AUDIT_PUBLIC_BASE_URL``), or matches an ``AUDIT_API_ALLOWED_HOSTS`` entry
  (``host`` = any port, ``host:port`` = that port), or the list contains ``*``.

Configured origins match on host *and* port: another port on the audit host is another
service, so it is not implied. Hosts are compared as written; names are not resolved,
so this is not a defence against DNS rebinding.
"""

from __future__ import annotations

import ipaddress
import logging
from functools import lru_cache
from typing import Iterable, Optional
from urllib.parse import urlsplit

log = logging.getLogger("rag-chat.repair")

_DEFAULT_PORTS = {"http": 80, "https": 443}

Origin = tuple[str, int]  # (lowercased host, port)


def _origin(url: str) -> Optional[Origin]:
    """(host, port) of an http(s) URL, or None when it has no usable scheme, host or port."""
    try:
        parts = urlsplit((url or "").strip())
        port = parts.port
    except ValueError:
        return None
    if parts.scheme not in _DEFAULT_PORTS or not parts.hostname:
        return None
    return parts.hostname.lower(), port or _DEFAULT_PORTS[parts.scheme]


@lru_cache(maxsize=8)
def parse_allowlist(raw: str) -> tuple[frozenset, bool]:
    """``AUDIT_API_ALLOWED_HOSTS`` -> ({(host, port or None)}, wildcard). Bad entries are logged once and skipped."""
    entries: set[tuple[str, Optional[int]]] = set()
    wildcard = False
    for item in (raw or "").split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item == "*":
            wildcard = True
            continue
        try:
            parts = urlsplit("//" + item)
            port = parts.port
        except ValueError:
            parts, port = None, None
        if parts is None or not parts.hostname or parts.path or parts.username:
            log.warning("AUDIT_API_ALLOWED_HOSTS: ignoring %r (expected host or host:port)", item)
            continue
        entries.add((parts.hostname, port))
    return frozenset(entries), wildcard


def _is_link_local(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return ip.is_link_local or bool(mapped and mapped.is_link_local)


def check_audit_url(url: str, field: str, *, allowlist: str, configured: Iterable[str]) -> Optional[str]:
    """None when ``url`` may be used, else the reason it may not (for a 422)."""
    origin = _origin(url)
    if origin is None:
        return f"{field} must be an http(s) URL with a host"
    host, port = origin
    if _is_link_local(host):
        return f"{field}: link-local hosts ({host}) are never allowed"
    entries, wildcard = parse_allowlist(allowlist)
    if wildcard or (host, port) in entries or (host, None) in entries:
        return None
    if any(_origin(c) == origin for c in configured if c):
        return None
    return (
        f"{field} host {host}:{port} is not allowed: use the configured audit API or add it to "
        f"AUDIT_API_ALLOWED_HOSTS on rag-chat"
    )

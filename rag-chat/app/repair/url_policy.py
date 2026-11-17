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
service, so it is not implied. Hosts are compared as written. IP literals in any form the
socket layer accepts (``2852039166``, ``0xA9FEA9FE``, ``169.254.43518``, a trailing dot,
IPv4 inside IPv6) are normalised before the link-local check.

A URL accepted through ``AUDIT_API_ALLOWED_HOSTS`` (not a configured origin) is also
resolved (``check_resolved``): it is refused when any address it resolves to is
link-local, or when it does not resolve. The connection is not pinned to the checked
address, so this is still not a defence against DNS rebinding.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
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


_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def _ip_literal(host: str) -> Optional[ipaddress._BaseAddress]:
    """The address a host string denotes as a literal, in any form ``inet_aton`` accepts, else None."""
    host = host.strip().rstrip(".").strip("[]")
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    try:  # decimal, hex, octal and short dotted forms: what getaddrinfo would connect to
        return ipaddress.IPv4Address(socket.inet_aton(host))
    except (OSError, ValueError):
        return None


def _embedded_v4(ip: ipaddress._BaseAddress) -> Optional[ipaddress.IPv4Address]:
    """The IPv4 address an IPv6 address carries (mapped, 6to4, IPv4-compatible, NAT64), if any."""
    if not isinstance(ip, ipaddress.IPv6Address):
        return None
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip.sixtofour is not None:
        return ip.sixtofour
    packed = int(ip)
    if ip in _NAT64 or (packed >> 32 == 0 and packed > 0xFFFF):  # 64:ff9b::/96, ::a.b.c.d
        return ipaddress.IPv4Address(packed & 0xFFFFFFFF)
    return None


def _address_is_link_local(ip: ipaddress._BaseAddress) -> bool:
    inner = _embedded_v4(ip)
    return ip.is_link_local or bool(inner and inner.is_link_local)


def _is_link_local(host: str) -> bool:
    ip = _ip_literal(host)
    return ip is not None and _address_is_link_local(ip)


async def _resolve(host: str, port: int) -> list[str]:
    """Every address ``host`` resolves to (patched in tests)."""
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


async def check_resolved(url: str, field: str, *, configured: Iterable[str]) -> Optional[str]:
    """None when ``url``'s host does not resolve to a link-local address, else why (for a 422).

    Run after ``check_audit_url`` accepted ``url``. Configured origins are the operator's own
    and are not resolved; any other name must resolve, and to no link-local address.
    """
    origin = _origin(url)
    if origin is None:
        return f"{field} must be an http(s) URL with a host"
    if any(_origin(c) == origin for c in configured if c):
        return None
    host, port = origin
    try:
        addresses = await _resolve(host, port)
    except (OSError, UnicodeError) as exc:
        return f"{field}: cannot resolve {host} ({exc})"
    if not addresses:
        return f"{field}: {host} resolves to no address"
    for addr in addresses:
        ip = _ip_literal(addr.split("%", 1)[0])
        if ip is None or _address_is_link_local(ip):
            return f"{field}: {host} resolves to a link-local address ({addr}), which is never allowed"
    return None


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

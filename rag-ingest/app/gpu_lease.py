"""argus-iris GPU leases around Ollama calls (#48).

``async with ollama_lease(model, kind=..., priority=...) as base_url:`` holds an
``ollama:<model>`` lease from the suite's GPU broker for its body and yields the Ollama
base URL to call: the local ``OLLAMA_URL`` unless the broker placed the work on another
node (spilling to jarrah), then that node's URL.

Off by default. ``ARGUS_GPU_COORDINATOR=broker`` turns it on, with argus-cortex's other
``ARGUS_GPU_*`` variables (``BROKER_URL``, ``BROKER_TOKEN``, ``TENANT``,
``TENANT_MANIFEST``, ``LEASE_TIMEOUT_S``). Here ``ARGUS_GPU_BROKER_FALLBACK`` defaults
to ``none``, not cortex's ``lease``: a host flock would serialise every Ollama call.

* **Fails open.** An unreachable broker is cortex's fallback (no gating, local Ollama).
  A broker that answers with a 4xx (a bad token, an incompatible version) is logged and
  the call goes to local Ollama too: a config slip must not take chat and SingGuard down.
* **Refusals are not OOMs.** A refused or timed-out lease raises
  :class:`OllamaLeaseUnavailable`, which is not a ``RuntimeError``, and the lease is taken
  outside ``run_with_oom_wait``, so the OOM loop never polls on it. Routes answer 503
  with ``Retry-After``.
* **Pinned models** (``OLLAMA_LEASE_PINNED_MODELS``) carry a node constraint
  (``ARGUS_GPU_HOME_NODE``): SingGuard exists only on dragon, and the text embedder must
  match the vectors ingest wrote.
* **Footprint**: ``min_vram_mb`` is the whole footprint (``OLLAMA_LEASE_VRAM_MB``) and
  ``model_mb`` its weights (``OLLAMA_LEASE_MODEL_MB``), so iris asks only for the working
  set when the model is already loaded. ``gpu-tenant.yaml``'s kinds carry the same figures.
* **Priority**: each call site passes its class; a request's ``X-Argus-Priority`` header
  overrides it (:class:`PriorityMiddleware`).

rag-chat and rag-ingest each carry a copy of this module (like ``inference_retry.py``);
keep them identical.
"""

from __future__ import annotations

import contextlib
import contextvars
import ipaddress
import logging
import os
import socket
from typing import AsyncIterator, Optional
from urllib.parse import urlsplit

log = logging.getLogger("gpu-lease")

PRIORITIES = ("realtime", "interactive", "batch", "background")
LEASE_RETRY_AFTER_S = 30

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
HOME_NODE = os.getenv("ARGUS_GPU_HOME_NODE", "").strip()
DEFAULT_VRAM_MB = 6000


def _pairs(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in raw.split(","):
        key, sep, value = item.strip().partition("=")
        if sep and key.strip() and value.strip():
            out[key.strip()] = value.strip()
    return out


def _mb_table(env: str, defaults: dict[str, int]) -> dict[str, int]:
    """*defaults* overridden by ``model=MiB,…`` pairs from *env*."""
    table = dict(defaults)
    for model, mb in _pairs(os.getenv(env, "")).items():
        try:
            table[model] = max(0, int(mb))
        except ValueError:
            log.warning("%s: ignoring %s=%r", env, model, mb)
    return table


# Whole footprint per model: weights plus working set (KV cache, activations).
_DEFAULT_VRAM_MB = {"nomic-embed-text": 600, "hf.co/inclusionAI/SingGuard-4b-GGUF": 6000}
# The part of it that is weights and stays loaded after the call (#50): Ollama /api/ps
# size_vram at the default context, measured 2026-11-16. iris counts a lease as
# working + model and skips the model part when the model is already resident, so a
# check against a loaded SingGuard asks only for its working set. A larger num_ctx grows
# the KV cache, which is working set: raise OLLAMA_LEASE_VRAM_MB, not this.
_DEFAULT_MODEL_MB = {"nomic-embed-text": 320, "hf.co/inclusionAI/SingGuard-4b-GGUF": 3900, "qwen3:8b": 5300}


# "jarrah=http://192.168.1.109:11434": the URL to call on a node, when the broker's
# base_url for it doesn't resolve from inside the container.
NODE_URLS = {node: url.rstrip("/") for node, url in _pairs(os.getenv("OLLAMA_NODE_URLS", "")).items()}
VRAM_MB = _mb_table("OLLAMA_LEASE_VRAM_MB", _DEFAULT_VRAM_MB)
MODEL_MB = _mb_table("OLLAMA_LEASE_MODEL_MB", _DEFAULT_MODEL_MB)
try:
    DEFAULT_VRAM_MB = max(0, int(os.getenv("OLLAMA_LEASE_DEFAULT_VRAM_MB", str(DEFAULT_VRAM_MB))))
except ValueError:
    pass



def _pinned_models() -> set[str]:
    """``OLLAMA_LEASE_PINNED_MODELS``; unset or empty means the guard and text-embed models, ``none`` none."""
    raw = os.getenv("OLLAMA_LEASE_PINNED_MODELS", "").strip()
    if raw.lower() == "none":
        return set()
    if not raw:
        raw = ",".join((
            os.getenv("GUARD_MODEL", "hf.co/inclusionAI/SingGuard-4b-GGUF:Q4_K_M"),
            os.getenv("EMBED_MODEL", "nomic-embed-text"),
        ))
    return {m.strip() for m in raw.split(",") if m.strip()}


PINNED_MODELS = _pinned_models()

_priority: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("argus_priority", default=None)


class OllamaLeaseUnavailable(Exception):
    """The broker refused the lease, or didn't grant it in time. Not an OOM: don't retry it as one."""

    def __init__(self, detail: str, *, retry_after: int | None = LEASE_RETRY_AFTER_S) -> None:
        super().__init__(detail)
        self.detail = detail
        self.retry_after = retry_after


class PriorityMiddleware:
    """Puts a valid ``X-Argus-Priority`` header in a context variable for the request's leases."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        value = None
        for name, raw in scope.get("headers", []):
            if name == b"x-argus-priority":
                value = raw.decode("latin-1").strip().lower()
                break
        token = _priority.set(value if value in PRIORITIES else None)
        try:
            await self.app(scope, receive, send)
        finally:
            _priority.reset(token)


_coordinator = None


def coordinator():
    """The process's coordinator, built once from the environment (``none`` when unset)."""
    global _coordinator
    if _coordinator is None:
        from argus_cortex.gpu import build_coordinator

        timeout = os.getenv("ARGUS_GPU_LEASE_TIMEOUT_S", "").strip()
        _coordinator = build_coordinator(
            os.getenv("ARGUS_GPU_COORDINATOR", "none"),
            timeout_s=float(timeout) if timeout else None,
            broker_url=os.getenv("ARGUS_GPU_BROKER_URL", "").strip() or None,
            broker_token=os.getenv("ARGUS_GPU_BROKER_TOKEN", "").strip() or None,
            tenant=os.getenv("ARGUS_GPU_TENANT", "").strip() or None,
            manifest_path=os.getenv("ARGUS_GPU_TENANT_MANIFEST", "").strip() or None,
            fallback=os.getenv("ARGUS_GPU_BROKER_FALLBACK", "none"),
        )
        if _coordinator.name == "broker" and PINNED_MODELS and not HOME_NODE:
            log.warning("ARGUS_GPU_HOME_NODE is unset: pinned models %s may be placed on any node",
                        sorted(PINNED_MODELS))
        log.info("GPU leases for Ollama: %s coordinator", _coordinator.name)
    return _coordinator


def _lookup(table: dict[str, int], model: str) -> int | None:
    """Exact model name, then the name without its tag."""
    if model in table:
        return table[model]
    return table.get(model.rsplit(":", 1)[0])


def vram_mb(model: str) -> int:
    """Declared footprint: exact model name, then the name without its tag, then the default."""
    mb = _lookup(VRAM_MB, model)
    return DEFAULT_VRAM_MB if mb is None else mb


def model_mb(model: str) -> int | None:
    """The weights part of :func:`vram_mb` (never more than it); ``None`` when unknown,
    which the broker reads as "all of it is working set"."""
    mb = _lookup(MODEL_MB, model)
    return None if mb is None else min(mb, vram_mb(model))


def _link_local(host: str) -> bool:
    """Is ``host`` a link-local address literal (cloud metadata lives there), in any form the
    socket layer accepts: decimal or hex IPv4, a trailing dot, IPv4 inside IPv6."""
    host = host.strip().rstrip(".").strip("[]").split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            ip = ipaddress.IPv4Address(socket.inet_aton(host))
        except (OSError, ValueError):
            return False
    inner = None
    if isinstance(ip, ipaddress.IPv6Address):
        packed = int(ip)
        inner = ip.ipv4_mapped or ip.sixtofour or (
            ipaddress.IPv4Address(packed & 0xFFFFFFFF)
            if ip in _NAT64 or (packed >> 32 == 0 and packed > 0xFFFF) else None)
    return ip.is_link_local or bool(inner and inner.is_link_local)


_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def vet_node_url(url: str) -> str:
    """A broker-supplied node URL, normalised, or :class:`OllamaLeaseUnavailable`.

    The broker names the node and its base URL, and every Ollama call of the lease goes
    there. It is accepted only as a bare ``http(s)://host[:port]`` origin (no credentials,
    path, query or fragment) whose host is not link-local, so a compromised or misconfigured
    broker can't point the service at cloud metadata or smuggle a path into the call.
    ``OLLAMA_NODE_URLS`` (operator config) is not vetted: it is used as written.
    """
    try:
        parts = urlsplit((url or "").strip())
        parts.port  # noqa: B018 — raises ValueError on a bad port
    except ValueError:
        parts = None
    if (parts is None or parts.scheme not in ("http", "https") or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.path not in ("", "/") or parts.query or parts.fragment):
        raise OllamaLeaseUnavailable(f"GPU broker gave an unusable node URL {url!r}: expected http(s)://host[:port]")
    if _link_local(parts.hostname):
        raise OllamaLeaseUnavailable(f"GPU broker gave a link-local node URL {url!r}, which is never allowed")
    return f"{parts.scheme}://{parts.netloc}"


def url_for(handle) -> str:
    """Where to send the call: local Ollama unless the broker placed it on another node.

    A node in ``OLLAMA_NODE_URLS`` uses that URL; any other node's broker-supplied
    ``base_url`` must pass :func:`vet_node_url` (else :class:`OllamaLeaseUnavailable`, a 503).
    """
    if getattr(handle, "via", "local") != "broker":
        return OLLAMA_URL
    node = handle.node
    if node and node in NODE_URLS:
        return NODE_URLS[node]
    if not node or node == HOME_NODE or not handle.base_url:
        return OLLAMA_URL
    return vet_node_url(handle.base_url)


@contextlib.asynccontextmanager
async def ollama_lease(model: str, *, kind: str, priority: str, caller: str = "") -> AsyncIterator[str]:
    """Hold an ``ollama:<model>`` lease for the body; yields the base URL to call."""
    coord = coordinator()
    if coord.name == "none":
        yield OLLAMA_URL
        return

    from argus_cortex.gpu import GpuLeaseRefused, GpuLeaseTimeout
    from argus_cortex.gpu_broker import BrokerError
    from argus_cortex.gpu_wire import GpuConstraints

    constraints = GpuConstraints(node=HOME_NODE) if HOME_NODE and model in PINNED_MODELS else None
    cm = coord.alease(
        caller=caller or kind,
        min_vram_mb=vram_mb(model),
        model_mb=model_mb(model),
        priority=_priority.get() or priority,
        target=f"ollama:{model}",
        kind=kind,
        preemptible=False,
        constraints=constraints,
    )
    try:
        handle = await cm.__aenter__()
    except GpuLeaseRefused as exc:
        raise OllamaLeaseUnavailable(f"GPU broker refused {kind} on ollama:{model}: {exc}", retry_after=None) from exc
    except GpuLeaseTimeout as exc:
        raise OllamaLeaseUnavailable(f"no GPU lease for {kind} on ollama:{model}: {exc}") from exc
    except BrokerError as exc:
        log.error("GPU broker error for %s on ollama:%s (%s); calling local Ollama unleased", kind, model, exc)
        yield OLLAMA_URL
        return

    # Everything after the grant runs inside the try, so the lease is released even if
    # resolving the node's URL or logging the placement fails.
    try:
        url = url_for(handle)
        if url != OLLAMA_URL:
            log.info("lease %s: %s on ollama:%s placed on %s (%s)", handle.lease_id, kind, model, handle.node, url)
        yield url
    except BaseException as exc:
        if not await cm.__aexit__(type(exc), exc, exc.__traceback__):
            raise
    else:
        await cm.__aexit__(None, None, None)

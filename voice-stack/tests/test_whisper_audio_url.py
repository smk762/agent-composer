"""SSRF guard on POST /transcriptions audio_url (whisper_server)."""

import asyncio
import ipaddress
import socket

import httpx
import pytest
from fastapi.testclient import TestClient

import whisper_server as ws


@pytest.fixture
def resolver(monkeypatch):
    mapping = {
        "public.example": ["93.184.216.34"],
        "internal.example": ["10.0.0.5"],
        "mixed.example": ["93.184.216.34", "127.0.0.1"],
        "minio": ["172.18.0.4"],
    }

    async def fake_resolve(host, port):
        try:
            return [str(ipaddress.ip_address(host))]
        except ValueError:
            pass
        if host not in mapping:
            raise socket.gaierror("no such host")
        return mapping[host]

    monkeypatch.setattr(ws, "_resolve_host", fake_resolve)
    return mapping


@pytest.mark.parametrize(
    "addr,public",
    [
        ("93.184.216.34", True),
        ("2606:2800:220:1:248:1893:25c8:1946", True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("172.16.0.1", False),
        ("192.168.1.198", False),
        ("169.254.169.254", False),
        ("100.64.0.1", False),
        ("0.0.0.0", False),
        ("224.0.0.1", False),
        ("240.0.0.1", False),
        ("::1", False),
        ("::", False),
        ("fe80::1%eth0", False),
        ("fc00::1", False),
        ("::ffff:127.0.0.1", False),
        ("::ffff:169.254.169.254", False),
        ("ff0e::1", False),
        ("not-an-ip", False),
    ],
)
def test_ip_is_public(addr, public):
    assert ws._ip_is_public(addr) is public


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://public.example/a.wav",
        "gopher://public.example/",
        "http://127.0.0.1:6333/collections",
        "http://[::1]/a.wav",
        "http://[::ffff:127.0.0.1]/a.wav",
        "http://169.254.169.254/latest/meta-data/",
        "http://internal.example/a.wav",
        "http://mixed.example/a.wav",
        "http://nxdomain.example/a.wav",
        "http://user:pw@public.example/a.wav",
        "http://public.example:99999/a.wav",
        "http:///nohost",
    ],
)
def test_vet_rejects(url, resolver):
    with pytest.raises(ws.AudioURLError) as ei:
        asyncio.run(ws._vet_audio_url(url, frozenset()))
    assert ei.value.status == 400
    assert ei.value.detail == ws._AUDIO_URL_REJECTED


def test_vet_pins_public_host(resolver):
    url, headers, ext = asyncio.run(ws._vet_audio_url("https://public.example:8443/x/a.wav?sig=1", frozenset()))
    assert url == "https://93.184.216.34:8443/x/a.wav?sig=1"
    assert headers == {"Host": "public.example:8443"}
    assert ext == {"sni_hostname": "public.example"}

    url, headers, ext = asyncio.run(ws._vet_audio_url("http://public.example/a.wav", frozenset()))
    assert url == "http://93.184.216.34/a.wav"
    assert headers == {"Host": "public.example"}
    assert ext == {}


def test_vet_allowlist_bypasses_private_check(resolver):
    allowed = ws._parse_allowed_hosts(" minio:9000 , Internal.Example ")
    assert asyncio.run(ws._vet_audio_url("http://minio:9000/voice/a.wav", allowed))[0] == "http://minio:9000/voice/a.wav"
    assert asyncio.run(ws._vet_audio_url("http://internal.example/a.wav", allowed))[0] == "http://internal.example/a.wav"
    # host:port entries only cover that port
    with pytest.raises(ws.AudioURLError):
        asyncio.run(ws._vet_audio_url("http://minio:6333/", allowed))


# ── endpoint ─────────────────────────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch, resolver):
    monkeypatch.setattr(ws, "STT_AUDIO_URL_ALLOWED_HOSTS", frozenset())

    async def fake_run(tmp_path, language, endpoint):
        with open(tmp_path, "rb") as f:
            data = f.read()
        return {"text": data.decode(), "language": "en", "duration_seconds": 1.0, "confidence": 1.0}

    monkeypatch.setattr(ws, "_run_transcription", fake_run)
    return TestClient(ws.app)


def _mock_upstream(monkeypatch, handler):
    seen = []
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        assert kwargs.get("follow_redirects") is False

        def wrapped(request):
            seen.append(request)
            return handler(request)

        kwargs["transport"] = httpx.MockTransport(wrapped)
        return real(*args, **kwargs)

    monkeypatch.setattr(ws.httpx, "AsyncClient", factory)
    return seen


def test_endpoint_fetches_pinned(client, monkeypatch):
    seen = _mock_upstream(monkeypatch, lambda r: httpx.Response(200, content=b"hello"))
    r = client.post("/transcriptions", json={"audio_url": "http://public.example/a.wav"})
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "hello"
    assert str(seen[0].url) == "http://93.184.216.34/a.wav"
    assert seen[0].headers["host"] == "public.example"


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434/api/tags", "http://169.254.169.254/", "file:///etc/passwd"])
def test_endpoint_rejects_internal(client, monkeypatch, url):
    seen = _mock_upstream(monkeypatch, lambda r: httpx.Response(200, content=b"secret"))
    r = client.post("/transcriptions", json={"audio_url": url})
    assert r.status_code == 400
    assert r.json() == {"detail": ws._AUDIO_URL_REJECTED}
    assert seen == []


@pytest.mark.parametrize(
    "handler",
    [
        lambda r: httpx.Response(302, headers={"Location": "http://127.0.0.1/"}),
        lambda r: httpx.Response(404),
        lambda r: httpx.Response(500),
    ],
)
def test_endpoint_upstream_failures_are_generic(client, monkeypatch, handler):
    seen = _mock_upstream(monkeypatch, handler)
    r = client.post("/transcriptions", json={"audio_url": "http://public.example/a.wav"})
    assert r.status_code == 502
    assert r.json() == {"detail": ws._AUDIO_URL_FAILED}
    assert len(seen) == 1  # the redirect was not followed


def test_endpoint_connect_error_is_generic(client, monkeypatch):
    def boom(request):
        raise httpx.ConnectError("[Errno 111] Connection refused to 10.0.0.9:22")

    _mock_upstream(monkeypatch, boom)
    r = client.post("/transcriptions", json={"audio_url": "http://public.example/a.wav"})
    assert r.status_code == 502
    assert r.json() == {"detail": ws._AUDIO_URL_FAILED}

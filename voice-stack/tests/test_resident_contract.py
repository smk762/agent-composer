"""argus-iris resident contract on the voice sidecars: /health ``gpu`` block,
POST /admin/{unload,evict,wake}, and ARGUS_ADMIN_TOKEN gating.

whisper runs as-is (no model is loaded); tts runs with torch stubbed, as in
test_tts_server.py. xtts shares tts's helper and is covered there by torch-gated
tests only.
"""

import os
import sys
import types

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.device = lambda d: d
    _torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    sys.modules["torch"] = _torch

import tts_server  # noqa: E402
import whisper_server as ws  # noqa: E402

GPU_KEYS = {"loaded", "model_state", "idle_s", "free_vram_mb"}


@pytest.fixture
def whisper(monkeypatch):
    async def decoder_ok():
        return {"ok": True}

    monkeypatch.setattr(ws, "_decoder_status", decoder_ok)
    monkeypatch.delenv("ARGUS_ADMIN_TOKEN", raising=False)
    monkeypatch.setattr(ws, "_model", None)
    monkeypatch.setattr(ws, "_last_used", 0.0)
    monkeypatch.setattr(ws, "_active", 0)
    return TestClient(ws.app)


def test_whisper_health_has_gpu_block(whisper):
    gpu = whisper.get("/health").json()["gpu"]
    assert GPU_KEYS <= set(gpu)
    assert gpu["loaded"] is False and gpu["model_state"] == "unloaded" and gpu["idle_s"] is None


def test_whisper_unload_frees_a_loaded_model(whisper, monkeypatch):
    monkeypatch.setattr(ws, "_model", object())
    monkeypatch.setattr(ws, "_last_used", 1.0)
    body = whisper.post("/admin/unload").json()
    assert body["unloaded"] is True and body["model_state"] == "unloaded"
    assert ws._model is None
    assert whisper.post("/admin/unload").json()["unloaded"] is False


def test_whisper_evict_matches_unload(whisper, monkeypatch):
    monkeypatch.setattr(ws, "_model", object())
    body = whisper.post("/admin/evict").json()
    assert body["status"] == "evicted" and body["unloaded"] is True
    assert ws._model is None


@pytest.mark.parametrize("route", ["/admin/unload", "/admin/evict", "/admin/wake"])
def test_whisper_admin_token(whisper, monkeypatch, route):
    monkeypatch.setenv("ARGUS_ADMIN_TOKEN", "s3cret")

    async def no_load():
        return None

    monkeypatch.setattr(ws, "_ensure_model", no_load)
    assert whisper.post(route).status_code == 401
    assert whisper.post(route, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert whisper.post(route, headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_whisper_health_stays_open_with_token(whisper, monkeypatch):
    monkeypatch.setenv("ARGUS_ADMIN_TOKEN", "s3cret")
    assert whisper.get("/health").status_code == 200


@pytest.fixture
def tts(monkeypatch):
    monkeypatch.delenv("ARGUS_ADMIN_TOKEN", raising=False)
    mgr = tts_server.TTSModelManager()
    monkeypatch.setattr(tts_server, "_mgr", mgr)
    return TestClient(tts_server.app), mgr


def test_tts_health_and_unload(tts, monkeypatch):
    client, mgr = tts
    gpu = client.get("/health").json()["gpu"]
    assert GPU_KEYS <= set(gpu) and gpu["loaded"] is False

    mgr._state = tts_server.TTSState.CPU
    mgr._last_used = 1.0
    body = client.post("/admin/unload").json()
    assert body["unloaded"] is True
    assert body["model_state"] == "unloaded" == body["gpu"]["model_state"]
    assert body["gpu"]["idle_s"] > 0


def test_tts_admin_token(tts, monkeypatch):
    client, _ = tts
    monkeypatch.setenv("ARGUS_ADMIN_TOKEN", "s3cret")
    assert client.post("/admin/unload").status_code == 401
    assert client.post("/admin/unload", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_whisper_refuses_unload_mid_transcription(whisper, monkeypatch):
    model = object()
    monkeypatch.setattr(ws, "_model", model)
    monkeypatch.setattr(ws, "_active", 1)
    for route in ("/admin/unload", "/admin/evict"):
        r = whisper.post(route)
        assert r.status_code == 409 and r.json()["status"] == "busy"
    assert ws._model is model
    health = whisper.get("/health").json()
    assert health["busy"] is True and health["idle_s"] is None


def test_top_level_state_matches_voluptas_shape(whisper, tts):
    """voluptas's argus-http residents put model_state, busy and idle_s at the
    top of /health; ours carry them there too, beside the lens-shaped gpu block."""
    for client in (whisper, tts[0]):
        body = client.get("/health").json()
        assert {"model_state", "busy", "idle_s", "gpu"} <= set(body)
        assert body["busy"] is False

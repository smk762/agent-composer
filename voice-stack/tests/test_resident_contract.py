"""argus-iris resident contract on the voice sidecars: /health ``gpu`` block,
POST /admin/{unload,evict,wake}, and ARGUS_ADMIN_TOKEN gating.

whisper runs as-is (no model is loaded). xtts's admin gating and lifecycle are
covered in test_xtts_clones.py and test_xtts_server.py.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import whisper_server as ws  # noqa: E402

GPU_KEYS = {"loaded", "model_state", "idle_s", "free_vram_mb"}


@pytest.fixture
def whisper(monkeypatch):
    async def decoder_ok():
        return {"ok": True}

    monkeypatch.setattr(ws, "_decoder_status", decoder_ok)
    monkeypatch.delenv("ARGUS_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("ARGUS_ADMIN_INSECURE", "1")
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


@pytest.mark.parametrize("route", ["/admin/unload", "/admin/evict", "/admin/wake"])
@pytest.mark.parametrize("token", [None, "", "  "])
def test_whisper_admin_fails_closed_without_token(whisper, monkeypatch, route, token):
    """ea4a750415: no ARGUS_ADMIN_TOKEN means 503, not an open admin route."""
    monkeypatch.delenv("ARGUS_ADMIN_INSECURE", raising=False)
    if token is None:
        monkeypatch.delenv("ARGUS_ADMIN_TOKEN", raising=False)
    else:
        monkeypatch.setenv("ARGUS_ADMIN_TOKEN", token)
    model = object()
    monkeypatch.setattr(ws, "_model", model)

    async def no_load():
        raise AssertionError("admin route ran")

    monkeypatch.setattr(ws, "_ensure_model", no_load)
    for headers in ({}, {"Authorization": "Bearer "}, {"Authorization": "Bearer anything"}):
        r = whisper.post(route, headers=headers)
        assert r.status_code == 503, (route, headers, r.status_code)
    assert ws._model is model  # nothing ran


def test_whisper_admin_insecure_opt_in_does_not_override_a_token(whisper, monkeypatch):
    monkeypatch.setenv("ARGUS_ADMIN_TOKEN", "s3cret")  # ARGUS_ADMIN_INSECURE=1 from the fixture
    assert whisper.post("/admin/unload").status_code == 401


def test_whisper_health_stays_open_with_token(whisper, monkeypatch):
    monkeypatch.setenv("ARGUS_ADMIN_TOKEN", "s3cret")
    assert whisper.get("/health").status_code == 200


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


def test_top_level_state_matches_voluptas_shape(whisper):
    """voluptas's argus-http residents put model_state, busy and idle_s at the
    top of /health; ours carry them there too, beside the lens-shaped gpu block."""
    body = whisper.get("/health").json()
    assert {"model_state", "busy", "idle_s", "gpu"} <= set(body)
    assert body["busy"] is False

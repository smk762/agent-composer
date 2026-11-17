"""whisper_server error responses: only a real OOM is a 503 (9dbfab3b6d), and
no exception text or GPU state reaches the client (f2a620b872).

``_load_model`` and ``_transcribe_sync`` are replaced: no model or GPU needed.
"""

import asyncio
import json

import pytest

import whisper_server as ws

SECRET = "/tmp/secret-path/tmpabc.wav"


@pytest.fixture
def run_with(monkeypatch):
    monkeypatch.setattr(ws, "_model", None)
    monkeypatch.setattr(ws, "_active", 0)
    monkeypatch.setattr(ws, "_evict_task", None)
    monkeypatch.setattr(ws, "_lock", asyncio.Lock())
    monkeypatch.setattr(ws, "STT_KEEP_ALIVE_GPU", -1)
    monkeypatch.setattr(ws, "_load_model", lambda: object())
    monkeypatch.setattr(ws, "_gpu_memory_info", lambda: {"allocated_mb": 1234.5})

    def run(exc):
        def boom(path, language, model):
            raise exc

        monkeypatch.setattr(ws, "_transcribe_sync", boom)
        resp = asyncio.run(ws._run_transcription(SECRET, None, "/transcribe"))
        return resp.status_code, json.loads(resp.body)

    return run


@pytest.mark.parametrize(
    "msg",
    [
        "CUDA error: device-side assert triggered",
        "CUDA error: an illegal memory access was encountered",
        "cuda kernel launch failure",
    ],
)
def test_non_oom_cuda_error_is_a_500(run_with, msg):
    status, body = run_with(RuntimeError(msg))
    assert status == 500
    assert body["error"] != "GPU OOM"


@pytest.mark.parametrize(
    "msg", ["CUDA failed with error out of memory", "CUDA out of memory. Tried to allocate 2 GiB"]
)
def test_out_of_memory_is_a_503(run_with, msg):
    status, body = run_with(RuntimeError(msg))
    assert status == 503
    assert body["error"] == "GPU OOM"


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError(f"CUDA out of memory while reading {SECRET}"),
        RuntimeError(f"cuda driver 550.1 failed on {SECRET}"),
        ValueError(f"bad input {SECRET}"),
        ws.AudioDecodeError(f"ffmpeg failed: {SECRET}: Invalid data"),
    ],
)
def test_error_bodies_carry_no_internals(run_with, exc):
    status, body = run_with(exc)
    assert status in (400, 500, 503)
    text = json.dumps(body)
    assert "secret-path" not in text
    assert "1234.5" not in text and "gpu" not in body and "detail" not in body
    assert set(body) == {"error", "request_id"} and body["request_id"]

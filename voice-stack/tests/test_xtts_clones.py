"""xtts_server endpoints that need no model or torch: admin auth, clip upload
bounds, companion_id validation and clone deletion.

S3 is a fake; the model manager is replaced by one that fails if touched.
"""

import pytest
import xtts_server as xs
from fastapi.testclient import TestClient


class _Untouchable:
    """Stands in for the model manager: any lease means the request got too far."""

    state = xs.TTSState.UNLOADED

    def lease(self):
        raise AssertionError("request reached the model")

    async def unload(self, force=False):
        raise AssertionError("admin route ran")

    async def evict(self):
        raise AssertionError("admin route ran")

    async def ensure_gpu(self):
        raise AssertionError("admin route ran")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(xs, "_mgr", _Untouchable())
    return TestClient(xs.app, raise_server_exceptions=False)


# ── d7d7f18cad: admin routes fail closed ──────────────────────────────────────


@pytest.mark.parametrize("route", ["/admin/unload", "/admin/evict", "/admin/wake"])
@pytest.mark.parametrize("token", [None, "", "  "])
def test_admin_fails_closed_without_token(client, monkeypatch, route, token):
    monkeypatch.delenv("ARGUS_ADMIN_INSECURE", raising=False)
    if token is None:
        monkeypatch.delenv("ARGUS_ADMIN_TOKEN", raising=False)
    else:
        monkeypatch.setenv("ARGUS_ADMIN_TOKEN", token)
    for headers in ({}, {"Authorization": "Bearer "}, {"Authorization": "Bearer anything"}):
        r = client.post(route, headers=headers)
        assert r.status_code == 503, (route, headers, r.status_code)


def test_admin_insecure_opt_in_opens_only_without_a_token(client, monkeypatch):
    monkeypatch.setenv("ARGUS_ADMIN_INSECURE", "1")
    monkeypatch.setenv("ARGUS_ADMIN_TOKEN", "s3cret")
    assert client.post("/admin/unload").status_code == 401


# ── e4a61a4271 / a21d2fc6db: bounded clip uploads ─────────────────────────────


def _clips(n, size=10):
    return [("audio", (f"c{i}.wav", b"x" * size, "audio/wav")) for i in range(n)]


@pytest.mark.parametrize("route,form", [("/analyze-clips", {}), ("/clone-voice", {"companion_id": "abc"})])
def test_too_many_clips_is_413(client, monkeypatch, route, form):
    monkeypatch.setattr(xs, "MAX_CLIPS", 2, raising=False)
    r = client.post(route, files=_clips(3), data=form)
    assert r.status_code == 413


@pytest.mark.parametrize("route,form", [("/analyze-clips", {}), ("/clone-voice", {"companion_id": "abc"})])
def test_oversized_clip_is_413(client, monkeypatch, route, form):
    monkeypatch.setattr(xs, "CLIP_MAX_BYTES", 100, raising=False)
    r = client.post(route, files=_clips(1, size=50) + _clips(1, size=101), data=form)
    assert r.status_code == 413


def test_oversized_clip_leaves_no_temp_file(client, monkeypatch, tmp_path):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(xs, "CLIP_MAX_BYTES", 100, raising=False)
    r = client.post("/analyze-clips", files=_clips(1, size=50) + _clips(1, size=101))
    assert r.status_code == 413
    assert list(tmp_path.iterdir()) == []


# ── 1b02db28cf: companion_id is one safe path segment ─────────────────────────


@pytest.mark.parametrize("cid", ["../other", "a/b", "..", "x" * 65, "", "naïve", "a b"])
def test_clone_voice_rejects_unsafe_companion_id(client, monkeypatch, cid):
    uploads = []
    monkeypatch.setattr(xs, "_upload_to_s3", lambda *a, **k: uploads.append(a))
    r = client.post("/clone-voice", files=_clips(1), data={"companion_id": cid})
    assert r.status_code == 422
    assert uploads == []


@pytest.mark.parametrize("cid", ["a.b", "a b", "x" * 65])
def test_delete_and_download_reject_unsafe_companion_id(client, monkeypatch, cid):
    monkeypatch.setattr(xs, "_get_s3_client", lambda: pytest.fail("S3 reached"))
    assert client.delete(f"/clones/{cid}").status_code == 422
    assert client.get(f"/clones/{cid}/download").status_code == 422


def test_slug_id_always_yields_a_valid_companion_id():
    for raw in ("Naïve Voice!", "../../etc", "x" * 200, "日本語", ""):
        assert xs._COMPANION_ID_RE.fullmatch(xs._slug_id(raw)), raw


# ── 54d31d335a: a partial S3 delete is not reported as deleted ────────────────


class _S3:
    def __init__(self, keys, errors=()):
        self.keys = keys
        self.errors = set(errors)
        self.deleted = []

    def get_paginator(self, name):
        s3 = self

        class P:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in s3.keys if k.startswith(Prefix)]}

        return P()

    def delete_objects(self, Bucket, Delete):
        ok = [o for o in Delete["Objects"] if o["Key"] not in self.errors]
        bad = [o for o in Delete["Objects"] if o["Key"] in self.errors]
        self.deleted += [o["Key"] for o in ok]
        resp = {"Deleted": [{"Key": o["Key"]} for o in ok]}
        if bad:
            resp["Errors"] = [{"Key": o["Key"], "Code": "AccessDenied", "Message": "no"} for o in bad]
        return resp


KEYS = ["voice_clones/abc/prompt.pt", "voice_clones/abc/meta.json"]


def test_partial_delete_is_an_error(client, monkeypatch):
    s3 = _S3(KEYS, errors=["voice_clones/abc/prompt.pt"])
    monkeypatch.setattr(xs, "_get_s3_client", lambda: s3)
    xs._clone_cache["voice_clones/abc/prompt.pt"] = object()
    r = client.delete("/clones/abc")
    assert r.status_code == 500
    assert "prompt.pt" in r.json()["error"]
    assert "voice_clones/abc/prompt.pt" not in xs._clone_cache


def test_other_delete_failures_do_not_leak_their_text(client, monkeypatch):
    def broken():
        raise RuntimeError("EndpointConnectionError: http://10.0.0.9:9000/voice-bucket")
    monkeypatch.setattr(xs, "_get_s3_client", broken)
    r = client.delete("/clones/abc")
    assert r.status_code == 500
    assert "10.0.0.9" not in r.text and "details in the xtts log" in r.json()["error"]


def test_delete_counts_what_s3_deleted(monkeypatch):
    s3 = _S3(KEYS)
    monkeypatch.setattr(xs, "_get_s3_client", lambda: s3)
    assert xs._delete_clone("abc") == 2
    assert sorted(s3.deleted) == sorted(KEYS)


def test_full_delete_is_ok(client, monkeypatch):
    monkeypatch.setattr(xs, "_get_s3_client", lambda: _S3(KEYS))
    r = client.delete("/clones/abc")
    assert r.status_code == 200
    assert r.json()["objects"] == 2

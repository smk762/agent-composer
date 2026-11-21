"""xtts_server hardening that runs without torch, the model or S3: generic
error bodies, clip scoring/flags, duration probing, loudness reporting, the
clone minimum-speech guard, clone listing and import metadata.

torch / pyloudnorm / torchaudio are stubbed per test where a helper imports them.
"""

import io
import json
import sys
import types
import zipfile

import numpy as np
import pytest
import xtts_server as xs
from fastapi.testclient import TestClient


class _Untouchable:
    state = xs.TTSState.UNLOADED

    def lease(self):
        raise AssertionError("request reached the model")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(xs, "_mgr", _Untouchable())
    return TestClient(xs.app, raise_server_exceptions=False)


# ── 2068fcbd12: no exception text in clone download errors ────────────────────


class _BrokenS3:
    def download_fileobj(self, bucket, key, buf):
        raise OSError("/srv/secret/path/voice_clones/abc/prompt.pt: disk exploded")


def test_download_failure_returns_a_generic_error(client, monkeypatch):
    monkeypatch.setattr(xs, "_get_s3_client", lambda: _BrokenS3())
    r = client.get("/clones/abc/download")
    assert r.status_code == 500
    assert r.json() == {"error": "Failed to bundle clone"}


def test_download_of_a_missing_clone_is_a_generic_404(client, monkeypatch):
    def missing(cid):
        raise FileNotFoundError("/srv/secret/path")

    monkeypatch.setattr(xs, "_build_clone_zip", missing)
    r = client.get("/clones/abc/download")
    assert r.status_code == 404
    assert "/srv" not in r.text


def test_list_failure_returns_a_generic_error(client, monkeypatch):
    def boom():
        raise RuntimeError("/srv/secret/path")

    monkeypatch.setattr(xs, "_list_clones", boom)
    r = client.get("/clones")
    assert r.status_code == 500
    assert "/srv" not in r.text


# ── 4c3efb4b29: analyze flags carry no exception text ─────────────────────────


class _FailingEmbed:
    def get_conditioning_latents(self, paths):
        raise RuntimeError("/tmp/tmpabc.wav: cuda internals")


def test_analyze_flags_are_fixed_codes(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    def metrics(path):
        if path == "bad":
            raise RuntimeError("/tmp/tmpxyz.wav: decoder internals")
        return {"flags": [], "speech_seconds": 8.0, "snr_db": 30.0, "clip_ratio": 0.0}

    monkeypatch.setattr(xs, "_quick_clip_metrics", metrics)
    monkeypatch.setattr(xs, "_mgr", type("M", (), {"model": _FailingEmbed()})())
    out = xs._analyze_clips(["good", "bad"], ["good", "bad"])
    flags = [f for c in out["clips"] for f in c["flags"]]
    assert "unreadable" in flags and "embed_failed" in flags
    assert not any("/tmp" in f or ":" in f for f in flags)


# ── f81496de05: zero detected speech is very_short and never recommended ─────


def test_clip_with_no_speech_is_flagged_and_not_recommended(monkeypatch):
    monkeypatch.setattr(xs, "ANALYZE_SCORE_THRESHOLD", 0.0)
    items = [{"name": "silent", "flags": [], "speech_seconds": 0.0, "snr_db": 40.0, "clip_ratio": 0.0,
              "consistency": 0.99}]
    xs._score_clips(items)
    assert "very_short" in items[0]["flags"]
    assert items[0]["recommended"] is False


def test_clip_with_speech_is_still_recommended(monkeypatch):
    monkeypatch.setattr(xs, "ANALYZE_SCORE_THRESHOLD", 0.0)
    items = [{"name": "ok", "flags": [], "speech_seconds": 8.0, "snr_db": 40.0, "clip_ratio": 0.0,
              "consistency": 0.99}]
    xs._score_clips(items)
    assert "very_short" not in items[0]["flags"]
    assert items[0]["recommended"] is True


# ── 1f40ab9def: unknown frame count is an unknown duration ────────────────────


def _torchaudio_info(monkeypatch, num_frames, sample_rate):
    sf = types.ModuleType("soundfile")

    def info(path):
        raise RuntimeError("unsupported container")

    sf.info = info
    ta = types.ModuleType("torchaudio")
    ta.info = lambda path: types.SimpleNamespace(num_frames=num_frames, sample_rate=sample_rate)
    monkeypatch.setitem(sys.modules, "soundfile", sf)
    monkeypatch.setitem(sys.modules, "torchaudio", ta)


def test_audio_duration_with_unknown_frames_is_none(monkeypatch):
    _torchaudio_info(monkeypatch, 0, 44100)
    assert xs._audio_duration("x.mp3") is None


def test_audio_duration_from_torchaudio(monkeypatch):
    _torchaudio_info(monkeypatch, 44100 * 3, 44100)
    assert xs._audio_duration("x.mp3") == 3.0


# ── 4069d02cbd: loudness report when the peak guard attenuates ────────────────


def _pyloudnorm(monkeypatch, loud_in, normed_peak):
    pyln = types.ModuleType("pyloudnorm")

    class Meter:
        def __init__(self, sr):
            pass

        def integrated_loudness(self, audio):
            return loud_in

    def loudness(audio, src, target):
        out = np.zeros_like(audio)
        out[0] = normed_peak
        return out

    pyln.Meter = Meter
    pyln.normalize = types.SimpleNamespace(loudness=loudness)
    monkeypatch.setitem(sys.modules, "pyloudnorm", pyln)


def test_peak_limited_loudness_reports_the_actual_level(monkeypatch):
    monkeypatch.setattr(xs, "PREPROCESS_TARGET_LUFS", -16.0)
    _pyloudnorm(monkeypatch, loud_in=-40.0, normed_peak=1.98)  # 2x over the guard: -6.02 dB
    report = {"warnings": [], "applied": []}
    out = xs._normalise_loudness(np.zeros(100, dtype=np.float32), 24000, report)
    assert float(np.max(np.abs(out))) == pytest.approx(0.99, abs=1e-6)
    assert "lufs_peak_limited" in report["warnings"]
    assert report["lufs_out"] == -22.0
    assert report["gain_db"] == 18.0


def test_unlimited_loudness_reports_the_target(monkeypatch):
    monkeypatch.setattr(xs, "PREPROCESS_TARGET_LUFS", -16.0)
    _pyloudnorm(monkeypatch, loud_in=-20.0, normed_peak=0.5)
    report = {"warnings": [], "applied": []}
    xs._normalise_loudness(np.zeros(100, dtype=np.float32), 24000, report)
    assert "lufs_peak_limited" not in report["warnings"]
    assert report["lufs_out"] == -16.0
    assert report["gain_db"] == 4.0


# ── c3cad22394: a measured zero seconds of speech is refused ──────────────────


def _clips(n):
    return [("audio", (f"c{i}.wav", b"x" * 10, "audio/wav")) for i in range(n)]


def test_clone_with_all_speech_trimmed_is_422(client, monkeypatch, tmp_path):
    monkeypatch.setattr(xs, "PREPROCESS", True)

    def preprocess(path):
        out = tmp_path / f"p{len(list(tmp_path.iterdir()))}.wav"
        out.write_bytes(b"")
        return str(out), {"kept_seconds": 0.0, "warnings": [], "applied": []}

    monkeypatch.setattr(xs, "_preprocess_clip", preprocess)
    r = client.post("/clone-voice", files=_clips(2), data={"companion_id": "abc"})
    assert r.status_code == 422


def test_clone_with_probed_zero_duration_is_422(client, monkeypatch):
    monkeypatch.setattr(xs, "PREPROCESS", False)
    monkeypatch.setattr(xs, "_audio_duration", lambda path: 0.0)
    r = client.post("/clone-voice", files=_clips(1), data={"companion_id": "abc"})
    assert r.status_code == 422


def test_clone_with_unprobeable_clips_defers_to_the_model(client, monkeypatch):
    monkeypatch.setattr(xs, "PREPROCESS", False)
    monkeypatch.setattr(xs, "_audio_duration", lambda path: None)
    r = client.post("/clone-voice", files=_clips(1), data={"companion_id": "abc"})
    assert r.status_code == 500  # got past the guard to the (untouchable) model


# ── 6bb8e86a97: one bad meta.json can't break the listing ─────────────────────


class _ListS3:
    def __init__(self, metas):
        self.metas = metas

    def get_paginator(self, name):
        s3 = self

        class P:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in s3.metas]}

        return P()

    def download_fileobj(self, bucket, key, buf):
        buf.write(self.metas[key])


def test_list_clones_skips_non_objects_and_sorts_mixed_dates(monkeypatch):
    metas = {
        "voice_clones/a/meta.json": json.dumps({"companion_id": "a", "created_at": 100.0}).encode(),
        "voice_clones/b/meta.json": json.dumps({"companion_id": "b", "created_at": "2026-01-01T00:00:00Z"}).encode(),
        "voice_clones/c/meta.json": json.dumps({"companion_id": "c"}).encode(),
        "voice_clones/d/meta.json": b"null",
        "voice_clones/e/meta.json": b"[1, 2]",
        "voice_clones/f/meta.json": json.dumps({"companion_id": "f", "created_at": "garbage"}).encode(),
    }
    monkeypatch.setattr(xs, "_get_s3_client", lambda: _ListS3(metas))
    out = xs._list_clones()
    ids = [c["companion_id"] for c in out]
    assert ids[:2] == ["b", "a"]
    assert sorted(ids) == ["a", "b", "c", "f"]


# ── cdd706b301: malformed meta.json fields don't crash the import ─────────────


def _zip(meta_raw: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("prompt.pt", b"latents")
        zf.writestr("meta.json", meta_raw)
    return buf.getvalue()


@pytest.fixture
def importer(monkeypatch):
    torch = types.ModuleType("torch")
    torch.save = lambda obj, buf: buf.write(b"saved")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(xs, "_load_prompt", lambda data: {"ok": True})
    monkeypatch.setattr(xs, "_s3_object_exists", lambda key: False)
    uploads = {}
    monkeypatch.setattr(xs, "_upload_to_s3", lambda data, key, ct="": uploads.__setitem__(key, data))
    monkeypatch.setattr(xs, "_clone_cache", type(xs._clone_cache)())
    return uploads


@pytest.mark.parametrize("meta_raw", [b"[1, 2]", b'"text"', b"42", b"null"])
def test_import_with_a_non_object_meta_json(importer, meta_raw):
    meta = xs._import_clone(_zip(meta_raw), "", "", False)
    assert meta["companion_id"] == "imported-voice"
    assert meta["num_clips"] == 0 and meta["total_seconds"] == 0.0


def test_import_with_wrongly_typed_meta_fields(importer):
    raw = json.dumps({"companion_id": 7, "name": ["x"], "num_clips": "3", "total_seconds": {"a": 1}}).encode()
    meta = xs._import_clone(_zip(raw), "", "", False)
    assert meta["companion_id"] == "imported-voice"
    assert meta["name"] == "imported-voice"
    assert meta["num_clips"] == 0 and meta["total_seconds"] == 0.0
    stored = json.loads(importer["voice_clones/imported-voice/meta.json"])
    assert stored["name"] == "imported-voice"


def test_import_keeps_valid_meta_fields(importer):
    raw = json.dumps({"companion_id": "kept", "name": "Kept", "num_clips": 3, "total_seconds": 12.345}).encode()
    meta = xs._import_clone(_zip(raw), "", "", False)
    assert (meta["companion_id"], meta["name"], meta["num_clips"], meta["total_seconds"]) == ("kept", "Kept", 3, 12.35)


# ── 0121b4afef: non-finite numeric strings sort as 0 ─────────────────────────


@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "-Infinity", "1e999"])
def test_created_at_key_rejects_non_finite_strings(raw):
    assert xs._created_at_key({"created_at": raw}) == 0.0


def test_created_at_key_keeps_finite_numeric_strings():
    assert xs._created_at_key({"created_at": "1700000000.5"}) == 1700000000.5


# ── 34f19c44e1: a huge int in meta.json is not an OverflowError / orphan ─────


def test_import_with_a_huge_integer_meta_field(importer):
    raw = ('{"companion_id": "big", "num_clips": ' + "9" * 400 + ', "total_seconds": ' + "1" * 400 + "}").encode()
    meta = xs._import_clone(_zip(raw), "", "", False)
    assert meta["num_clips"] == 0 and meta["total_seconds"] == 0.0
    assert set(importer) == {"voice_clones/big/prompt.pt", "voice_clones/big/meta.json"}


def test_import_failure_while_building_meta_uploads_nothing(importer, monkeypatch):
    def boom(*a, **k):
        raise ValueError("bad meta")

    monkeypatch.setattr(xs.json, "dumps", boom)
    with pytest.raises(ValueError):
        xs._import_clone(_zip(b'{"companion_id": "x"}'), "", "", False)
    assert importer == {}


# ── 8edd82b3b0: only a real OOM is a 503 ─────────────────────────────────────


class _InferenceMgr:
    state = xs.TTSState.GPU

    def lease(self):
        import contextlib

        @contextlib.asynccontextmanager
        async def cm():
            yield self

        return cm()


@pytest.mark.parametrize("msg,code", [
    ("CUDA error: device-side assert triggered", 500),
    ("Expected all tensors to be on the same device (cuda:0 vs cpu)", 500),
    ("some other runtime failure", 500),
    ("CUDA out of memory. Tried to allocate 2.00 GiB", 503),
])
def test_synthesise_runtime_errors(monkeypatch, msg, code):
    monkeypatch.setattr(xs, "_mgr", _InferenceMgr())
    monkeypatch.setattr(xs, "_resolve_latents", lambda req: (None, None))
    monkeypatch.setattr(xs, "_gpu_memory_info", lambda: {})

    def fail(*a):
        raise RuntimeError(msg)

    monkeypatch.setattr(xs, "_run_inference", fail)
    c = TestClient(xs.app, raise_server_exceptions=False)
    r = c.post("/synthesise", json={"text": "hi", "language": "en"})
    assert r.status_code == code, r.text
    assert ("GPU OOM" in r.text) == (code == 503)

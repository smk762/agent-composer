"""xtts_server: safe clone loading, model leases, streaming backpressure.

The real XTTS model is never loaded: the manager's promote step is replaced by
one that installs a fake model, so these run without a GPU or Coqui weights.
"""

import asyncio
import io
import itertools
import json
import os

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "zeros"):  # another test module's stub, not real torch
    pytest.skip("real torch not installed", allow_module_level=True)
np = pytest.importorskip("numpy")

import xtts_server as xs  # noqa: E402


# ── 59b9e3a46b: torch.load(weights_only=True) + structure validation ─────────


class _Evil:
    def __reduce__(self):
        return (os.system, ("touch /tmp/xtts_pwned_marker",))


def _saved(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def test_load_prompt_accepts_clone_format():
    prompt = {"gpt_cond_latent": torch.randn(1, 32, 1024), "speaker_embedding": torch.randn(1, 512, 1)}
    out = xs._load_prompt(_saved(prompt))
    assert set(out) == {"gpt_cond_latent", "speaker_embedding"}
    assert torch.equal(out["gpt_cond_latent"], prompt["gpt_cond_latent"])


def test_load_prompt_refuses_pickle_payload():
    marker = "/tmp/xtts_pwned_marker"
    if os.path.exists(marker):
        os.unlink(marker)
    blob = _saved({"gpt_cond_latent": _Evil(), "speaker_embedding": torch.zeros(1)})
    with pytest.raises(ValueError):
        xs._load_prompt(blob)
    assert not os.path.exists(marker)


@pytest.mark.parametrize(
    "obj",
    [
        [torch.zeros(1), torch.zeros(1)],
        {"gpt_cond_latent": torch.zeros(1)},
        {"gpt_cond_latent": torch.zeros(1), "speaker_embedding": "nope"},
        {"gpt_cond_latent": torch.zeros(1, dtype=torch.long), "speaker_embedding": torch.zeros(1)},
        {"gpt_cond_latent": torch.zeros(0), "speaker_embedding": torch.zeros(1)},
    ],
)
def test_load_prompt_rejects_bad_structure(obj):
    with pytest.raises(ValueError):
        xs._load_prompt(_saved(obj))


def test_get_clone_latents_uses_safe_loader(monkeypatch):
    blob = _saved({"gpt_cond_latent": _Evil(), "speaker_embedding": torch.zeros(1)})

    class S3:
        def download_fileobj(self, bucket, key, buf):
            buf.write(blob)

    monkeypatch.setattr(xs, "_get_s3_client", lambda: S3())
    xs._clone_cache.clear()
    with pytest.raises(ValueError, match="not a valid XTTS clone"):
        xs._get_clone_latents("voice_clones/x/prompt.pt")
    assert "voice_clones/x/prompt.pt" not in xs._clone_cache


# ── fake model / manager wiring ───────────────────────────────────────────────


class _FakeModel:
    def __init__(self, n_chunks=0, chunk_len=256):
        self.n_chunks = n_chunks
        self.chunk_len = chunk_len
        self.produced = 0
        self.closed = False
        self.device_seen = []

    def parameters(self):
        return iter([torch.zeros(1)])

    def inference_stream(self, *args, **kwargs):
        try:
            for i in range(self.n_chunks):
                self.produced += 1
                yield np.full(self.chunk_len, i, dtype=np.float32)
        finally:
            self.closed = True


@pytest.fixture
def mgr(monkeypatch):
    m = xs.XTTSModelManager()
    model = _FakeModel()
    moves = []

    async def promote():
        if m._state != xs.TTSState.GPU:
            m._model = model
            m._state = xs.TTSState.GPU

    def move(device):
        moves.append(device)

    monkeypatch.setattr(m, "_ensure_gpu_locked", promote)
    monkeypatch.setattr(m, "_move_to_device", move)
    monkeypatch.setattr(xs, "DEVICE", "cuda:0")
    m.moves = moves
    m.fake = model
    return m


# ── d196101bad: leases block demote/unload ────────────────────────────────────


def test_keep_alive_zero_waits_for_lease(mgr):
    async def run():
        mgr.keep_alive_gpu, mgr.keep_alive_cpu = 0, -1
        async with mgr.lease():
            for _ in range(5):
                await asyncio.sleep(0)
            assert mgr.state == xs.TTSState.GPU and mgr.moves == []
            assert await mgr.demote_to_cpu() is False
            assert await mgr.unload() is False
            assert mgr.state == xs.TTSState.GPU
        await asyncio.sleep(0.05)  # release re-armed eviction
        assert mgr.active == 0
        assert mgr.state == xs.TTSState.CPU and mgr.moves == ["cpu"]

    asyncio.run(run())


def test_release_restarts_idle_timer(mgr):
    async def run():
        mgr.keep_alive_gpu, mgr.keep_alive_cpu = 0.1, -1
        async with mgr.lease():
            await asyncio.sleep(0.2)  # longer than keep_alive_gpu
            assert mgr.state == xs.TTSState.GPU
        await asyncio.sleep(0.05)
        assert mgr.state == xs.TTSState.GPU  # idle only 0.05 s
        await asyncio.sleep(0.15)
        assert mgr.state == xs.TTSState.CPU

    asyncio.run(run())


def test_negative_gpu_keep_alive_never_unloads(mgr):
    async def run():
        mgr.keep_alive_gpu, mgr.keep_alive_cpu = -1, 0
        async with mgr.lease():
            pass
        await asyncio.sleep(0.05)
        assert mgr.state == xs.TTSState.GPU and mgr.moves == []

    asyncio.run(run())


def test_evict_arms_the_cpu_stage_with_gpu_pinned(mgr):
    """#52: keep_alive_gpu=-1 (iris owns GPU residency) must not keep an evicted model in RAM."""
    async def run():
        mgr.keep_alive_gpu, mgr.keep_alive_cpu = -1, 0.05
        await mgr.ensure_gpu()
        assert await mgr.evict() is True
        assert mgr.state == xs.TTSState.CPU
        await asyncio.sleep(0.02)
        assert mgr.state == xs.TTSState.CPU
        await asyncio.sleep(0.08)
        assert mgr.state == xs.TTSState.UNLOADED

    asyncio.run(run())


def test_evict_with_zero_cpu_keep_alive_unloads_at_once(mgr):
    async def run():
        mgr.keep_alive_gpu, mgr.keep_alive_cpu = 120, 0
        await mgr.ensure_gpu()
        assert await mgr.evict() is True
        await asyncio.sleep(0.02)
        assert mgr.state == xs.TTSState.UNLOADED

    asyncio.run(run())


def test_evict_with_negative_cpu_keep_alive_stays_on_cpu(mgr):
    async def run():
        mgr.keep_alive_gpu, mgr.keep_alive_cpu = -1, -1
        await mgr.ensure_gpu()
        assert await mgr.evict() is True
        await asyncio.sleep(0.05)
        assert mgr.state == xs.TTSState.CPU

    asyncio.run(run())


def test_cpu_stage_after_evict_is_timed_from_the_evict(mgr):
    # The old check (idle < gpu + cpu) kept a model evicted early in its GPU
    # keep-alive in RAM after the CPU keep-alive ran out.
    async def run():
        mgr.keep_alive_gpu, mgr.keep_alive_cpu = 0.3, 0.05
        await mgr.ensure_gpu()
        assert await mgr.evict() is True
        await asyncio.sleep(0.1)
        assert mgr.state == xs.TTSState.UNLOADED

    asyncio.run(run())


def test_use_during_the_cpu_stage_cancels_the_unload(mgr):
    async def run():
        mgr.keep_alive_gpu, mgr.keep_alive_cpu = -1, 0.05
        await mgr.ensure_gpu()
        await mgr.evict()
        async with mgr.lease():
            await asyncio.sleep(0.08)
            assert mgr.state == xs.TTSState.GPU
        await asyncio.sleep(0.08)
        assert mgr.state == xs.TTSState.GPU  # pinned again once back on GPU

    asyncio.run(run())


def test_evict_while_leased_is_refused_and_arms_nothing(mgr):
    async def run():
        mgr.keep_alive_gpu, mgr.keep_alive_cpu = -1, 0
        async with mgr.lease():
            assert await mgr.evict() is False
            await asyncio.sleep(0.02)
            assert mgr.state == xs.TTSState.GPU

    asyncio.run(run())


def test_force_unload_ignores_leases(mgr):
    async def run():
        await mgr.acquire()
        assert await mgr.unload(force=True) is True
        assert mgr.state == xs.TTSState.UNLOADED

    asyncio.run(run())


# ── 4777913708: streaming backpressure + sentinel ─────────────────────────────


def _install(monkeypatch, mgr):
    monkeypatch.setattr(xs, "_mgr", mgr)
    monkeypatch.setattr(xs, "_resolve_latents", lambda req: (torch.zeros(1), torch.zeros(1)))


def test_stream_slow_consumer_gets_every_chunk(monkeypatch, mgr):
    n = xs._STREAM_QUEUE_MAX * 3
    mgr.fake.n_chunks = n
    _install(monkeypatch, mgr)
    mgr.keep_alive_gpu, mgr.keep_alive_cpu = -1, -1

    async def run():
        resp = await xs.synthesise_stream(xs.SynthesiseRequest(text="hi"))
        await asyncio.sleep(0.3)  # let the producer fill the bounded queue
        got = []
        async for item in resp.body_iterator:
            got.append(np.frombuffer(item, dtype=np.float32)[0])
        assert got == list(range(n))  # nothing dropped, ends (sentinel arrived)
        for _ in range(50):
            if mgr.active == 0:
                break
            await asyncio.sleep(0.02)
        assert mgr.active == 0

    asyncio.run(asyncio.wait_for(run(), 30))


def test_stream_disconnect_stops_producer_and_releases(monkeypatch, mgr):
    mgr.fake.n_chunks = 10_000
    _install(monkeypatch, mgr)
    mgr.keep_alive_gpu, mgr.keep_alive_cpu = -1, -1

    async def run():
        resp = await xs.synthesise_stream(xs.SynthesiseRequest(text="hi"))
        it = resp.body_iterator
        for _ in range(3):
            await it.__anext__()
        assert mgr.active == 1  # the stream holds its lease while generating
        await it.aclose()  # client went away
        for _ in range(100):
            if mgr.active == 0:
                break
            await asyncio.sleep(0.05)
        assert mgr.active == 0
        assert mgr.fake.closed
        assert mgr.fake.produced < 10_000

    asyncio.run(asyncio.wait_for(run(), 30))


def test_stream_failing_mid_generation_is_not_a_complete_body(monkeypatch, mgr):
    """After the 200 headers, a generation failure must abort the body, not end it cleanly."""
    _install(monkeypatch, mgr)
    mgr.keep_alive_gpu, mgr.keep_alive_cpu = -1, -1

    def broken_stream(*args, **kwargs):
        yield np.full(256, 0, dtype=np.float32)
        raise RuntimeError("CUDA error: device-side assert")

    monkeypatch.setattr(mgr.fake, "inference_stream", broken_stream)

    async def run():
        resp = await xs.synthesise_stream(xs.SynthesiseRequest(text="hi"))
        got = []
        with pytest.raises(xs.StreamTruncated):
            async for item in resp.body_iterator:
                got.append(item)
        assert len(got) == 1  # what was generated still went out first
        for _ in range(50):
            if mgr.active == 0:
                break
            await asyncio.sleep(0.02)
        assert mgr.active == 0

    asyncio.run(asyncio.wait_for(run(), 30))


def test_evict_says_whether_it_demoted(monkeypatch, mgr):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(xs, "_mgr", mgr)
    monkeypatch.setattr(xs, "_resident_gpu", lambda: {})
    monkeypatch.setenv("ARGUS_ADMIN_TOKEN", "t0k")

    async def nothing_on_gpu():
        return False

    monkeypatch.setattr(mgr, "evict", nothing_on_gpu)
    body = TestClient(xs.app).post("/admin/evict", headers={"Authorization": "Bearer t0k"}).json()
    assert (body["status"], body["demoted"]) == ("evicted", False)  # off the GPU, but this call moved nothing


def test_stream_bad_clone_is_422_and_releases(monkeypatch, mgr):
    monkeypatch.setattr(xs, "_mgr", mgr)

    def bad(req):
        raise ValueError("Voice clone not found: x")

    monkeypatch.setattr(xs, "_resolve_latents", bad)
    mgr.keep_alive_gpu, mgr.keep_alive_cpu = -1, -1

    async def run():
        resp = await xs.synthesise_stream(xs.SynthesiseRequest(text="hi", voice_clone_id="x"))
        assert resp.status_code == 422
        assert mgr.active == 0

    asyncio.run(run())


def test_stream_holds_one_lease_from_validation(monkeypatch, mgr):
    # With keep_alive_gpu=0 the old code released after validation (arming an
    # immediate demote) and re-acquired inside the generator.
    mgr.fake.n_chunks = 10_000
    _install(monkeypatch, mgr)
    mgr.keep_alive_gpu, mgr.keep_alive_cpu = 0, -1

    async def run():
        resp = await xs.synthesise_stream(xs.SynthesiseRequest(text="hi"))
        assert mgr.active == 1
        await asyncio.sleep(0.1)
        assert mgr.moves == [] and mgr.state == xs.TTSState.GPU
        it = resp.body_iterator
        await it.__anext__()
        await it.aclose()
        for _ in range(100):
            if mgr.active == 0:
                break
            await asyncio.sleep(0.05)
        assert mgr.active == 0

    asyncio.run(asyncio.wait_for(run(), 30))


def test_concurrent_streams_do_not_starve_default_executor(monkeypatch, mgr):
    # A one-thread default executor: the old producer-in-executor + to_thread
    # consumer design deadlocked here as soon as two streams overlapped.
    from concurrent.futures import ThreadPoolExecutor

    n = xs._STREAM_QUEUE_MAX * 2
    mgr.fake.n_chunks = n
    _install(monkeypatch, mgr)
    mgr.keep_alive_gpu, mgr.keep_alive_cpu = -1, -1

    async def run():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(1))
        resps = [await xs.synthesise_stream(xs.SynthesiseRequest(text="hi")) for _ in range(4)]
        await asyncio.sleep(0.2)  # producers fill their bounded queues

        async def drain(resp):
            return len([item async for item in resp.body_iterator])

        assert await asyncio.gather(*(drain(r) for r in resps)) == [n] * 4

    asyncio.run(asyncio.wait_for(run(), 30))



# ── 7cb28ebefe: a caller cancelled mid-load/move doesn't strand the state ─────


def _blocking(fn_after):
    import threading

    started, release = threading.Event(), threading.Event()

    def blocked(*args):
        started.set()
        release.wait(5)
        fn_after(*args)

    return started, release, blocked


async def _until(cond, timeout=2.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not cond():
        assert loop.time() < deadline, "condition not reached"
        await asyncio.sleep(0.01)


@pytest.fixture
def real_mgr(monkeypatch):
    """A manager with its real load/promote logic; only the weights are fake."""
    m = xs.XTTSModelManager()
    m.keep_alive_gpu = m.keep_alive_cpu = -1
    model = _FakeModel()
    monkeypatch.setattr(xs, "DEVICE", "cuda:0")
    monkeypatch.setattr(m, "_move_to_device", lambda device: None)
    m.fake = model
    return m


def _cancel_during(started, coro_fn):
    async def go():
        t = asyncio.create_task(coro_fn())
        await _until(started.is_set)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t

    return go()


def test_cancelled_first_load_does_not_stick_in_loading(real_mgr, monkeypatch):
    m = real_mgr
    started, release, load = _blocking(lambda: setattr(m, "_model", m.fake))
    monkeypatch.setattr(m, "_load_to_cpu", load)

    async def run():
        await _cancel_during(started, m.acquire)
        assert m.active == 0 and not m._lock.locked()
        release.set()
        await _until(lambda: m.state == xs.TTSState.CPU)
        async with m.lease() as model:
            assert model is m.fake
            assert m.state == xs.TTSState.GPU

    asyncio.run(run())


def test_lease_during_orphaned_load_waits_for_the_model(real_mgr, monkeypatch):
    m = real_mgr
    started, release, load = _blocking(lambda: setattr(m, "_model", m.fake))
    monkeypatch.setattr(m, "_load_to_cpu", load)

    async def run():
        await _cancel_during(started, m.acquire)

        async def use():
            async with m.lease() as model:
                return model

        t = asyncio.create_task(use())
        await asyncio.sleep(0.05)
        assert not t.done()  # no lease handed out mid-load
        release.set()
        assert await asyncio.wait_for(t, 2) is m.fake

    asyncio.run(run())


def test_cancelled_load_that_fails_lands_in_error(real_mgr, monkeypatch):
    m = real_mgr

    def boom():
        raise RuntimeError("no weights")

    started, release, load = _blocking(boom)
    monkeypatch.setattr(m, "_load_to_cpu", load)

    async def run():
        await _cancel_during(started, m.acquire)
        release.set()
        await _until(lambda: m.state == xs.TTSState.ERROR)
        assert m._error == "no weights"

    asyncio.run(run())


def test_cancelled_promote_still_lands_on_gpu(real_mgr, monkeypatch):
    m = real_mgr
    m._model, m._state = m.fake, xs.TTSState.CPU
    started, release, move = _blocking(lambda device: None)
    monkeypatch.setattr(m, "_move_to_device", move)

    async def run():
        await _cancel_during(started, m.acquire)
        release.set()
        await _until(lambda: m.state == xs.TTSState.GPU)

    asyncio.run(run())


def test_lease_never_yields_a_missing_model(real_mgr):
    m = real_mgr
    m._state = xs.TTSState.LOADING  # stranded, with no load running

    async def run():
        with pytest.raises(RuntimeError):
            async with m.lease():
                pass
        assert m.active == 0

    asyncio.run(run())


# ── clip analysis: 8720a216fb, 119643bf2a, 95ca17862c ─────────────────────────


_GOOD = {"flags": [], "raw_seconds": 8.0, "peak": 0.5, "clip_ratio": 0.0, "speech_seconds": 8.0, "snr_db": 30.0}


class _EmbedModel:
    def __init__(self, fail=()):
        self.fail = set(fail)

    def get_conditioning_latents(self, paths):
        if paths[0] in self.fail:
            raise RuntimeError("embedding failed")
        return None, torch.ones(1, 512, 1)


def _analyze(monkeypatch, paths, model):
    def metrics(path):
        if "bad" in path:
            raise RuntimeError("could not decode")
        return {**_GOOD, "flags": []}

    monkeypatch.setattr(xs, "_quick_clip_metrics", metrics)
    monkeypatch.setattr(xs, "_mgr", type("M", (), {"model": model})())
    return xs._analyze_clips(paths, paths)


def test_unreadable_clip_scores_zero_and_sorts_last(monkeypatch):
    monkeypatch.setattr(xs, "ANALYZE_SCORE_THRESHOLD", 40.0)
    out = _analyze(monkeypatch, ["good1", "bad", "good2"], _EmbedModel())
    bad = out["clips"][1]
    assert bad["score"] == 0
    assert bad["recommended"] is False
    assert out["order"][-1] == 1
    assert all(c["recommended"] for i, c in enumerate(out["clips"]) if i != 1)


def test_clip_whose_embedding_failed_is_not_given_neutral_consistency(monkeypatch):
    out = _analyze(monkeypatch, ["good1", "noembed", "good2"], _EmbedModel(fail=["noembed"]))
    clip = out["clips"][1]
    assert any(f.startswith("embed_failed") for f in clip["flags"])
    assert "outlier" in clip["flags"]
    assert clip["score"] < out["clips"][0]["score"]


def _fake_audio_stack(monkeypatch, wav, sr, timestamps):
    """Stub torchaudio + silero_vad so the clip helpers run on a given tensor."""
    import sys
    import types

    ta = types.ModuleType("torchaudio")
    af = types.ModuleType("torchaudio.functional")
    af.resample = lambda x, a, b: x
    af.highpass_biquad = lambda x, s, f: x
    ta.functional = af
    ta.load = lambda path: (wav.clone(), sr)
    sv = types.ModuleType("silero_vad")
    sv.get_speech_timestamps = lambda *a, **k: list(timestamps)
    monkeypatch.setitem(sys.modules, "torchaudio", ta)
    monkeypatch.setitem(sys.modules, "torchaudio.functional", af)
    monkeypatch.setitem(sys.modules, "silero_vad", sv)
    monkeypatch.setattr(xs, "_get_vad_model", lambda: None)


def test_clip_with_no_speech_reports_no_speech_seconds(monkeypatch):
    _fake_audio_stack(monkeypatch, torch.rand(1, 16000 * 5) * 0.01, 16000, [])
    m = xs._quick_clip_metrics("silence.wav")
    assert "vad_no_speech" in m["flags"]
    assert m["raw_seconds"] == 5.0
    assert m["speech_seconds"] == 0.0


def test_padded_vad_segments_do_not_repeat_audio(monkeypatch):
    sr = xs.PREPROCESS_SR
    # Two segments 0.1 s apart; a 0.1 s pad on each side makes them overlap by 0.1 s.
    ts = [{"start": 1.0, "end": 2.0}, {"start": 2.1, "end": 3.0}]
    _fake_audio_stack(monkeypatch, torch.rand(1, sr * 4) - 0.5, sr, ts)
    monkeypatch.setattr(xs, "PREPROCESS_VAD_PAD", 0.1)
    monkeypatch.setattr(xs, "PREPROCESS_DENOISE", False)
    monkeypatch.setattr(xs, "PREPROCESS_HIGHPASS_HZ", 0)
    path, report = xs._preprocess_clip("clip.wav")
    os.unlink(path)
    assert "vad_trim" in report["applied"]
    assert report["kept_seconds"] == 2.2  # 0.9 s .. 3.1 s, each sample once


def test_padded_spans_merge_overlaps():
    ts = [{"start": 1.0, "end": 2.0}, {"start": 2.05, "end": 2.1}, {"start": 2.1, "end": 3.0}]
    spans = xs._padded_spans(ts, 0.1, 100, 1000)
    assert all(b > a for a, b in spans)
    assert all(prev[1] <= nxt[0] for prev, nxt in itertools.pairwise(spans))  # no sample twice
    assert spans[0][0] == 90 and spans[-1][1] == 310
    assert sum(b - a for a, b in spans) == 220


# ── /clones/import: 8b6eb3b552, 980cfee602 ────────────────────────────────────


def _archive(meta=None, meta_raw=None, prompt=None, extra=0) -> bytes:
    import zipfile

    if prompt is None:
        prompt = {"gpt_cond_latent": torch.randn(1, 32, 1024), "speaker_embedding": torch.randn(1, 512, 1)}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prompt.pt", _saved(prompt))
        if meta_raw is not None:
            zf.writestr("meta.json", meta_raw)
        elif meta is not None:
            zf.writestr("meta.json", json.dumps(meta))
        for i in range(extra):
            zf.writestr(f"junk{i}", b"")
    return buf.getvalue()


class _ImportS3:
    def __init__(self, existing=()):
        self.existing = set(existing)

    def head_object(self, Bucket, Key):
        from botocore.exceptions import ClientError

        if Key not in self.existing:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {}


@pytest.fixture
def importer(monkeypatch):
    from fastapi.testclient import TestClient

    uploads = {}
    s3 = _ImportS3()
    monkeypatch.setattr(xs, "_get_s3_client", lambda: s3)
    monkeypatch.setattr(xs, "_upload_to_s3", lambda data, key, ct="": uploads.__setitem__(key, data))
    client = TestClient(xs.app, raise_server_exceptions=False)

    def post(archive, **form):
        return client.post("/clones/import", files={"archive": ("v.zip", archive, "application/zip")}, data=form)

    post.uploads, post.s3 = uploads, s3
    return post


def test_import_refuses_an_oversized_meta_json(importer):
    # ~1 MiB of zeros compresses to a few KiB: a small archive, a large member.
    r = importer(_archive(meta_raw=b'{"name": "' + b"0" * (1024 * 1024) + b'"}'))
    assert r.status_code == 422
    assert "meta.json" in r.json()["error"]
    assert importer.uploads == {}


def test_import_refuses_an_archive_with_too_many_entries(importer):
    r = importer(_archive(meta={"companion_id": "abc"}, extra=40))
    assert r.status_code == 422
    assert importer.uploads == {}


def test_import_refuses_an_oversized_archive(importer, monkeypatch):
    monkeypatch.setattr(xs, "_MAX_ARCHIVE_BYTES", 1000, raising=False)
    r = importer(_archive(meta={"companion_id": "abc"}))
    assert r.status_code == 413
    assert importer.uploads == {}


def test_import_does_not_overwrite_an_existing_clone(importer):
    importer.s3.existing.add("voice_clones/victim/prompt.pt")
    r = importer(_archive(meta={"companion_id": "victim", "name": "Victim"}))
    assert r.status_code == 409
    assert importer.uploads == {}


def test_import_replaces_an_existing_clone_only_when_asked(importer):
    importer.s3.existing.add("voice_clones/victim/prompt.pt")
    r = importer(_archive(meta={"companion_id": "victim"}), replace="true")
    assert r.status_code == 200, r.text
    assert "voice_clones/victim/prompt.pt" in importer.uploads


def test_import_of_a_new_id_still_works(importer):
    r = importer(_archive(meta={"companion_id": "Fresh Voice", "name": "Fresh"}))
    assert r.status_code == 200, r.text
    assert r.json()["companion_id"] == "fresh-voice"
    assert set(importer.uploads) == {"voice_clones/fresh-voice/prompt.pt", "voice_clones/fresh-voice/meta.json"}

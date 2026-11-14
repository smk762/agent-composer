"""xtts_server: safe clone loading, model leases, streaming backpressure.

The real XTTS model is never loaded: the manager's promote step is replaced by
one that installs a fake model, so these run without a GPU or Coqui weights.
"""

import asyncio
import io
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

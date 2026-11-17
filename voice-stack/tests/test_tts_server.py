"""TTSModelManager lease / keep-alive and /clone-voice validation tests.

No torch, qwen_tts or S3 needed: torch is stubbed and S3 helpers are patched.
Needs fastapi, python-multipart, httpx and structlog.
Run from voice-stack/: python -m pytest tests/test_tts_server.py -v
"""
import asyncio
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.device = lambda d: d
    _torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    sys.modules["torch"] = _torch

import tts_server  # noqa: E402
from tts_server import TTSModelManager, TTSState  # noqa: E402


class FakeInner:
    def __init__(self):
        self.device = "cuda:0"
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        self.device = device
        return self


class FakeWrapper:
    def __init__(self):
        self.model = FakeInner()
        self.device = "cuda:0"


def _manager(gpu: float, cpu: float) -> TTSModelManager:
    mgr = TTSModelManager()
    mgr.keep_alive_gpu = gpu
    mgr.keep_alive_cpu = cpu
    mgr._base_model = FakeWrapper()
    mgr._custom_model = FakeWrapper()
    mgr._state = TTSState.GPU
    return mgr


async def _spin(seconds: float = 0.05) -> None:
    await asyncio.sleep(seconds)


def _blocking_load(mgr: TTSModelManager, fail: bool = False):
    """Replace ``mgr._load`` with one that blocks its worker thread until released."""
    import threading

    started, release = threading.Event(), threading.Event()

    def load(device):
        started.set()
        release.wait(5)
        if fail:
            raise RuntimeError("boom")
        mgr._base_model = FakeWrapper()
        mgr._custom_model = FakeWrapper()

    mgr._load = load
    return started, release


async def _until(cond, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not cond():
        if loop.time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


@mock.patch.object(tts_server, "DEVICE", "cuda:0")
class PartialLoadFailureTests(unittest.IsolatedAsyncioTestCase):
    """8a668dd362: a load that fails after the first model loaded must free it."""

    async def test_failed_second_model_frees_the_first(self):
        mgr = TTSModelManager()
        mgr.keep_alive_gpu = mgr.keep_alive_cpu = -1

        def load(device):
            mgr._base_model = FakeWrapper()
            raise RuntimeError("CUDA out of memory")

        mgr._load = load
        with self.assertRaises(RuntimeError):
            async with mgr.use():
                pass
        self.assertEqual(mgr.state, TTSState.ERROR)
        self.assertIsNone(mgr.base_model)
        self.assertIsNone(mgr.custom_model)


@mock.patch.object(tts_server, "DEVICE", "cuda:0")
class LoadCancelTests(unittest.IsolatedAsyncioTestCase):
    """A caller cancelled during the first load must not strand the manager in LOADING."""

    async def _cancel_mid_load(self, mgr, started):
        async def gen():
            async with mgr.use():
                pass

        t = asyncio.create_task(gen())
        await _until(started.is_set)
        self.assertEqual(mgr.state, TTSState.LOADING)
        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t
        self.assertEqual(mgr._in_use, 0)
        self.assertFalse(mgr._lock.locked())

    async def test_cancelled_first_load_still_reaches_gpu_and_idle_timer(self):
        mgr = TTSModelManager()
        mgr.keep_alive_gpu, mgr.keep_alive_cpu = 0.05, 0
        started, release = _blocking_load(mgr)
        await self._cancel_mid_load(mgr, started)

        release.set()
        await _until(lambda: mgr.state == TTSState.GPU)
        self.assertTrue(mgr.is_ready)
        # Nobody is left to arm eviction: the load did, so the models leave the GPU.
        await _until(lambda: mgr.state == TTSState.UNLOADED)

    async def test_request_during_orphaned_load_waits_for_it(self):
        mgr = TTSModelManager()
        mgr.keep_alive_gpu, mgr.keep_alive_cpu = -1, -1
        started, release = _blocking_load(mgr)
        await self._cancel_mid_load(mgr, started)

        async def gen():
            async with mgr.use():
                return mgr.base_model, mgr.state

        t = asyncio.create_task(gen())
        await _spin()
        self.assertFalse(t.done())  # no lease handed out while still loading
        release.set()
        model, state = await asyncio.wait_for(t, 2)
        self.assertIsNotNone(model)
        self.assertEqual(state, TTSState.GPU)

    async def test_unload_during_orphaned_load_waits_then_frees(self):
        mgr = TTSModelManager()
        mgr.keep_alive_gpu, mgr.keep_alive_cpu = -1, -1
        started, release = _blocking_load(mgr)
        await self._cancel_mid_load(mgr, started)

        u = asyncio.create_task(mgr.unload())
        await _spin()
        self.assertFalse(u.done())
        release.set()
        await asyncio.wait_for(u, 2)
        self.assertEqual(mgr.state, TTSState.UNLOADED)
        self.assertIsNone(mgr.base_model)

    async def test_cancelled_load_that_fails_lands_in_error_and_retries(self):
        mgr = TTSModelManager()
        started, release = _blocking_load(mgr, fail=True)
        await self._cancel_mid_load(mgr, started)
        release.set()
        await _until(lambda: mgr.state == TTSState.ERROR)
        self.assertEqual(mgr._error, "boom")

        started, release = _blocking_load(mgr)
        release.set()
        async with mgr.use():
            self.assertEqual(mgr.state, TTSState.GPU)


@mock.patch.object(tts_server, "DEVICE", "cuda:0")
class LeaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_immediate_keep_alive_does_not_demote_mid_generation(self):
        mgr = _manager(gpu=0, cpu=0)
        base = mgr.base_model
        async with mgr.use():
            await _spin()
            self.assertEqual(mgr.state, TTSState.GPU)
            self.assertIs(mgr.base_model, base)
            self.assertEqual(base.model.moves, [])
        await _spin()
        self.assertEqual(mgr.state, TTSState.UNLOADED)
        self.assertEqual([str(d) for d in base.model.moves], ["cpu"])

    async def test_unload_and_demote_wait_for_generation(self):
        mgr = _manager(gpu=-1, cpu=-1)
        async with mgr.use():
            demote = asyncio.create_task(mgr.demote_to_cpu())
            await _spin()
            self.assertFalse(demote.done())
            self.assertEqual(mgr.state, TTSState.GPU)
        await asyncio.wait_for(demote, 1)
        self.assertEqual(mgr.state, TTSState.CPU)

        mgr._state = TTSState.GPU
        async with mgr.use():
            unload = asyncio.create_task(mgr.unload())
            await _spin()
            self.assertFalse(unload.done())
            self.assertIsNotNone(mgr.custom_model)
        await asyncio.wait_for(unload, 1)
        self.assertEqual(mgr.state, TTSState.UNLOADED)

    async def test_last_used_updated_on_release_and_timer_starts_then(self):
        mgr = _manager(gpu=0.05, cpu=-1)
        async with mgr.use():
            await _spin(0.1)
            started = mgr._last_used
        self.assertGreater(mgr._last_used, started)
        await _spin(0.02)
        self.assertEqual(mgr.state, TTSState.GPU)
        await _spin(0.08)
        self.assertEqual(mgr.state, TTSState.CPU)


@mock.patch.object(tts_server, "DEVICE", "cuda:0")
class KeepAliveTests(unittest.IsolatedAsyncioTestCase):
    async def test_gpu_forever_never_unloads(self):
        for cpu in (0, 0.01):
            mgr = _manager(gpu=-1, cpu=cpu)
            await mgr.ensure_gpu()
            await _spin()
            self.assertEqual(mgr.state, TTSState.GPU, f"cpu={cpu}")
            self.assertIsNotNone(mgr.base_model)

    async def test_cpu_stage_after_demotion(self):
        mgr = _manager(gpu=0.03, cpu=0.05)
        async with mgr.use():
            pass
        await _spin(0.05)
        self.assertEqual(mgr.state, TTSState.CPU)
        await _spin(0.06)
        self.assertEqual(mgr.state, TTSState.UNLOADED)


@mock.patch.object(tts_server, "DEVICE", "cuda:0")
class EvictCpuStageTests(unittest.IsolatedAsyncioTestCase):
    """#52: /admin/evict starts the CPU stage, even with the GPU keep-alive pinned."""

    async def test_evict_with_gpu_pinned_unloads_after_cpu_keep_alive(self):
        mgr = _manager(gpu=-1, cpu=0.05)
        await mgr.demote_to_cpu()
        self.assertEqual(mgr.state, TTSState.CPU)
        await _spin(0.02)
        self.assertEqual(mgr.state, TTSState.CPU)
        await _spin(0.08)
        self.assertEqual(mgr.state, TTSState.UNLOADED)

    async def test_evict_with_zero_cpu_keep_alive_unloads_at_once(self):
        mgr = _manager(gpu=120, cpu=0)
        await mgr.demote_to_cpu()
        await _spin(0.02)
        self.assertEqual(mgr.state, TTSState.UNLOADED)

    async def test_evict_with_negative_cpu_keep_alive_stays_on_cpu(self):
        mgr = _manager(gpu=-1, cpu=-1)
        await mgr.demote_to_cpu()
        await _spin()
        self.assertEqual(mgr.state, TTSState.CPU)
        self.assertIsNotNone(mgr.base_model)

    async def test_use_during_the_cpu_stage_cancels_the_unload(self):
        mgr = _manager(gpu=-1, cpu=0.05)
        await mgr.demote_to_cpu()
        async with mgr.use():
            await _spin(0.08)
            self.assertEqual(mgr.state, TTSState.GPU)
        await _spin(0.08)
        self.assertEqual(mgr.state, TTSState.GPU)

    async def test_evict_when_not_on_gpu_arms_nothing(self):
        mgr = _manager(gpu=-1, cpu=0)
        mgr._state = TTSState.UNLOADED
        await mgr.demote_to_cpu()
        self.assertIsNone(mgr._evict_task)


def _blocking_move(mgr: TTSModelManager):
    """Replace ``mgr._move_to_device`` with one that blocks its worker thread until released."""
    import threading

    started, release = threading.Event(), threading.Event()
    real = mgr._move_to_device

    def move(device):
        started.set()
        release.wait(5)
        real(device)

    mgr._move_to_device = move
    return started, release


@mock.patch.object(tts_server, "DEVICE", "cuda:0")
class MoveCancelTests(unittest.IsolatedAsyncioTestCase):
    """202b6e1e0f: a caller cancelled mid-move must not leave _state naming the old device."""

    async def test_cancelled_demote_still_lands_on_cpu(self):
        mgr = _manager(gpu=-1, cpu=-1)
        started, release = _blocking_move(mgr)
        t = asyncio.create_task(mgr.demote_to_cpu())
        await _until(started.is_set)
        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t
        release.set()
        await _until(lambda: str(mgr.base_model.model.device) == "cpu")
        await _until(lambda: mgr.state == TTSState.CPU)

    async def test_cancelled_promote_still_lands_on_gpu(self):
        mgr = _manager(gpu=-1, cpu=-1)
        mgr._move_to_device("cpu")
        mgr._state = TTSState.CPU
        started, release = _blocking_move(mgr)

        async def gen():
            async with mgr.use():
                pass

        t = asyncio.create_task(gen())
        await _until(started.is_set)
        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t
        release.set()
        await _until(lambda: str(mgr.base_model.model.device) == "cuda:0")
        await _until(lambda: mgr.state == TTSState.GPU)

    async def test_request_after_cancelled_demote_promotes_again(self):
        mgr = _manager(gpu=-1, cpu=-1)
        started, release = _blocking_move(mgr)
        t = asyncio.create_task(mgr.demote_to_cpu())
        await _until(started.is_set)
        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t
        release.set()
        await _until(lambda: str(mgr.base_model.model.device) == "cpu")
        async with mgr.use():
            self.assertEqual(mgr.state, TTSState.GPU)
            self.assertEqual(str(mgr.base_model.model.device), "cuda:0")

    async def test_cancelled_demote_still_runs_the_cpu_stage(self):
        mgr = _manager(gpu=-1, cpu=0.05)
        started, release = _blocking_move(mgr)
        t = asyncio.create_task(mgr.demote_to_cpu())
        await _until(started.is_set)
        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t
        release.set()
        await _until(lambda: mgr.state == TTSState.UNLOADED)


class CloneVoiceTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        self.client = TestClient(tts_server.app)
        self.files = {"audio": ("a.wav", b"RIFF0000", "audio/wav")}

    def _post(self, **data):
        return self.client.post("/clone-voice", files=self.files, data=data)

    def test_rejects_bad_companion_ids(self):
        with mock.patch.object(tts_server, "_s3_object_exists") as exists:
            for bad in ("../other", "a/b", "", "x" * 65, "a b", "a.b"):
                r = self._post(companion_id=bad)
                self.assertEqual(r.status_code, 422, bad)
            exists.assert_not_called()

    def test_refuses_to_overwrite_existing_clone(self):
        with mock.patch.object(tts_server, "_s3_object_exists", return_value=True) as exists, \
                mock.patch.object(tts_server, "_clone_voice_leased") as leased:
            r = self._post(companion_id="alice_1")
        self.assertEqual(r.status_code, 409)
        exists.assert_called_once_with("voice_clones/alice_1/prompt.pt")
        leased.assert_not_called()

    def test_new_or_replace_proceeds(self):
        async def fake_leased(audio, reference_text, companion_id, s3_key):
            return {"voice_clone_id": s3_key}

        tts_server._mgr._state = TTSState.GPU
        try:
            for exists_val, data in ((False, {}), (True, {"replace": "true"})):
                with mock.patch.object(tts_server, "_s3_object_exists", return_value=exists_val), \
                        mock.patch.object(tts_server, "_clone_voice_leased", side_effect=fake_leased):
                    r = self._post(companion_id="Bob-2", **data)
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(r.json()["voice_clone_id"], "voice_clones/Bob-2/prompt.pt")
        finally:
            tts_server._mgr._state = TTSState.UNLOADED
            tts_server._mgr._idle.set()

    def test_concurrent_clones_of_one_id_are_serialised(self):
        # The existence check and the upload run under one per-id lock, so the
        # second replace=false request sees the first one's clone and gets 409.
        import asyncio

        stored = set()

        def exists(key):
            return key in stored

        async def fake_leased(audio, reference_text, companion_id, s3_key):
            await asyncio.sleep(0.05)  # "compute latents"
            stored.add(s3_key)
            return {"voice_clone_id": s3_key}

        async def run():
            files = {"audio": ("a.wav", b"RIFF0000", "audio/wav")}
            import httpx

            transport = httpx.ASGITransport(app=tts_server.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                return await asyncio.gather(*(
                    c.post("/clone-voice", files=files, data={"companion_id": "bob"}) for _ in range(2)
                ))

        tts_server._mgr._state = TTSState.GPU
        try:
            with mock.patch.object(tts_server, "_s3_object_exists", side_effect=exists), \
                    mock.patch.object(tts_server, "_clone_voice_leased", side_effect=fake_leased):
                codes = sorted(r.status_code for r in asyncio.run(run()))
            self.assertEqual(codes, [200, 409])
        finally:
            tts_server._mgr._state = TTSState.UNLOADED
            tts_server._mgr._idle.set()


if __name__ == "__main__":
    unittest.main()


class _FakeCustom:
    def __init__(self, speakers=(), exc=None):
        self.speakers = list(speakers)
        self.exc = exc

    def get_supported_speakers(self):
        return self.speakers

    def generate_custom_voice(self, **kw):
        raise self.exc


class GenerateErrorTests(unittest.TestCase):
    def _run(self, req, custom):
        mgr = TTSModelManager()
        mgr._custom_model = custom
        with mock.patch.object(tts_server, "_mgr", mgr), \
                mock.patch.object(tts_server, "DEFAULT_SPEAKER", ""), \
                mock.patch.object(tts_server, "_gpu_memory_info", return_value={"total_mb": 1.0}):
            return asyncio.run(tts_server._generate_audio_leased(req))

    def test_voice_design_fallback_with_no_speakers_is_503(self):
        # 09c88227e2: no IndexError (and no 500) when the fallback model lists no speakers.
        r = self._run(tts_server.SynthesiseRequest(text="hi", voice_description="calm"), _FakeCustom())
        self.assertEqual(r.status_code, 503)
        self.assertIn(b"no speakers", r.body)

    def test_generation_errors_do_not_leak_exception_text(self):
        # 2e8dbc523a: exception text and GPU info stay in the logs.
        for exc, code in ((OSError("/srv/secret/clones/x.pt missing"), 500),
                          (RuntimeError("CUDA out of memory at /srv/secret driver 555"), 503)):
            r = self._run(tts_server.SynthesiseRequest(text="hi", speaker="ryan"), _FakeCustom(exc=exc))
            self.assertEqual(r.status_code, code)
            self.assertNotIn(b"secret", r.body)
            self.assertNotIn(b"total_mb", r.body)
            self.assertIn(b"error_id", r.body)

"""ModelManager lease / keep-alive tests. No torch or GPU needed: torch and transformers are stubbed.

Run from modernbert/: python -m pytest tests/test_model_lifecycle.py -v
"""
import asyncio
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.float16 = "float16"
    _torch.float32 = "float32"
    _torch.cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)
    sys.modules["torch"] = _torch
if "transformers" not in sys.modules:
    _tf = types.ModuleType("transformers")
    _tf.AutoModel = _tf.AutoModelForSequenceClassification = _tf.AutoTokenizer = object
    sys.modules["transformers"] = _tf

if not hasattr(sys.modules["torch"], "float32"):  # another test file's stub
    sys.modules["torch"].float32 = "float32"

import threading  # noqa: E402

import app.model as model_mod  # noqa: E402
from app.model import ModelManager, ModelState  # noqa: E402


class FakeModel:
    def __init__(self):
        self.device = "cuda"
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        self.device = device
        return self


def _manager(gpu: float, cpu: float, state: ModelState = ModelState.GPU) -> ModelManager:
    mgr = ModelManager()
    mgr._target_device = "cuda"
    mgr.keep_alive_gpu = gpu
    mgr.keep_alive_cpu = cpu
    mgr._model = FakeModel()
    mgr._state = state
    return mgr


async def _spin(seconds: float = 0.05) -> None:
    await asyncio.sleep(seconds)


class LeaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_immediate_keep_alive_does_not_demote_mid_inference(self):
        mgr = _manager(gpu=0, cpu=0)
        async with mgr.use():
            await _spin()  # the old code demoted + unloaded at the first await
            self.assertEqual(mgr.state, ModelState.GPU)
            self.assertIsNotNone(mgr.model)
            self.assertNotIn("cpu", mgr.model.moves)
        await _spin()
        self.assertEqual(mgr.state, ModelState.UNLOADED)
        self.assertIsNone(mgr.model)

    async def test_unload_waits_for_in_flight_inference(self):
        mgr = _manager(gpu=-1, cpu=-1)
        async with mgr.use():
            unload = asyncio.create_task(mgr.unload())
            await _spin()
            self.assertFalse(unload.done())
            self.assertIsNotNone(mgr.model)
        await asyncio.wait_for(unload, 1)
        self.assertEqual(mgr.state, ModelState.UNLOADED)

    async def test_evict_waits_for_in_flight_inference(self):
        mgr = _manager(gpu=-1, cpu=-1)
        model = mgr.model
        async with mgr.use():
            evict = asyncio.create_task(mgr.evict())
            await _spin()
            self.assertFalse(evict.done())
            self.assertEqual(model.moves, [])
        await asyncio.wait_for(evict, 1)
        self.assertEqual(mgr.state, ModelState.CPU)
        self.assertEqual(model.moves, ["cpu"])

    async def test_new_lease_waits_behind_pending_unload(self):
        mgr = _manager(gpu=-1, cpu=-1)
        order = []
        async with mgr.use():
            unload = asyncio.create_task(mgr.unload())
            await _spin(0.01)

            async def second():
                mgr._load_sync = lambda n: (setattr(mgr, "_model", FakeModel()), (None, mgr._model))[1]
                async with mgr.use():
                    order.append(("second", mgr.state))

            task = asyncio.create_task(second())
            await _spin(0.01)
            order.append("first-done")
        await asyncio.wait_for(asyncio.gather(unload, task), 1)
        self.assertEqual(order, ["first-done", ("second", ModelState.GPU)])

    async def test_timer_starts_after_release(self):
        mgr = _manager(gpu=0.05, cpu=-1)
        async with mgr.use():
            await _spin(0.1)  # longer than keep_alive_gpu
            self.assertEqual(mgr.state, ModelState.GPU)
        await _spin(0.02)
        self.assertEqual(mgr.state, ModelState.GPU)
        await _spin(0.08)
        self.assertEqual(mgr.state, ModelState.CPU)

    async def test_concurrent_leases_refcounted(self):
        mgr = _manager(gpu=0, cpu=-1)
        release_first = asyncio.Event()

        async def hold():
            async with mgr.use():
                await release_first.wait()

        t = asyncio.create_task(hold())
        await _spin(0.01)
        async with mgr.use():
            release_first.set()
            await t
            await _spin(0.01)
            self.assertEqual(mgr.state, ModelState.GPU)  # second lease still held
        await _spin(0.01)
        self.assertEqual(mgr.state, ModelState.CPU)


class KeepAliveTests(unittest.IsolatedAsyncioTestCase):
    async def test_gpu_forever_never_unloads(self):
        for cpu in (0, 0.01):
            mgr = _manager(gpu=-1, cpu=cpu)
            async with mgr.use():
                pass
            await _spin()
            self.assertEqual(mgr.state, ModelState.GPU, f"cpu={cpu}")
            self.assertIsNotNone(mgr.model)

    async def test_ensure_ready_gpu_forever_never_unloads(self):
        mgr = _manager(gpu=-1, cpu=0)
        await mgr.ensure_ready()
        await _spin()
        self.assertEqual(mgr.state, ModelState.GPU)

    async def test_cpu_stage_runs_after_demotion(self):
        mgr = _manager(gpu=0.03, cpu=0.05)
        async with mgr.use():
            pass
        await _spin(0.05)
        self.assertEqual(mgr.state, ModelState.CPU)
        await _spin(0.06)
        self.assertEqual(mgr.state, ModelState.UNLOADED)

    async def test_cpu_resident_runs_cpu_stage(self):
        mgr = _manager(gpu=0, cpu=0.03, state=ModelState.CPU)
        mgr._target_device = "cpu"  # CPU-only config: CPU is the ready state
        async with mgr.use():
            self.assertEqual(mgr.state, ModelState.CPU)
        await _spin(0.01)
        self.assertEqual(mgr.state, ModelState.CPU)
        await _spin(0.05)
        self.assertEqual(mgr.state, ModelState.UNLOADED)

    async def test_cpu_resident_honours_gpu_keep_alive_first(self):
        # CPU-only device with the defaults' shape (gpu keep-alive > 0, cpu = 0): the model must
        # stay resident for the keep-alive, not unload as soon as each request finishes.
        mgr = _manager(gpu=0.05, cpu=0, state=ModelState.CPU)
        mgr._target_device = "cpu"
        async with mgr.use():
            pass
        await _spin(0.02)
        self.assertEqual(mgr.state, ModelState.CPU)
        self.assertIsNotNone(mgr.model)
        await _spin(0.06)
        self.assertEqual(mgr.state, ModelState.UNLOADED)


class EvictCpuStageTests(unittest.IsolatedAsyncioTestCase):
    """#52: evict() (/admin/evict, /model/evict) starts the CPU stage, even with the GPU keep-alive pinned."""

    async def test_evict_with_gpu_pinned_unloads_after_cpu_keep_alive(self):
        mgr = _manager(gpu=-1, cpu=0.05)
        await mgr.evict()
        self.assertEqual(mgr.state, ModelState.CPU)
        await _spin(0.02)
        self.assertEqual(mgr.state, ModelState.CPU)
        await _spin(0.08)
        self.assertEqual(mgr.state, ModelState.UNLOADED)

    async def test_evict_with_default_cpu_keep_alive_unloads_at_once(self):
        # MODERNBERT_KEEP_ALIVE_CPU=0, the default: seen on dragon staying `cpu` for minutes.
        mgr = _manager(gpu=300, cpu=0)
        await mgr.evict()
        await _spin(0.02)
        self.assertEqual(mgr.state, ModelState.UNLOADED)
        self.assertIsNone(mgr.model)

    async def test_evict_with_negative_cpu_keep_alive_stays_on_cpu(self):
        mgr = _manager(gpu=-1, cpu=-1)
        await mgr.evict()
        await _spin()
        self.assertEqual(mgr.state, ModelState.CPU)
        self.assertIsNotNone(mgr.model)

    async def test_use_during_the_cpu_stage_cancels_the_unload(self):
        mgr = _manager(gpu=-1, cpu=0.05)
        await mgr.evict()
        async with mgr.use():
            await _spin(0.08)
            self.assertEqual(mgr.state, ModelState.GPU)
        await _spin(0.08)
        self.assertEqual(mgr.state, ModelState.GPU)

    async def test_cpu_resident_with_gpu_pinned_stays(self):
        mgr = _manager(gpu=-1, cpu=0, state=ModelState.CPU)
        mgr._target_device = "cpu"
        async with mgr.use():
            pass
        await _spin()
        self.assertEqual(mgr.state, ModelState.CPU)


class LoadDtypeTests(unittest.TestCase):
    """The load dtype follows the device: fp16 on CPU crashes or crawls on common torch builds."""

    def _loaded_dtype(self, target: str, cuda: bool) -> object:
        seen = {}

        class FakeAuto:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                seen.update(kwargs)
                return types.SimpleNamespace(eval=lambda: None)

        mgr = ModelManager()
        mgr._target_device = target
        saved = (model_mod.AutoModel, model_mod.AutoTokenizer, model_mod.torch.cuda)
        model_mod.AutoModel = model_mod.AutoTokenizer = FakeAuto
        model_mod.torch.cuda = types.SimpleNamespace(is_available=lambda: cuda, empty_cache=lambda: None)
        try:
            mgr._load_sync(2)
        finally:
            model_mod.AutoModel, model_mod.AutoTokenizer, model_mod.torch.cuda = saved
        return seen["torch_dtype"]

    def test_cpu_device_loads_fp32(self):
        self.assertEqual(self._loaded_dtype("cpu", cuda=True), model_mod.torch.float32)

    def test_auto_without_cuda_loads_fp32(self):
        self.assertEqual(self._loaded_dtype("auto", cuda=False), model_mod.torch.float32)

    def test_cuda_loads_fp16(self):
        self.assertEqual(self._loaded_dtype("cuda", cuda=True), model_mod.torch.float16)


class BlockingModel(FakeModel):
    """A model whose ``to()`` blocks its worker thread until released."""

    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.started = threading.Event()
        self.release = threading.Event()

    def to(self, device):
        self.started.set()
        self.release.wait(5)
        return super().to(device)


async def _until(event: threading.Event) -> None:
    for _ in range(200):
        if event.is_set():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("worker never started")


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    """A request cancelled mid-load or mid-promote must not wedge or desync the manager."""

    async def test_cancelled_cold_load_does_not_stay_loading(self):
        mgr = _manager(gpu=-1, cpu=-1, state=ModelState.UNLOADED)
        mgr._model = None
        started, release = threading.Event(), threading.Event()

        def slow_load(n):
            started.set()
            release.wait(5)
            return None, FakeModel()

        mgr._load_sync = slow_load
        task = asyncio.create_task(mgr.ensure_ready())
        await _until(started)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        release.set()
        self.assertNotEqual(mgr.state, ModelState.LOADING)

        # The next request loads again and gets a real model.
        mgr._load_sync = lambda n: (setattr(mgr, "_model", FakeModel()), (None, mgr._model))[1]
        async with mgr.use():
            self.assertEqual(mgr.state, ModelState.GPU)
            self.assertIsNotNone(mgr.model)

    async def test_cancelled_promote_tracks_the_device_and_arms_the_timer(self):
        mgr = _manager(gpu=0.05, cpu=-1, state=ModelState.CPU)
        model = BlockingModel()
        mgr._model = model
        task = asyncio.create_task(mgr.ensure_ready())
        await _until(model.started)
        task.cancel()
        await asyncio.sleep(0.02)
        model.release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        # The worker thread moved the weights to the GPU: the state must say so ...
        self.assertEqual(model.device, "cuda")
        self.assertEqual(mgr.state, ModelState.GPU)
        # ... and the idle timer must still demote it, or the VRAM is held for good.
        await _spin(0.15)
        self.assertEqual(mgr.state, ModelState.CPU)
        self.assertEqual(model.device, "cpu")


if __name__ == "__main__":
    unittest.main()

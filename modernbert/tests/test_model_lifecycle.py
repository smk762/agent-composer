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
    _torch.cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)
    sys.modules["torch"] = _torch
if "transformers" not in sys.modules:
    _tf = types.ModuleType("transformers")
    _tf.AutoModel = _tf.AutoModelForSequenceClassification = _tf.AutoTokenizer = object
    sys.modules["transformers"] = _tf

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
                mgr._load_sync = lambda n: setattr(mgr, "_model", FakeModel())
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


if __name__ == "__main__":
    unittest.main()

"""Idle-reaper lifecycle: the task is kept, restarted after a crash and cancelled on shutdown."""
import asyncio
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import manager  # noqa: E402


class ReaperLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._saved = (manager._idle_reaper, manager._process)
        manager._process = None

    async def asyncTearDown(self):
        manager._idle_reaper, manager._process = self._saved
        task = manager._reaper_task
        if task is not None:
            task.remove_done_callback(manager._reaper_done)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        manager._reaper_task = None

    async def test_startup_keeps_reaper_and_shutdown_cancels_it(self):
        with mock.patch.dict(os.environ, {"INFINITY_PRELOAD": "0"}):
            await manager.startup()
        task = manager._reaper_task
        self.assertIsInstance(task, asyncio.Task)
        self.assertFalse(task.done())

        await manager.shutdown()
        self.assertTrue(task.cancelled())
        self.assertIsNone(manager._reaper_task)

    async def test_crashed_reaper_is_restarted(self):
        calls = []

        async def flaky() -> None:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")
            await asyncio.sleep(3600)

        manager._idle_reaper = flaky
        with mock.patch.dict(os.environ, {"INFINITY_PRELOAD": "0"}):
            await manager.startup()
        first = manager._reaper_task
        for _ in range(20):
            await asyncio.sleep(0)
            if len(calls) >= 2:
                break
        self.assertEqual(len(calls), 2)
        self.assertIsNot(manager._reaper_task, first)
        self.assertFalse(manager._reaper_task.done())


if __name__ == "__main__":
    unittest.main()

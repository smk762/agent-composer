import asyncio
import unittest

from app.inference_retry import InferenceWorkers, RetryableOomError, run_with_oom_wait


class OomRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_oom_retries_then_succeeds(self) -> None:
        attempts = 0
        oom_events = 0
        cleanup_calls = 0
        sleeps: list[float] = []
        now = 0.0

        async def operation() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("CUDA out of memory")
            return "ok"

        def on_oom() -> None:
            nonlocal oom_events
            oom_events += 1

        def clear_memory() -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1

        def monotonic() -> float:
            return now

        async def sleep(seconds: float) -> None:
            nonlocal now
            sleeps.append(seconds)
            now += seconds

        result = await run_with_oom_wait(
            operation,
            on_oom=on_oom,
            interval_s=5,
            max_wait_s=20,
            sleep=sleep,
            monotonic=monotonic,
            clear_memory=clear_memory,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 3)
        self.assertEqual(oom_events, 2)
        self.assertEqual(cleanup_calls, 2)
        self.assertEqual(sleeps, [5, 5])

    async def test_oom_retries_until_deadline_failure(self) -> None:
        attempts = 0
        oom_events = 0
        cleanup_calls = 0
        sleeps: list[float] = []
        now = 0.0

        async def operation() -> str:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("CUDA out of memory")

        def on_oom() -> None:
            nonlocal oom_events
            oom_events += 1

        def clear_memory() -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1

        def monotonic() -> float:
            return now

        async def sleep(seconds: float) -> None:
            nonlocal now
            sleeps.append(seconds)
            now += seconds

        with self.assertRaisesRegex(RetryableOomError, "CUDA out of memory after waiting for capacity"):
            await run_with_oom_wait(
                operation,
                on_oom=on_oom,
                interval_s=4,
                max_wait_s=10,
                sleep=sleep,
                monotonic=monotonic,
                clear_memory=clear_memory,
            )

        self.assertEqual(attempts, 4)
        self.assertEqual(oom_events, 4)
        self.assertEqual(cleanup_calls, 4)
        self.assertEqual(sleeps, [4, 4, 2.0])

    async def test_worker_pool_serializes_tasks(self) -> None:
        workers = InferenceWorkers(1)
        order: list[str] = []

        async def first() -> str:
            order.append("first-start")
            await asyncio.sleep(0.01)
            order.append("first-end")
            return "first"

        async def second() -> str:
            order.append("second-start")
            await asyncio.sleep(0)
            order.append("second-end")
            return "second"

        r1, r2 = await asyncio.gather(
            workers.run(first),
            workers.run(second),
        )

        self.assertEqual((r1, r2), ("first", "second"))
        self.assertEqual(order, ["first-start", "first-end", "second-start", "second-end"])


if __name__ == "__main__":
    unittest.main()

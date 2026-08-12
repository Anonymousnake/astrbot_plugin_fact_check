from __future__ import annotations

import asyncio
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from astrbot_plugin_fact_check.runtime import (
    AsyncSingleFlight,
    run_blocking_with_timeout,
)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_singleflight_cancel_all_stops_and_forgets_active_tasks(self) -> None:
        flight: AsyncSingleFlight[str] = AsyncSingleFlight()
        started = asyncio.Event()

        async def factory() -> str:
            started.set()
            await asyncio.Event().wait()
            return "never"

        task, _ = flight.start_if("active", factory, can_start=lambda: True)
        await started.wait()

        await flight.cancel_all()

        self.assertTrue(task.cancelled())
        self.assertEqual(flight.active_count, 0)

    async def test_singleflight_runs_identical_work_only_once(self) -> None:
        flight: AsyncSingleFlight[str] = AsyncSingleFlight()
        calls = 0
        gate = asyncio.Event()

        async def factory() -> str:
            nonlocal calls
            calls += 1
            await gate.wait()
            return "done"

        first = asyncio.create_task(flight.run("same", factory))
        await asyncio.sleep(0)
        second = asyncio.create_task(flight.run("same", factory))
        await asyncio.sleep(0)
        gate.set()

        first_result, second_result = await asyncio.gather(first, second)

        self.assertEqual(calls, 1)
        self.assertEqual(first_result, ("done", False))
        self.assertEqual(second_result, ("done", True))

    async def test_blocking_work_returns_control_at_the_hard_timeout(self) -> None:
        started = time.perf_counter()

        with self.assertRaises(asyncio.TimeoutError):
            await run_blocking_with_timeout(lambda: time.sleep(0.2), timeout=0.02)

        self.assertLess(time.perf_counter() - started, 0.15)

    async def test_timed_out_worker_keeps_capacity_until_thread_finishes(self) -> None:
        capacity = asyncio.Semaphore(1)
        release = threading.Event()
        second_started = threading.Event()

        with self.assertRaises(asyncio.TimeoutError):
            await run_blocking_with_timeout(
                lambda: release.wait(timeout=1),
                timeout=0.02,
                capacity=capacity,
            )

        self.assertTrue(capacity.locked())
        with self.assertRaises(asyncio.TimeoutError):
            await run_blocking_with_timeout(
                lambda: second_started.set(),
                timeout=0.02,
                capacity=capacity,
            )
        self.assertFalse(second_started.is_set())

        release.set()
        for _ in range(50):
            if not capacity.locked():
                break
            await asyncio.sleep(0.01)
        self.assertFalse(capacity.locked())

    async def test_cancelled_blocking_call_waits_for_worker_to_really_exit(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def work() -> str:
            started.set()
            release.wait(timeout=1)
            return "done"

        task = asyncio.create_task(
            run_blocking_with_timeout(work, timeout=2),
        )
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0.02)

        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_singleflight_start_if_admits_or_joins_atomically(self) -> None:
        flight: AsyncSingleFlight[str] = AsyncSingleFlight()
        gate = asyncio.Event()
        calls = 0

        async def factory() -> str:
            nonlocal calls
            calls += 1
            await gate.wait()
            return "done"

        first, first_joined = flight.start_if(
            "same",
            factory,
            can_start=lambda: True,
        )
        duplicate, duplicate_joined = flight.start_if(
            "same",
            factory,
            can_start=lambda: False,
        )
        rejected, rejected_joined = flight.start_if(
            "different",
            factory,
            can_start=lambda: False,
        )

        self.assertIs(first, duplicate)
        self.assertFalse(first_joined)
        self.assertTrue(duplicate_joined)
        self.assertIsNone(rejected)
        self.assertFalse(rejected_joined)
        self.assertEqual(flight.active_count, 1)
        gate.set()
        self.assertEqual(await first, "done")
        self.assertEqual(calls, 1)

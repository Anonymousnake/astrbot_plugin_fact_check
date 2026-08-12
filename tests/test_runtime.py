from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from astrbot_plugin_fact_check.runtime import (
    AsyncSingleFlight,
    run_blocking_with_timeout,
)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
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

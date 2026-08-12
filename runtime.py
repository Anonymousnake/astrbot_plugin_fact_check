from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar("T")


async def run_blocking_with_timeout(
    func: Callable[[], T],
    *,
    timeout: float,
    capacity: asyncio.Semaphore | None = None,
) -> T:
    """Bound caller wait while retaining capacity until its thread really exits."""
    budget = max(0.01, float(timeout))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget
    acquired = False
    if capacity is not None:
        await asyncio.wait_for(capacity.acquire(), timeout=budget)
        acquired = True

    remaining = max(0.001, deadline - loop.time())
    worker = asyncio.create_task(asyncio.to_thread(func))

    def finalize(done: asyncio.Task[T]) -> None:
        if capacity is not None and acquired:
            capacity.release()
        if done.cancelled():
            return
        try:
            done.exception()
        except asyncio.CancelledError:
            pass

    worker.add_done_callback(finalize)
    return await asyncio.wait_for(asyncio.shield(worker), timeout=remaining)


class AsyncSingleFlight(Generic[T]):
    """Share one in-flight task for identical keys inside one event loop."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[T]] = {}

    def has(self, key: str) -> bool:
        task = self._tasks.get(str(key))
        return bool(task and not task.done())

    @property
    def active_count(self) -> int:
        return sum(1 for task in self._tasks.values() if not task.done())

    def start_if(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
        *,
        can_start: Callable[[], bool],
    ) -> tuple[asyncio.Task[T] | None, bool]:
        """Atomically join an existing task or admit one new task."""
        normalized_key = str(key)
        existing = self._tasks.get(normalized_key)
        if existing is not None and not existing.done():
            return existing, True
        if not can_start():
            return None, False

        task = asyncio.create_task(factory())
        self._tasks[normalized_key] = task

        def cleanup(done: asyncio.Task[T]) -> None:
            if self._tasks.get(normalized_key) is done:
                self._tasks.pop(normalized_key, None)

        task.add_done_callback(cleanup)
        return task, False

    async def run(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
    ) -> tuple[T, bool]:
        task, joined = self.start_if(key, factory, can_start=lambda: True)
        assert task is not None
        return await asyncio.shield(task), joined

    async def cancel_all(self) -> None:
        """Cancel and await every shared pipeline owned by this instance."""
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

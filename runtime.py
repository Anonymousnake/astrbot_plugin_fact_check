from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar("T")


async def run_blocking_with_timeout(
    func: Callable[[], T],
    *,
    timeout: float,
) -> T:
    """Bound the caller's wait even though the worker thread cannot be killed."""
    return await asyncio.wait_for(
        asyncio.to_thread(func),
        timeout=max(0.01, float(timeout)),
    )


class AsyncSingleFlight(Generic[T]):
    """Share one in-flight task for identical keys inside one event loop."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[T]] = {}

    def has(self, key: str) -> bool:
        task = self._tasks.get(str(key))
        return bool(task and not task.done())

    async def run(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
    ) -> tuple[T, bool]:
        normalized_key = str(key)
        existing = self._tasks.get(normalized_key)
        if existing is not None and not existing.done():
            return await asyncio.shield(existing), True

        task = asyncio.create_task(factory())
        self._tasks[normalized_key] = task

        def cleanup(done: asyncio.Task[T]) -> None:
            if self._tasks.get(normalized_key) is done:
                self._tasks.pop(normalized_key, None)

        task.add_done_callback(cleanup)
        return await asyncio.shield(task), False

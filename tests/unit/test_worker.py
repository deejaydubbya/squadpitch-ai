import asyncio

import pytest

from squadpitch_ai.core.config import Settings
from squadpitch_ai.worker.app import Worker


@pytest.mark.asyncio
async def test_worker_startup_shutdown() -> None:
    worker = Worker(Settings(app_env="test"))
    task = asyncio.create_task(worker.run())

    await asyncio.sleep(0)
    await worker.stop()
    await asyncio.wait_for(task, timeout=1)

import asyncio
from datetime import UTC, datetime

import pytest
from redis.exceptions import TimeoutError as RedisTimeoutError

from squadpitch_ai.worker.heartbeat import WorkerHeartbeat


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, tuple[str, int]] = {}
        self.members: dict[str, float] = {}

    async def set(self, name: str, value: str, *, ex: int) -> None:
        self.values[name] = (value, ex)

    async def zadd(self, _name: str, mapping: dict[str, float]) -> None:
        self.members.update(mapping)

    async def zremrangebyscore(self, _name: str, _min: float, max: float) -> None:
        self.members = {key: score for key, score in self.members.items() if score > max}

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_heartbeat_is_bounded_and_contains_only_safe_metadata() -> None:
    redis = FakeRedis()
    heartbeat = WorkerHeartbeat(redis, instance="worker-1", release="release-1", ttl_seconds=360)
    payload = await heartbeat.write(now=datetime(2026, 8, 1, 12, 0, tzinfo=UTC))

    assert payload == {
        "timestamp": "2026-08-01T12:00:00+00:00",
        "service": "squadpitch-ai-worker",
        "instance": "worker-1",
        "release": "release-1",
        "status": "running",
    }
    assert next(iter(redis.values.values()))[1] == 360
    assert redis.members == {"worker-1": datetime(2026, 8, 1, 12, 0, tzinfo=UTC).timestamp() * 1000}


@pytest.mark.asyncio
async def test_heartbeat_recovers_after_transient_redis_timeout() -> None:
    redis = FakeRedis()
    heartbeat = WorkerHeartbeat(redis, instance="worker-1", interval_seconds=0.001)
    original_write = heartbeat.write
    attempts = 0

    async def flaky_write() -> dict[str, str | None]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RedisTimeoutError("synthetic timeout")
        return await original_write()

    heartbeat.write = flaky_write  # type: ignore[assignment]
    stop_event = asyncio.Event()
    task = asyncio.create_task(heartbeat.run(stop_event))

    for _ in range(100):
        if redis.values:
            break
        await asyncio.sleep(0.001)

    stop_event.set()
    await asyncio.wait_for(task, timeout=1)

    assert attempts >= 2
    assert redis.values

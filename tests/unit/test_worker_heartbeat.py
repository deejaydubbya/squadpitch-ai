from datetime import UTC, datetime

import pytest

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
    assert redis.members == {"worker-1": datetime(2026, 8, 1, 12, 0, tzinfo=UTC).timestamp()}

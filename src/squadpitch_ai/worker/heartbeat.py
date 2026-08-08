from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from datetime import UTC, datetime
from typing import Any

from redis.exceptions import RedisError

SERVICE = "squadpitch-ai-worker"
KEY_PREFIX = "sp:worker-health"
logger = logging.getLogger(__name__)


class WorkerHeartbeat:
    def __init__(
        self,
        client: Any,
        *,
        interval_seconds: int = 30,
        ttl_seconds: int = 360,
        instance: str | None = None,
        release: str | None = None,
    ) -> None:
        self.client = client
        self.interval_seconds = interval_seconds
        self.ttl_seconds = ttl_seconds
        self.instance = instance or os.getenv("FLY_MACHINE_ID") or socket.gethostname()
        self.release = release or os.getenv("SP_AI_BUILD_SHA") or os.getenv("FLY_IMAGE_REF")

    async def write(self, *, now: datetime | None = None) -> dict[str, str | None]:
        observed = now or datetime.now(tz=UTC)
        timestamp_ms = observed.timestamp() * 1000
        payload = {
            "timestamp": observed.isoformat(),
            "service": SERVICE,
            "instance": self.instance,
            "release": self.release,
            "status": "running",
        }
        heartbeat_key = f"{KEY_PREFIX}:heartbeat:{SERVICE}:{self.instance}"
        instances_key = f"{KEY_PREFIX}:instances:{SERVICE}"
        await self.client.set(
            heartbeat_key,
            json.dumps(payload, separators=(",", ":")),
            ex=self.ttl_seconds,
        )
        await self.client.zadd(instances_key, {self.instance: timestamp_ms})
        await self.client.zremrangebyscore(
            instances_key,
            0,
            timestamp_ms - (self.ttl_seconds * 2_000),
        )
        return payload

    async def run(self, stop_event: asyncio.Event) -> None:
        redis_unavailable = False
        while not stop_event.is_set():
            try:
                await self.write()
                if redis_unavailable:
                    logger.info("Worker heartbeat Redis connection recovered")
                redis_unavailable = False
            except RedisError as error:
                if not redis_unavailable:
                    logger.warning(
                        "Worker heartbeat write failed; retrying on the next interval (%s)",
                        type(error).__name__,
                    )
                redis_unavailable = True
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                continue

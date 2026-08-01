import asyncio
import signal
from contextlib import suppress
from dataclasses import dataclass

import structlog
from redis.asyncio import Redis

from squadpitch_ai.core.config import Settings, get_settings
from squadpitch_ai.observability.logging import configure_logging
from squadpitch_ai.observability.sentry import capture_exception, init_sentry
from squadpitch_ai.worker.heartbeat import WorkerHeartbeat

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class JobRegistry:
    jobs: tuple[str, ...] = ()


class Worker:
    def __init__(self, settings: Settings, registry: JobRegistry | None = None) -> None:
        self.settings = settings
        self.registry = registry or JobRegistry()
        self._stop_event = asyncio.Event()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._redis: Redis | None = None

    async def start(self) -> None:
        if self.settings.redis_url:
            self._redis = Redis.from_url(self.settings.redis_url, decode_responses=True)
            heartbeat = WorkerHeartbeat(
                self._redis,
                interval_seconds=self.settings.worker_heartbeat_interval_seconds,
                ttl_seconds=self.settings.worker_heartbeat_ttl_seconds,
            )
            await heartbeat.write()
            self._heartbeat_task = asyncio.create_task(heartbeat.run(self._stop_event))
            self._heartbeat_task.add_done_callback(self._heartbeat_failed)
        logger.info(
            "worker_startup",
            requestId=None,
            traceId=None,
            taskName=None,
            taskVersion=None,
            schemaVersion=None,
            errorCode=None,
            latencyMs=None,
            registeredJobs=list(self.registry.jobs),
        )

    async def stop(self) -> None:
        self._stop_event.set()
        if self._heartbeat_task:
            await self._heartbeat_task
        if self._redis:
            await self._redis.aclose()
        logger.info(
            "worker_shutdown",
            requestId=None,
            traceId=None,
            taskName=None,
            taskVersion=None,
            schemaVersion=None,
            errorCode=None,
            latencyMs=None,
        )

    async def run(self) -> None:
        await self.start()
        await self._stop_event.wait()

    def _heartbeat_failed(self, task: asyncio.Task[None]) -> None:
        if task.cancelled() or self._stop_event.is_set():
            return
        error = task.exception()
        if error:
            capture_exception(
                error,
                source="worker-health",
                service="squadpitch-ai-worker",
                severity="critical",
                incident_type="redis-unavailable",
            )


async def run_worker(settings: Settings | None = None) -> None:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    init_sentry(resolved_settings, service="squadpitch-ai-worker")
    worker = Worker(resolved_settings)
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(signum, lambda: asyncio.create_task(worker.stop()))
    await worker.run()


def main() -> None:
    try:
        asyncio.run(run_worker())
    except Exception as exc:
        capture_exception(exc, operation="worker_main")
        raise


if __name__ == "__main__":
    main()

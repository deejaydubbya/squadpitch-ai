import asyncio
import signal
from contextlib import suppress
from dataclasses import dataclass

import structlog

from squadpitch_ai.core.config import Settings, get_settings
from squadpitch_ai.observability.logging import configure_logging
from squadpitch_ai.observability.sentry import capture_exception, init_sentry

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class JobRegistry:
    jobs: tuple[str, ...] = ()


class Worker:
    def __init__(self, settings: Settings, registry: JobRegistry | None = None) -> None:
        self.settings = settings
        self.registry = registry or JobRegistry()
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
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

from dataclasses import dataclass
from typing import Protocol

from redis.asyncio import Redis

from squadpitch_ai.core.config import Settings, get_settings


class HealthDependency(Protocol):
    name: str

    async def ready(self) -> bool: ...


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    ready: bool


class DependencyRegistry:
    def __init__(self, dependencies: list[HealthDependency] | None = None) -> None:
        self._dependencies = dependencies or []

    async def readiness(self) -> list[DependencyStatus]:
        statuses: list[DependencyStatus] = []
        for dependency in self._dependencies:
            statuses.append(
                DependencyStatus(name=dependency.name, ready=await dependency.ready()),
            )
        return statuses

    async def is_ready(self) -> bool:
        return all(status.ready for status in await self.readiness())


class ConfigurationDependency:
    name = "configuration"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def ready(self) -> bool:
        return self._settings.app_env != "production" or bool(
            self._settings.service_auth_secrets_by_key_id
        )


class RedisDependency:
    name = "redis"

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url

    async def ready(self) -> bool:
        client = Redis.from_url(self._redis_url, socket_connect_timeout=2, socket_timeout=2)
        try:
            return bool(await client.ping())
        except Exception:  # dependency failures are readiness state, not process crashes
            return False
        finally:
            await client.aclose()


def build_dependency_registry(settings: Settings | None = None) -> DependencyRegistry:
    resolved = settings or get_settings()
    dependencies: list[HealthDependency] = [ConfigurationDependency(resolved)]
    if resolved.redis_url:
        dependencies.append(RedisDependency(resolved.redis_url))
    return DependencyRegistry(dependencies)

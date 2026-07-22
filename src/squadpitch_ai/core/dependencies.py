from dataclasses import dataclass
from typing import Protocol


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


def build_dependency_registry() -> DependencyRegistry:
    return DependencyRegistry()

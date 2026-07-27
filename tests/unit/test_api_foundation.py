import socket

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from squadpitch_ai.api.app import create_app, create_dual_stack_socket
from squadpitch_ai.core.config import Settings
from squadpitch_ai.core.dependencies import DependencyRegistry


class ReadyDependency:
    name = "ready_dependency"

    async def ready(self) -> bool:
        return True


class NotReadyDependency:
    name = "not_ready_dependency"

    async def ready(self) -> bool:
        return False


def test_health() -> None:
    app = create_app(settings=Settings(app_env="test"))
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "squadpitch-ai"}


def test_dual_stack_socket_enables_ipv4_mapped_connections() -> None:
    server_socket = create_dual_stack_socket(0)
    try:
        assert server_socket.family == socket.AF_INET6
        assert server_socket.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY) == 0
    finally:
        server_socket.close()


def test_ready_with_no_required_dependencies() -> None:
    app = create_app(settings=Settings(app_env="test"))
    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "dependencies": []}


def test_ready_reports_dependency_status() -> None:
    app = create_app(
        settings=Settings(app_env="test"),
        registry=DependencyRegistry([ReadyDependency(), NotReadyDependency()]),
    )
    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "not_ready",
        "dependencies": [
            {"name": "ready_dependency", "ready": True},
            {"name": "not_ready_dependency", "ready": False},
        ],
    }


def test_error_envelope() -> None:
    app = create_app(settings=Settings(app_env="test"))
    with TestClient(app) as client:
        response = client.get("/__test/error", headers={"x-request-id": "req-1"})

    assert response.status_code == 418
    assert response.json() == {
        "code": "INTERNAL_ERROR",
        "message": "Test error",
        "retryable": False,
        "requestId": "req-1",
        "traceId": "req-1",
    }


def test_not_found_uses_error_envelope() -> None:
    app = create_app(settings=Settings(app_env="test"))
    with TestClient(app) as client:
        response = client.get("/missing", headers={"x-request-id": "req-404"})

    assert response.status_code == 404
    assert response.json()["requestId"] == "req-404"
    assert response.json()["code"] == "INTERNAL_ERROR"


def test_request_and_trace_id_propagation() -> None:
    app = create_app(settings=Settings(app_env="test"))
    with TestClient(app) as client:
        response = client.get(
            "/health",
            headers={"x-request-id": "req-1", "x-trace-id": "trace-1"},
        )

    assert response.headers["x-request-id"] == "req-1"
    assert response.headers["x-trace-id"] == "trace-1"


def test_config_failure() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"app_env": "invalid"})

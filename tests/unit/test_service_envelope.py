import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from squadpitch_ai.api.app import create_app
from squadpitch_ai.contracts.errors import ErrorCode
from squadpitch_ai.contracts.service_envelope import (
    SCHEMA_VERSION,
    AiServiceScope,
    BoundedNonceStore,
    ServiceAuthError,
    ServiceEnvelope,
    canonicalize_envelope,
    sign_envelope,
    verify_service_envelope,
)
from squadpitch_ai.core.config import Settings

SECRET_V1 = "node-python-service-secret-v1"
SECRET_V2 = "node-python-service-secret-v2"
NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def make_envelope(
    *,
    secret: str = SECRET_V1,
    key_id: str = "v1",
    scopes: list[str] | None = None,
    issued_at: datetime = NOW,
    expires_at: datetime | None = None,
    nonce: str = "nonce-1234567890abcdef",
    workspace_id: str = "workspace-1",
    payload_workspace_id: str = "workspace-1",
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schemaVersion": schema_version,
        "requestId": "req-1",
        "traceId": "trace-1",
        "workspaceId": workspace_id,
        "actorUserId": "user-1",
        "scopes": scopes or ["eval:run"],
        "issuedAt": issued_at.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "expiresAt": (expires_at or issued_at + timedelta(seconds=60)).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        ),
        "nonce": nonce,
        "payload": {"workspaceId": payload_workspace_id, "task": "noop"},
        "signature": {
            "keyId": key_id,
            "algorithm": "HMAC-SHA256",
            "signature": "0" * 64,
        },
    }
    body["signature"]["signature"] = sign_envelope(body, secret)  # type: ignore[index]
    return body


def verify(
    body: dict[str, object], *, scope: AiServiceScope = AiServiceScope.EVAL_RUN
) -> ServiceEnvelope:
    return verify_service_envelope(
        ServiceEnvelope.model_validate(body),
        secrets_by_key_id={"v1": SECRET_V1, "v2": SECRET_V2},
        nonce_store=BoundedNonceStore(),
        required_scope=scope,
        now=NOW,
    )


def test_valid_signature() -> None:
    envelope = verify(make_envelope())

    assert envelope.request_id == "req-1"


def test_invalid_signature() -> None:
    body = make_envelope()
    body["payload"] = {"workspaceId": "workspace-1", "task": "tampered"}

    with pytest.raises(ServiceAuthError) as exc:
        verify(body)

    assert exc.value.code == ErrorCode.AUTH_SIGNATURE_INVALID


def test_missing_signature() -> None:
    app = create_app(
        settings=Settings(app_env="test", service_auth_secrets="v1:node-python-service-secret-v1"),
    )
    body = make_envelope(scopes=["health:read"])
    del body["signature"]

    with TestClient(app) as client:
        response = client.post("/v1/health/check", json=body)

    assert response.status_code == 401
    assert response.json()["code"] == "AUTH_SIGNATURE_MISSING"


def test_expired_request() -> None:
    with pytest.raises(ServiceAuthError) as exc:
        verify(
            make_envelope(
                issued_at=NOW - timedelta(minutes=2),
                expires_at=NOW - timedelta(seconds=1),
            ),
        )

    assert exc.value.code == ErrorCode.AUTH_REQUEST_EXPIRED


def test_future_dated_request() -> None:
    with pytest.raises(ServiceAuthError) as exc:
        verify(make_envelope(issued_at=NOW + timedelta(seconds=31)))

    assert exc.value.code == ErrorCode.AUTH_REQUEST_FUTURE_DATED


def test_replayed_nonce() -> None:
    store = BoundedNonceStore(now_func=lambda: NOW.timestamp())
    envelope = ServiceEnvelope.model_validate(make_envelope())

    verify_service_envelope(
        envelope,
        secrets_by_key_id={"v1": SECRET_V1},
        nonce_store=store,
        required_scope=AiServiceScope.EVAL_RUN,
        now=NOW,
    )
    with pytest.raises(ServiceAuthError) as exc:
        verify_service_envelope(
            envelope,
            secrets_by_key_id={"v1": SECRET_V1},
            nonce_store=store,
            required_scope=AiServiceScope.EVAL_RUN,
            now=NOW,
        )

    assert exc.value.code == ErrorCode.AUTH_NONCE_REPLAYED


def test_wrong_scope() -> None:
    with pytest.raises(ServiceAuthError) as exc:
        verify(make_envelope(scopes=["health:read"]))

    assert exc.value.code == ErrorCode.AUTH_SCOPE_DENIED


def test_cross_workspace_payload_mismatch() -> None:
    with pytest.raises(ServiceAuthError) as exc:
        verify(make_envelope(payload_workspace_id="workspace-2"))

    assert exc.value.code == ErrorCode.CONTRACT_WORKSPACE_MISMATCH


def test_unsupported_schema_version_maps_to_stable_error() -> None:
    app = create_app(
        settings=Settings(app_env="test", service_auth_secrets="v1:node-python-service-secret-v1"),
    )
    body = make_envelope(schema_version="ai-service-envelope.v2")

    with TestClient(app) as client:
        response = client.post("/v1/eval/run", json=body)

    assert response.status_code == 422
    assert response.json()["code"] == "CONTRACT_UNSUPPORTED_SCHEMA_VERSION"


def test_key_rotation_compatibility() -> None:
    v1 = verify(make_envelope(key_id="v1", secret=SECRET_V1, nonce="nonce-1234567890abcdef"))
    v2 = verify(make_envelope(key_id="v2", secret=SECRET_V2, nonce="nonce-2234567890abcdef"))

    assert v1.signature.key_id == "v1"
    assert v2.signature.key_id == "v2"


def test_stable_error_mapping_has_no_stack_trace() -> None:
    app = create_app(
        settings=Settings(app_env="test", service_auth_secrets="v1:node-python-service-secret-v1"),
    )
    body = make_envelope(scopes=["health:read"], issued_at=datetime.now(tz=UTC))
    body["signature"]["signature"] = "f" * 64  # type: ignore[index]

    with TestClient(app) as client:
        response = client.post("/v1/health/check", json=body)

    assert response.status_code == 401
    assert response.json() == {
        "code": "AUTH_SIGNATURE_INVALID",
        "message": "Invalid request signature",
        "retryable": False,
        "requestId": "req-1",
        "traceId": "trace-1",
        "schemaVersion": SCHEMA_VERSION,
    }


def test_node_python_contract_compatibility() -> None:
    envelope = ServiceEnvelope.model_validate(
        make_envelope(payload_workspace_id="workspace-1"),
    )

    assert canonicalize_envelope(envelope) == (
        '{"actorUserId":"user-1","expiresAt":"2026-07-22T12:01:00.000Z",'
        '"issuedAt":"2026-07-22T12:00:00.000Z","nonce":"nonce-1234567890abcdef",'
        '"payload":{"task":"noop","workspaceId":"workspace-1"},"requestId":"req-1",'
        '"schemaVersion":"ai-service-envelope.v1","scopes":["eval:run"],'
        '"traceId":"trace-1","workspaceId":"workspace-1"}'
    )


def test_schema_scope_parity_with_checked_in_source_of_truth() -> None:
    schema_path = (
        Path(__file__).parents[3]
        / "docs"
        / "ai-platform"
        / "contracts"
        / "v1"
        / "service-envelope.schema.json"
    )
    if not schema_path.exists():
        pytest.skip("monorepo contract schema is not present in this checkout")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["properties"]["schemaVersion"]["const"] == SCHEMA_VERSION
    assert schema["properties"]["scopes"]["items"]["enum"] == [
        scope.value for scope in AiServiceScope
    ]

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from squadpitch_ai.api.app import create_app
from squadpitch_ai.brand_quality import BrandQualityScoreRequest
from squadpitch_ai.contracts.service_envelope import SCHEMA_VERSION, sign_envelope
from squadpitch_ai.core.config import Settings
from squadpitch_ai.model_registry import (
    ModelRegistryError,
    SelfHostedBrandQualityInference,
    benchmark_brand_quality_inference,
    build_default_model_registry,
    checksum_file,
    verify_artifact_integrity,
)

SECRET = "node-python-service-secret-v1"


def quality_request(**overrides: object) -> BrandQualityScoreRequest:
    payload: dict[str, object] = {
        "schemaVersion": "brand-content-quality.v1",
        "workspaceId": "workspace-1",
        "contentId": "content-1",
        "sanitizedText": "Professional local listing update.",
        "channel": "INSTAGRAM",
        "industry": "real_estate",
        "brandConstraints": ["professional", "local"],
        "bannedPhrases": ["guaranteed profit"],
        "language": "en",
        "modelVersion": "brand-quality-neural-shadow.v1",
        "traceId": "trace-quality",
    }
    payload.update(overrides)
    return BrandQualityScoreRequest.model_validate(payload)


def make_envelope(
    payload: dict[str, object],
    scopes: list[str] | None = None,
    nonce: str = "nonce-model-registry-health",
) -> dict[str, object]:
    now = datetime.now(tz=UTC)
    body: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "requestId": "req-registry",
        "traceId": "trace-registry",
        "workspaceId": "workspace-1",
        "actorUserId": "user-1",
        "scopes": scopes or ["health:read"],
        "issuedAt": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "expiresAt": (now + timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "nonce": nonce,
        "payload": {"probe": "model-registry"},
        "signature": {
            "keyId": "v1",
            "algorithm": "HMAC-SHA256",
            "signature": "0" * 64,
        },
    }
    body["payload"] = payload
    body["signature"]["signature"] = sign_envelope(body, SECRET)  # type: ignore[index]
    return body


def test_artifact_checksum_and_integrity_verification(tmp_path: Path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"tiny-local-artifact")

    assert checksum_file(artifact) == checksum_file(artifact)
    entry = build_default_model_registry().get(
        "brand-content-quality",
        "brand-quality-neural-shadow.v1",
    )
    assert verify_artifact_integrity(entry) is True

    broken = entry.model_copy(update={"artifact_checksum_sha256": "0" * 64})
    with pytest.raises(ModelRegistryError, match="Checksum mismatch"):
        verify_artifact_integrity(broken)


def test_registry_rejects_missing_version_and_incompatible_schema() -> None:
    registry = build_default_model_registry()
    with pytest.raises(ModelRegistryError, match="Model not found"):
        registry.get("brand-content-quality", "missing")
    with pytest.raises(ModelRegistryError, match="not compatible"):
        registry.require_compatible(
            "brand-content-quality",
            "brand-quality-neural-shadow.v1",
            "future-schema.v1",
        )


def test_inference_cold_start_warmup_and_metrics_emission() -> None:
    inference = SelfHostedBrandQualityInference()

    response, metric = inference.predict(quality_request())
    second_response, second_metric = inference.predict(quality_request(contentId="content-2"))

    assert response.proposal_only is True
    assert second_response.content_id == "content-2"
    assert metric.cold_start is True
    assert second_metric.cold_start is False
    assert inference.metrics[-1].status == "succeeded"


def test_benchmark_reports_latency_throughput_cost_and_failure_behavior() -> None:
    report = benchmark_brand_quality_inference(
        SelfHostedBrandQualityInference(),
        quality_request(),
        iterations=3,
    )

    assert cast(float, report["p50LatencyMs"]) >= 0
    assert cast(float, report["p95LatencyMs"]) >= 0
    assert cast(float, report["throughputPerSecond"]) > 0
    assert report["batchSize"] == 1
    assert report["failureBehavior"] == "fallback_to_deterministic_shadow"


def test_signed_registry_health_requires_health_scope() -> None:
    app = create_app(settings=Settings(app_env="test", service_auth_secrets=f"v1:{SECRET}"))

    with TestClient(app) as client:
        denied = client.post(
            "/v1/models/registry/health",
            json=make_envelope(
                {"probe": "model-registry"},
                scopes=["content-score:read"],
                nonce="nonce-model-registry-denied",
            ),
        )
        response = client.post(
            "/v1/models/registry/health",
            json=make_envelope({"probe": "model-registry"}, nonce="nonce-model-registry-ok"),
        )

    assert denied.status_code == 403
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["version"] == "brand-quality-neural-shadow.v1"

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from squadpitch_ai.api.app import create_app
from squadpitch_ai.contracts.service_envelope import SCHEMA_VERSION, sign_envelope
from squadpitch_ai.core.config import Settings
from squadpitch_ai.experimentation import ExperimentAnalysisRequest, analyze_experiment

SECRET = "node-python-service-secret-v1"
NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def definition(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "experimentId": "exp-hook-v1",
        "workspaceId": "workspace-1",
        "name": "Hook experiment",
        "hypothesis": "Local proof hooks improve approval rate.",
        "primaryMetric": "approved",
        "metricType": "proportion",
        "guardrailMetrics": ["provider_failure_rate"],
        "eligibilityRules": {"channels": ["INSTAGRAM"]},
        "variants": [
            {"key": "control", "label": "Current hook", "allocation": 0.5, "isControl": True},
            {
                "key": "treatment",
                "label": "Local proof hook",
                "allocation": 0.5,
                "isControl": False,
            },
        ],
        "attributionWindowHours": 24,
        "sampleSizeGuidance": {"minimumPerVariant": 30, "power": 0.8},
        "stoppingRules": ["Do not stop before minimum sample unless guardrail fails."],
        "analysisPlan": "Compare approval proportions by variant with 95% confidence intervals.",
        "segments": ["channel"],
        "status": "running",
    }
    data.update(overrides)
    return data


def exposure(exposure_id: str, variant: str, entity: str, **overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "exposureId": exposure_id,
        "experimentId": "exp-hook-v1",
        "workspaceId": "workspace-1",
        "subjectId": f"subject-{entity}",
        "variantKey": variant,
        "exposedAt": NOW.isoformat(),
        "entityType": "draft",
        "entityId": entity,
        "segments": {"channel": "INSTAGRAM"},
    }
    data.update(overrides)
    return data


def outcome(
    outcome_id: str, entity: str, metric: str, value: float, observed_at: datetime
) -> dict[str, object]:
    return {
        "outcomeId": outcome_id,
        "experimentId": "exp-hook-v1",
        "workspaceId": "workspace-1",
        "entityType": "draft",
        "entityId": entity,
        "metric": metric,
        "value": value,
        "observedAt": observed_at.isoformat(),
    }


def request_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schemaVersion": "experiment-analysis.v1",
        "definition": definition(),
        "exposures": [
            exposure("e1", "control", "d1"),
            exposure("e2", "control", "d2"),
            exposure("e3", "treatment", "d3"),
            exposure("e4", "treatment", "d4"),
        ],
        "outcomes": [
            outcome("o1", "d1", "approved", 0, NOW + timedelta(hours=2)),
            outcome("o2", "d2", "approved", 1, NOW + timedelta(hours=2)),
            outcome("o3", "d3", "approved", 1, NOW + timedelta(hours=2)),
            outcome("o4", "d4", "approved", 1, NOW + timedelta(hours=25)),
            outcome("g1", "d3", "provider_failure_rate", 0.1, NOW + timedelta(hours=2)),
        ],
        "traceId": "trace-exp",
    }
    payload.update(overrides)
    return payload


def make_envelope(payload: dict[str, object], scopes: list[str] | None = None) -> dict[str, object]:
    issued_at = datetime.now(tz=UTC)
    body: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "requestId": "req-exp",
        "traceId": "trace-exp",
        "workspaceId": "workspace-1",
        "actorUserId": "user-1",
        "scopes": scopes or ["eval:run"],
        "issuedAt": issued_at.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "expiresAt": (issued_at + timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "nonce": f"nonce-experiment-analysis-{'-'.join(scopes or ['eval:run'])}",
        "payload": payload,
        "signature": {"keyId": "v1", "algorithm": "HMAC-SHA256", "signature": "0" * 64},
    }
    body["signature"]["signature"] = sign_envelope(body, SECRET)  # type: ignore[index]
    return body


def test_metric_calculation_confidence_interval_missing_data_and_warnings() -> None:
    report = analyze_experiment(ExperimentAnalysisRequest.model_validate(request_payload()))

    control = next(item for item in report.variant_reports if item.variant_key == "control")
    treatment = next(item for item in report.variant_reports if item.variant_key == "treatment")

    assert control.mean == 0.5
    assert treatment.mean == 1.0
    assert treatment.missing_count == 1
    assert report.effects[0].absolute_effect == 0.5
    assert report.effects[0].confidence_interval[0] <= report.effects[0].absolute_effect
    assert any("minimum guidance" in warning for warning in report.warnings)
    assert any("missing primary outcomes" in warning for warning in report.warnings)
    guardrail = cast(dict[str, Any], report.guardrails["provider_failure_rate"])
    segment_report = cast(dict[str, Any], report.segment_reports["channel"])
    assert guardrail["rollbackRecommended"] is True
    assert "INSTAGRAM" in segment_report


def test_attribution_window_boundaries_and_cross_workspace_rejection() -> None:
    report = analyze_experiment(ExperimentAnalysisRequest.model_validate(request_payload()))
    treatment = next(item for item in report.variant_reports if item.variant_key == "treatment")
    assert treatment.observed_count == 1

    bad_exposures = [exposure("e-bad", "control", "d-bad", workspaceId="workspace-2")]
    with pytest.raises(ValueError, match="cross-workspace"):
        analyze_experiment(
            ExperimentAnalysisRequest.model_validate(request_payload(exposures=bad_exposures))
        )


def test_signed_endpoint_requires_eval_scope_and_returns_report() -> None:
    app = create_app(settings=Settings(app_env="test", service_auth_secrets=f"v1:{SECRET}"))

    with TestClient(app) as client:
        denied = client.post(
            "/v1/experiments/analyze",
            json=make_envelope(request_payload(), scopes=["health:read"]),
        )
        response = client.post("/v1/experiments/analyze", json=make_envelope(request_payload()))

    assert denied.status_code == 403
    assert response.status_code == 200
    body = response.json()
    assert body["schemaVersion"] == "experiment-analysis.v1"
    assert body["causalityCaution"].startswith("Randomized")

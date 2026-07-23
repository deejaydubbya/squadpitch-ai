from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from squadpitch_ai.api.app import create_app
from squadpitch_ai.brand_quality import (
    BrandQualityScoreRequest,
    build_brand_quality_dataset,
    score_brand_quality,
)
from squadpitch_ai.brand_quality.baselines import compare_baselines
from squadpitch_ai.brand_quality.pytorch_training import train_pytorch_quality_model
from squadpitch_ai.contracts.service_envelope import SCHEMA_VERSION, sign_envelope
from squadpitch_ai.core.config import Settings

SECRET = "node-python-service-secret-v1"
NOW = datetime.now(tz=UTC)


def row(example_id: str, created_at: str, **overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "exampleId": example_id,
        "workspaceId": "workspace-1",
        "industry": "real_estate",
        "channel": "INSTAGRAM",
        "sanitizedText": "Professional local listing update with concise neighborhood context.",
        "brandConstraints": ["professional", "local"],
        "bannedPhrases": ["guaranteed profit"],
        "accepted": True,
        "corrected": False,
        "labels": {
            "brand_voice_match": 1,
            "excessive_promotion": 0,
            "excessive_verbosity": 0,
            "prohibited_phrase": 0,
            "channel_suitability": 1,
            "unsupported_language_risk": 0,
            "needs_human_review": 0,
        },
        "createdAt": created_at,
    }
    data.update(overrides)
    return data


def quality_request(**overrides: object) -> BrandQualityScoreRequest:
    payload: dict[str, object] = {
        "schemaVersion": "brand-content-quality.v1",
        "workspaceId": "workspace-1",
        "contentId": "content-1",
        "sanitizedText": "This exclusive offer has guaranteed profit and must be reviewed.",
        "channel": "X",
        "industry": "real_estate",
        "brandConstraints": ["local", "professional"],
        "bannedPhrases": ["guaranteed profit"],
        "language": "fr",
        "modelVersion": "brand-quality-neural-shadow.v1",
        "traceId": "trace-quality",
    }
    payload.update(overrides)
    return BrandQualityScoreRequest.model_validate(payload)


def make_envelope(payload: dict[str, object], scopes: list[str] | None = None) -> dict[str, object]:
    body: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "requestId": "req-quality",
        "traceId": "trace-quality",
        "workspaceId": "workspace-1",
        "actorUserId": "user-1",
        "scopes": scopes or ["content-score:read"],
        "issuedAt": NOW.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "expiresAt": (NOW + timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "nonce": "nonce-brand-quality-score",
        "payload": payload,
        "signature": {
            "keyId": "v1",
            "algorithm": "HMAC-SHA256",
            "signature": "0" * 64,
        },
    }
    body["signature"]["signature"] = sign_envelope(body, SECRET)  # type: ignore[index]
    return body


def dataset_rows() -> list[dict[str, object]]:
    return [
        row("ex-1", "2026-07-01T10:00:00Z"),
        row("ex-2", "2026-07-02T10:00:00Z", accepted=False, labels={"needs_human_review": 1}),
        row("ex-3", "2026-07-03T10:00:00Z", corrected=True),
        row("ex-4", "2026-07-04T10:00:00Z"),
    ]


def test_dataset_card_inputs_are_reproducible_balanced_and_leakage_checked() -> None:
    first = build_brand_quality_dataset(dataset_rows())
    second = build_brand_quality_dataset(list(reversed(dataset_rows())))

    assert first.dataset_version == second.dataset_version
    assert first.schema_version == "brand-quality-dataset.v1"
    assert first.leakage_controls["rawPrompts"] is False
    assert "needs_human_review" in first.class_balance

    with pytest.raises(ValueError, match="leakage fields"):
        build_brand_quality_dataset([row("bad", "2026-07-05T10:00:00Z", rawOutput="unsafe")])
    with pytest.raises(ValueError, match="sanitized"):
        build_brand_quality_dataset(
            [row("pii", "2026-07-05T10:00:00Z", sanitizedText="lead@example.com")]
        )


def test_baseline_spec_covers_required_model_families() -> None:
    baselines = compare_baselines(build_brand_quality_dataset(dataset_rows()))

    assert "tfidfLogisticRegression" in baselines
    assert "embeddingsLinearClassifier" in baselines
    assert "pytorchNeuralClassifier" in baselines
    assert "smallTransformerFineTune" in baselines
    assert "loraPeft" in baselines


def test_score_flags_banned_language_and_human_review_without_actions() -> None:
    result = score_brand_quality(quality_request())

    assert result.proposal_only is True
    assert result.needs_human_review is True
    assert result.model_family == "deterministic_shadow"
    assert {"prohibited_phrase", "unsupported_language_risk", "needs_human_review"} <= set(
        result.categories
    )


def test_score_rejects_model_version_mismatch() -> None:
    with pytest.raises(ValueError, match="unsupported model version"):
        score_brand_quality(quality_request(modelVersion="future-model"))


def test_signed_endpoint_requires_content_score_scope_and_returns_proposal_only_score() -> None:
    app = create_app(settings=Settings(app_env="test", service_auth_secrets=f"v1:{SECRET}"))
    payload = quality_request().model_dump(mode="json", by_alias=True)

    with TestClient(app) as client:
        denied = client.post(
            "/v1/content-quality/score",
            json=make_envelope(payload, scopes=["health:read"]),
        )
        response = client.post("/v1/content-quality/score", json=make_envelope(payload))

    assert denied.status_code == 403
    assert response.status_code == 200
    body = response.json()
    assert body["schemaVersion"] == "brand-content-quality.v1"
    assert body["proposalOnly"] is True
    assert body["needsHumanReview"] is True


def test_pytorch_training_requires_optional_ml_extra_when_torch_absent(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed in this environment")

    with pytest.raises(RuntimeError, match="PyTorch is not installed"):
        train_pytorch_quality_model(build_brand_quality_dataset(dataset_rows()), tmp_path)

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from squadpitch_ai.api.app import create_app
from squadpitch_ai.autopilot_ranker import (
    AutopilotRankingRequest,
    build_training_dataset,
    rank_autopilot_opportunities,
    split_dataset,
)
from squadpitch_ai.autopilot_ranker.features import assert_no_target_leakage, vectorize_candidate
from squadpitch_ai.contracts.service_envelope import SCHEMA_VERSION, sign_envelope
from squadpitch_ai.core.config import Settings

SECRET = "node-python-service-secret-v1"
NOW = datetime.now(tz=UTC)


def candidate(
    candidate_id: str,
    *,
    workspace_id: str = "workspace-1",
    **overrides: object,
) -> dict[str, object]:
    data: dict[str, object] = {
        "candidateId": candidate_id,
        "workspaceId": workspace_id,
        "triggerType": "NEW_LISTING",
        "industry": "real_estate",
        "channel": "INSTAGRAM",
        "heuristicScore": 0.5,
        "listingAgeDays": 3,
        "priceChangePercent": 0,
        "daysSinceLastPost": 10,
        "mediaAvailable": True,
        "historicalApprovalRate": 0.75,
        "historicalEngagementRate": 0.03,
        "hourOfDay": 10,
        "dayOfWeek": 2,
        "contentType": "listing",
        "recentAudienceEngagement": 0.04,
        "detectedAt": "2026-07-22T10:00:00Z",
    }
    data.update(overrides)
    return data


def ranking_request(**overrides: object) -> AutopilotRankingRequest:
    payload = {
        "schemaVersion": "autopilot-opportunity-ranking.v1",
        "workspaceId": "workspace-1",
        "candidates": [
            candidate("c1", heuristicScore=0.2, mediaAvailable=False),
            candidate("c2", heuristicScore=0.8, priceChangePercent=5),
        ],
        "modelVersion": "autopilot-logistic-ranker.v1",
        "shadowMode": True,
        "traceId": "trace-rank",
    }
    payload.update(overrides)
    return AutopilotRankingRequest.model_validate(payload)


def make_envelope(payload: dict[str, object], scopes: list[str] | None = None) -> dict[str, object]:
    body: dict[str, object] = {
        "schemaVersion": SCHEMA_VERSION,
        "requestId": "req-rank",
        "traceId": "trace-rank",
        "workspaceId": "workspace-1",
        "actorUserId": "user-1",
        "scopes": scopes or ["autopilot-rank:read"],
        "issuedAt": NOW.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "expiresAt": (NOW + timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "nonce": "nonce-autopilot-rank",
        "payload": payload,
        "signature": {
            "keyId": "v1",
            "algorithm": "HMAC-SHA256",
            "signature": "0" * 64,
        },
    }
    body["signature"]["signature"] = sign_envelope(body, SECRET)  # type: ignore[index]
    return body


def test_dataset_correctness_reproducible_and_time_split() -> None:
    rows = [
        {"candidate": candidate("c1", detectedAt="2026-07-01T10:00:00Z"), "label": "APPROVED"},
        {"candidate": candidate("c2", detectedAt="2026-07-02T10:00:00Z"), "label": "DISMISSED"},
        {"candidate": candidate("c3", detectedAt="2026-07-03T10:00:00Z"), "label": "PUBLISHED"},
        {"candidate": candidate("c4", detectedAt="2026-07-04T10:00:00Z"), "label": "UNKNOWN"},
    ]

    first = build_training_dataset(rows)
    second = build_training_dataset(list(reversed(rows)))
    splits = split_dataset(first)

    assert first.dataset_version == second.dataset_version
    assert len(first.examples) == 3
    assert [example.label for example in first.examples] == [1, 0, 1]
    assert sum(len(values) for values in splits.values()) == 3
    assert first.privacy["rawPrompts"] is False


def test_target_leakage_detection_and_missing_feature_policy() -> None:
    with pytest.raises(ValueError, match="target leakage"):
        assert_no_target_leakage(["heuristic_score", "published"])

    parsed = ranking_request(
        candidates=[candidate("missing", listingAgeDays=None, daysSinceLastPost=None)]
    ).candidates[0]
    _features, missing = vectorize_candidate(parsed)

    assert {"listingAgeDays", "daysSinceLastPost"} <= set(missing)


def test_ranker_is_reproducible_and_proposal_only() -> None:
    request = ranking_request()

    first = rank_autopilot_opportunities(request)
    second = rank_autopilot_opportunities(request)

    assert first == second
    assert first.proposal_only is True
    assert first.model_metadata["modelType"] == "logistic_regression"
    assert {ranked.candidate_id for ranked in first.ranked_candidates} == {"c1", "c2"}


def test_ranker_rejects_cross_workspace_and_version_mismatch() -> None:
    with pytest.raises(ValueError, match="workspace"):
        rank_autopilot_opportunities(
            ranking_request(candidates=[candidate("bad", workspace_id="workspace-2")])
        )

    with pytest.raises(ValueError, match="unsupported model version"):
        rank_autopilot_opportunities(ranking_request(modelVersion="future-model"))


def test_signed_autopilot_rank_endpoint_requires_scope_and_returns_rankings() -> None:
    app = create_app(settings=Settings(app_env="test", service_auth_secrets=f"v1:{SECRET}"))
    payload = ranking_request().model_dump(mode="json", by_alias=True)

    with TestClient(app) as client:
        denied = client.post(
            "/v1/autopilot/rank",
            json=make_envelope(payload, scopes=["health:read"]),
        )
        response = client.post("/v1/autopilot/rank", json=make_envelope(payload))

    assert denied.status_code == 403
    assert response.status_code == 200
    body = response.json()
    assert body["schemaVersion"] == "autopilot-opportunity-ranking.v1"
    assert body["proposalOnly"] is True
    assert body["shadowMode"] is True
    assert body["provenance"]["inferenceMode"] == "logistic_regression"
    assert body["provenance"]["modelVersion"] == "autopilot-logistic-ranker.v1"

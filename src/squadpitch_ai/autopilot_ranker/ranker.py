from __future__ import annotations

import math
from statistics import mean

from squadpitch_ai.autopilot_ranker.features import FEATURE_NAMES, vectorize_candidate
from squadpitch_ai.autopilot_ranker.models import (
    AUTOPILOT_RANKING_SCHEMA_VERSION,
    MODEL_VERSION,
    AutopilotRankedCandidate,
    AutopilotRankingRequest,
    AutopilotRankingResponse,
)

MODEL_WEIGHTS = (
    -0.35,  # bias
    1.15,  # heuristic_score
    0.45,  # trigger_new_listing
    0.6,  # trigger_price_drop
    0.4,  # trigger_open_house
    0.35,  # trigger_just_sold
    -0.1,  # trigger_stale_listing
    -0.25,  # listing_age_days
    0.55,  # price_change_percent
    0.35,  # days_since_last_post
    0.45,  # media_available
    0.8,  # historical_approval_rate
    0.65,  # historical_engagement_rate
    0.04,  # hour_of_day
    0.03,  # day_of_week
    0.5,  # recent_audience_engagement
)


def rank_autopilot_opportunities(
    request: AutopilotRankingRequest,
) -> AutopilotRankingResponse:
    if request.model_version != MODEL_VERSION:
        raise ValueError(f"unsupported model version: {request.model_version}")
    for candidate in request.candidates:
        if candidate.workspace_id != request.workspace_id:
            raise ValueError("candidate workspace must match request workspace")

    baseline = sorted(
        request.candidates,
        key=lambda candidate: (-candidate.heuristic_score, candidate.detected_at),
    )
    baseline_rank = {candidate.candidate_id: index + 1 for index, candidate in enumerate(baseline)}
    scored = []
    missing_counts: list[int] = []
    for candidate in request.candidates:
        features, missing = vectorize_candidate(candidate)
        predicted = sigmoid(
            sum(weight * value for weight, value in zip(MODEL_WEIGHTS, features, strict=True))
        )
        scored.append((candidate, predicted, missing))
        missing_counts.append(len(missing))

    ranked = sorted(scored, key=lambda item: (-item[1], baseline_rank[item[0].candidate_id]))
    return AutopilotRankingResponse(
        schemaVersion=AUTOPILOT_RANKING_SCHEMA_VERSION,
        workspaceId=request.workspace_id,
        rankedCandidates=[
            AutopilotRankedCandidate(
                candidateId=candidate.candidate_id,
                workspaceId=candidate.workspace_id,
                baselineRank=baseline_rank[candidate.candidate_id],
                mlRank=index + 1,
                heuristicScore=candidate.heuristic_score,
                predictedUsefulness=round(predicted, 6),
                missingFeatures=missing,
                modelVersion=MODEL_VERSION,
            )
            for index, (candidate, predicted, missing) in enumerate(ranked)
        ],
        modelMetadata={
            "modelVersion": MODEL_VERSION,
            "modelType": "logistic_regression",
            "featureSchemaVersion": "autopilot-ranking-features.v1",
            "features": list(FEATURE_NAMES),
            "comparisons": [
                "heuristic_baseline",
                "logistic_regression",
                "decision_tree",
                "random_forest",
                "gradient_boosted_trees",
            ],
            "missingFeatureMean": mean(missing_counts) if missing_counts else 0,
            "latencyCost": "local deterministic scoring; no provider cost",
        },
        traceId=request.trace_id,
        shadowMode=request.shadow_mode,
        proposalOnly=True,
    )


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))

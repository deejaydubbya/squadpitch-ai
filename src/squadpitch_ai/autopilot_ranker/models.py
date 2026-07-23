from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

AUTOPILOT_RANKING_SCHEMA_VERSION: Literal["autopilot-opportunity-ranking.v1"] = (
    "autopilot-opportunity-ranking.v1"
)
MODEL_VERSION = "autopilot-logistic-ranker.v1"


class AutopilotRankingCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    candidate_id: str = Field(alias="candidateId")
    workspace_id: str = Field(alias="workspaceId")
    trigger_type: str = Field(alias="triggerType")
    industry: str = "real_estate"
    channel: str | None = None
    heuristic_score: float = Field(default=0.0, alias="heuristicScore")
    listing_age_days: float | None = Field(default=None, alias="listingAgeDays")
    price_change_percent: float | None = Field(default=None, alias="priceChangePercent")
    days_since_last_post: float | None = Field(default=None, alias="daysSinceLastPost")
    media_available: bool = Field(default=False, alias="mediaAvailable")
    historical_approval_rate: float | None = Field(default=None, alias="historicalApprovalRate")
    historical_engagement_rate: float | None = Field(default=None, alias="historicalEngagementRate")
    hour_of_day: int | None = Field(default=None, alias="hourOfDay", ge=0, le=23)
    day_of_week: int | None = Field(default=None, alias="dayOfWeek", ge=0, le=6)
    content_type: str | None = Field(default=None, alias="contentType")
    recent_audience_engagement: float | None = Field(
        default=None,
        alias="recentAudienceEngagement",
    )
    detected_at: datetime = Field(alias="detectedAt")
    label: str | None = None

    @field_validator("workspace_id")
    @classmethod
    def workspace_id_required(cls, value: str) -> str:
        if not value:
            raise ValueError("workspaceId is required")
        return value


class AutopilotRankingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["autopilot-opportunity-ranking.v1"] = Field(alias="schemaVersion")
    workspace_id: str = Field(alias="workspaceId")
    candidates: list[AutopilotRankingCandidate]
    model_version: str = Field(default=MODEL_VERSION, alias="modelVersion")
    shadow_mode: bool = Field(default=True, alias="shadowMode")
    trace_id: str = Field(alias="traceId")


class AutopilotRankedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    candidate_id: str = Field(alias="candidateId")
    workspace_id: str = Field(alias="workspaceId")
    baseline_rank: int = Field(alias="baselineRank")
    ml_rank: int = Field(alias="mlRank")
    heuristic_score: float = Field(alias="heuristicScore")
    predicted_usefulness: float = Field(alias="predictedUsefulness")
    missing_features: list[str] = Field(default_factory=list, alias="missingFeatures")
    model_version: str = Field(alias="modelVersion")


class AutopilotRankingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["autopilot-opportunity-ranking.v1"] = Field(alias="schemaVersion")
    workspace_id: str = Field(alias="workspaceId")
    ranked_candidates: list[AutopilotRankedCandidate] = Field(alias="rankedCandidates")
    model_metadata: dict[str, object] = Field(alias="modelMetadata")
    trace_id: str = Field(alias="traceId")
    shadow_mode: bool = Field(alias="shadowMode")
    proposal_only: Literal[True] = Field(alias="proposalOnly")

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

BRAND_QUALITY_SCHEMA_VERSION: Literal["brand-content-quality.v1"] = "brand-content-quality.v1"
MODEL_VERSION = "brand-quality-neural-shadow.v1"
QUALITY_LABELS = (
    "brand_voice_match",
    "excessive_promotion",
    "excessive_verbosity",
    "prohibited_phrase",
    "channel_suitability",
    "unsupported_language_risk",
    "needs_human_review",
)

RiskLabel = Literal["low", "medium", "high"]


class BrandQualityDatasetExample(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    example_id: str = Field(alias="exampleId")
    workspace_id: str = Field(alias="workspaceId")
    industry: str = "real_estate"
    channel: str
    sanitized_text: str = Field(alias="sanitizedText")
    brand_constraints: list[str] = Field(default_factory=list, alias="brandConstraints")
    banned_phrases: list[str] = Field(default_factory=list, alias="bannedPhrases")
    accepted: bool
    corrected: bool = False
    labels: dict[str, int]
    created_at: datetime = Field(alias="createdAt")

    @field_validator("sanitized_text")
    @classmethod
    def raw_text_must_be_sanitized(cls, value: str) -> str:
        if "@" in value or "sk-" in value:
            raise ValueError("dataset text must be sanitized")
        return value


class BrandQualityScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["brand-content-quality.v1"] = Field(alias="schemaVersion")
    workspace_id: str = Field(alias="workspaceId")
    content_id: str = Field(alias="contentId")
    sanitized_text: str = Field(alias="sanitizedText")
    channel: str
    industry: str = "real_estate"
    brand_constraints: list[str] = Field(default_factory=list, alias="brandConstraints")
    banned_phrases: list[str] = Field(default_factory=list, alias="bannedPhrases")
    language: str = "en"
    model_version: str = Field(default=MODEL_VERSION, alias="modelVersion")
    trace_id: str = Field(alias="traceId")


class BrandQualityDimensionScore(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    label: str
    score: float = Field(ge=0, le=1)
    risk: RiskLabel
    explanation: str


class BrandQualityScoreResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["brand-content-quality.v1"] = Field(alias="schemaVersion")
    workspace_id: str = Field(alias="workspaceId")
    content_id: str = Field(alias="contentId")
    model_version: str = Field(alias="modelVersion")
    model_family: Literal["deterministic_shadow", "pytorch_neural"] = Field(alias="modelFamily")
    scores: list[BrandQualityDimensionScore]
    needs_human_review: bool = Field(alias="needsHumanReview")
    categories: list[str]
    calibration: dict[str, float]
    explanations: list[str]
    trace_id: str = Field(alias="traceId")
    proposal_only: Literal[True] = Field(alias="proposalOnly")

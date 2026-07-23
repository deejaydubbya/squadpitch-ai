from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EXPERIMENT_ANALYSIS_SCHEMA_VERSION: Literal["experiment-analysis.v1"] = "experiment-analysis.v1"


class ExperimentVariant(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    key: str
    label: str
    allocation: float = Field(ge=0, le=1)
    is_control: bool = Field(default=False, alias="isControl")


class ExperimentDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    experiment_id: str = Field(alias="experimentId")
    workspace_id: str = Field(alias="workspaceId")
    name: str
    hypothesis: str
    primary_metric: str = Field(alias="primaryMetric")
    metric_type: Literal["proportion", "mean"] = Field(alias="metricType")
    guardrail_metrics: list[str] = Field(default_factory=list, alias="guardrailMetrics")
    eligibility_rules: dict[str, object] = Field(default_factory=dict, alias="eligibilityRules")
    variants: list[ExperimentVariant]
    attribution_window_hours: int = Field(alias="attributionWindowHours", ge=1)
    sample_size_guidance: dict[str, object] = Field(alias="sampleSizeGuidance")
    stopping_rules: list[str] = Field(alias="stoppingRules")
    analysis_plan: str = Field(alias="analysisPlan")
    segments: list[str] = Field(default_factory=list)
    status: Literal["draft", "running", "paused", "rolled_back", "completed"] = "draft"

    @model_validator(mode="after")
    def exactly_one_control(self) -> ExperimentDefinition:
        if sum(1 for variant in self.variants if variant.is_control) != 1:
            raise ValueError("exactly one control variant is required")
        allocation = sum(variant.allocation for variant in self.variants)
        if round(allocation, 6) != 1:
            raise ValueError("variant allocations must sum to 1")
        return self


class ExperimentExposure(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    exposure_id: str = Field(alias="exposureId")
    experiment_id: str = Field(alias="experimentId")
    workspace_id: str = Field(alias="workspaceId")
    subject_id: str = Field(alias="subjectId")
    variant_key: str = Field(alias="variantKey")
    exposed_at: datetime = Field(alias="exposedAt")
    entity_type: str = Field(alias="entityType")
    entity_id: str = Field(alias="entityId")
    segments: dict[str, str] = Field(default_factory=dict)


class ExperimentOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    outcome_id: str = Field(alias="outcomeId")
    experiment_id: str = Field(alias="experimentId")
    workspace_id: str = Field(alias="workspaceId")
    entity_type: str = Field(alias="entityType")
    entity_id: str = Field(alias="entityId")
    metric: str
    value: float
    observed_at: datetime = Field(alias="observedAt")


class VariantMetricReport(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    variant_key: str = Field(alias="variantKey")
    sample_size: int = Field(alias="sampleSize")
    observed_count: int = Field(alias="observedCount")
    missing_count: int = Field(alias="missingCount")
    mean: float
    variance: float
    confidence_interval: tuple[float, float] = Field(alias="confidenceInterval")


class EffectReport(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    treatment_variant: str = Field(alias="treatmentVariant")
    control_variant: str = Field(alias="controlVariant")
    absolute_effect: float = Field(alias="absoluteEffect")
    relative_effect: float | None = Field(alias="relativeEffect")
    confidence_interval: tuple[float, float] = Field(alias="confidenceInterval")
    method: str


class ExperimentAnalysisReport(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["experiment-analysis.v1"] = Field(alias="schemaVersion")
    experiment_id: str = Field(alias="experimentId")
    workspace_id: str = Field(alias="workspaceId")
    primary_metric: str = Field(alias="primaryMetric")
    metric_type: Literal["proportion", "mean"] = Field(alias="metricType")
    variant_reports: list[VariantMetricReport] = Field(alias="variantReports")
    effects: list[EffectReport]
    guardrails: dict[str, object]
    segment_reports: dict[str, object] = Field(alias="segmentReports")
    warnings: list[str]
    missing_data_policy: str = Field(alias="missingDataPolicy")
    outlier_policy: str = Field(alias="outlierPolicy")
    multiple_comparison_caution: str = Field(alias="multipleComparisonCaution")
    sequential_testing_caution: str = Field(alias="sequentialTestingCaution")
    causality_caution: str = Field(alias="causalityCaution")
    calibration: dict[str, object]
    rollback_recommended: bool = Field(alias="rollbackRecommended")
    trace_id: str = Field(alias="traceId")


class ExperimentAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["experiment-analysis.v1"] = Field(alias="schemaVersion")
    definition: ExperimentDefinition
    exposures: list[ExperimentExposure]
    outcomes: list[ExperimentOutcome]
    trace_id: str = Field(alias="traceId")

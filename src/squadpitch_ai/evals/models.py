from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from squadpitch_ai.contracts.errors import ErrorCode

EVAL_SAMPLE_SCHEMA_VERSION = "ai-eval-sample-v1"

TASK_NAMES = (
    "content_draft_generation",
    "remix",
    "campaign_generation",
    "campaign_post_regeneration",
    "sites_page_generation",
    "sites_translation",
    "inbox_reply_suggestion",
    "ads_package_generation",
    "autopilot",
    "ideas",
    "inline_ai_action",
    "vision_auto_tagging",
    "real_estate_factual_safety",
    "prompt_injection",
    "tenant_isolation",
    "provider_failure_behavior",
)

BLOCKER_CODES = {
    ErrorCode.PROVIDER_NOT_CONFIGURED,
    ErrorCode.SCHEMA_INVALID,
    ErrorCode.FABRICATED_CRITICAL_FACT,
    ErrorCode.PROMPT_INJECTION_FOLLOWED,
    ErrorCode.TENANT_LEAKAGE,
    ErrorCode.RAW_SECRET_OR_PII_OUTPUT,
    ErrorCode.UNSAFE_POLICY_CONTENT,
    ErrorCode.UNINTENDED_SIDE_EFFECT,
}

HIGH_CODES = {
    ErrorCode.PROVIDER_UNAVAILABLE,
    ErrorCode.PROVIDER_TIMEOUT,
    ErrorCode.PROVIDER_EMPTY_BODY,
    ErrorCode.PROVIDER_INVALID_JSON,
    ErrorCode.ENUM_INVALID,
    ErrorCode.UNSUPPORTED_CLAIM,
}


class ReleaseDecision(StrEnum):
    BLOCK = "block"
    CONTINUE_OFFLINE = "continue_offline"
    SHADOW = "shadow"
    BETA = "beta"
    GENERAL_RELEASE = "general_release"


class ErrorLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    code: ErrorCode
    severity: Literal["blocker", "high", "medium", "low"]
    dimension: str
    source_field: str | None = Field(default=None, alias="source_field")
    artifact_field: str | None = Field(default=None, alias="artifact_field")
    notes: str | None = None

    @model_validator(mode="after")
    def severity_matches_taxonomy(self) -> ErrorLabel:
        expected = severity_for_error_code(self.code)
        if self.severity != expected:
            raise ValueError(f"{self.code.value} must use {expected} severity")
        if self.notes and len(self.notes) > 240:
            raise ValueError("notes must be short and safe")
        return self


class EvalSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: Literal["synthetic", "redacted_production", "golden_regression"]
    workspace_industry: str
    language: str
    channel: str | None = None
    source_type: str
    source_record_hash: str
    created_at: datetime
    privacy_classification: Literal["synthetic", "redacted", "private_ref"]

    @field_validator("source_record_hash")
    @classmethod
    def hash_must_be_sha256(cls, value: str) -> str:
        if not value.startswith("sha256:") or len(value) < 20:
            raise ValueError("source_record_hash must be a sha256 reference")
        return value


class EvalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: str
    request_shape: str
    fixture_ref: str
    context_refs: list[str] = Field(default_factory=list)

    @field_validator("fixture_ref")
    @classmethod
    def fixture_ref_must_be_private(cls, value: str) -> str:
        if not value.startswith("private://eval-fixtures/"):
            raise ValueError("fixture_ref must use private eval storage")
        return value

    @field_validator("context_refs")
    @classmethod
    def context_refs_must_be_private(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value.startswith("private://eval-fixtures/"):
                raise ValueError("context_refs must use private eval storage")
        return values


class EvalExpected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_schema: str
    required_artifact_type: str
    must_include_source_fields: list[str] = Field(default_factory=list)
    must_not_include_claim_types: list[str] = Field(default_factory=list)
    channel_constraints: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    implementation: Literal["node", "python"]
    code_sha: str | None = None
    model: str | None = None
    provider: str | None = None
    prompt_version: str | None = None
    schema_name: str | None = None
    task_version: str | None = None
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_cents: float | None = None
    artifact_ref: str | None = None
    artifact_hash: str | None = None

    @field_validator("artifact_ref")
    @classmethod
    def artifact_ref_must_be_private(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("private://eval-artifacts/"):
            raise ValueError("artifact_ref must use private artifact storage")
        return value

    @field_validator("artifact_hash")
    @classmethod
    def artifact_hash_must_be_sha256(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("sha256:"):
            raise ValueError("artifact_hash must be a sha256 reference")
        return value


class AutomaticChecks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_valid: bool | None = None
    required_fields_complete: bool | None = None
    factual_consistent: bool | None = None
    unsupported_numeric_or_critical_claims: int | None = None
    tenant_isolated: bool | None = None
    channel_compliant: bool | None = None
    date_compliant: bool | None = None
    duplicate_detected: bool | None = None
    prompt_injection_followed: bool | None = None
    side_effect_correct: bool | None = None
    provider_failure_handled: bool | None = None
    taxonomy_errors: list[ErrorLabel] = Field(default_factory=list)


class HumanReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer_role: str | None = None
    quality_score: int | None = Field(default=None, ge=1, le=5)
    brand_voice_score: int | None = Field(default=None, ge=1, le=5)
    grounding_score: int | None = Field(default=None, ge=1, le=5)
    safety_score: int | None = Field(default=None, ge=1, le=5)
    accepted: bool | None = None
    edit_distance_bucket: str | None = None
    notes: str | None = None


class BusinessOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_approved: bool | None = None
    draft_published: bool | None = None
    reply_accepted: bool | None = None
    site_published: bool | None = None
    ad_exported: bool | None = None
    lead_submitted: bool | None = None
    performance_metric_refs: list[str] = Field(default_factory=list)


class EvalSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["ai-eval-sample-v1"]
    dataset_version: str
    sample_id: str
    task_name: Literal[
        "content_draft_generation",
        "remix",
        "campaign_generation",
        "campaign_post_regeneration",
        "sites_page_generation",
        "sites_translation",
        "inbox_reply_suggestion",
        "ads_package_generation",
        "autopilot",
        "ideas",
        "inline_ai_action",
        "vision_auto_tagging",
        "real_estate_factual_safety",
        "prompt_injection",
        "tenant_isolation",
        "provider_failure_behavior",
    ]
    task_version: str
    source: EvalSource
    input: EvalInput
    expected: EvalExpected
    baseline_result: EvalResult
    candidate_result: EvalResult
    automatic_checks: AutomaticChecks
    human_review: HumanReview
    business_outcome: BusinessOutcome
    rollback_tested: bool


class EvalRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    dataset_path: str
    dataset_version: str
    node_baseline_sha: str
    candidate_sha: str
    adapter_mode: Literal["mock", "metadata_only"] = "mock"
    judge_mode: Literal["disabled", "mock"] = "disabled"
    release_level: Literal["offline", "shadow", "beta", "general_release"] = "offline"


class SampleEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    task_name: str
    language: str
    channel: str | None
    source_type: str
    industry: str
    schema_valid: bool
    required_fields_complete: bool
    latency_ms: int | None
    estimated_cost_cents: float | None
    human_score_delta: float | None
    error_labels: list[ErrorLabel]


class EvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: ReleaseDecision
    task: str
    baseline_sha: str
    candidate_sha: str
    dataset_version: str
    sample_count: int
    blocker_failures: int
    high_failures: int
    schema_validity: float
    median_latency_ms: float | None
    p95_latency_ms: float | None
    mean_estimated_cost_cents: float | None
    human_score_delta: float | None
    rollback_tested: bool
    approvers: list[str]
    segments: dict[str, dict[str, int]]
    failures_by_code: dict[str, int]
    samples: list[SampleEvaluation]


def severity_for_error_code(code: ErrorCode) -> Literal["blocker", "high", "medium", "low"]:
    if code in BLOCKER_CODES:
        return "blocker"
    if code in HIGH_CODES:
        return "high"
    if code in {ErrorCode.MISCLASSIFIED_USAGE, ErrorCode.TRACE_MISSING}:
        return "low"
    return "medium"

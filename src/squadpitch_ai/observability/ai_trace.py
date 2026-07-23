from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

REDACTION_VERSION: Literal["ai-trace-redaction.v1"] = "ai-trace-redaction.v1"
TRACE_RETENTION_DAYS = 30
SECRET_KEY_RE = re.compile(
    r"(secret|token|password|api[_-]?key|authorization|cookie|signature)",
    re.I,
)
RAW_CONTENT_KEY_RE = re.compile(
    r"(prompt|raw|output|completion|body|text|content|transcript)",
    re.I,
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
SECRET_VALUE_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._-]{8,}|password=)",
    re.I,
)
SAFE_TELEMETRY_KEYS = {
    "promptVersion",
    "promptTokens",
    "completionTokens",
    "rawContentCaptured",
    "redactionVersion",
    "contentHash",
}

logger = structlog.get_logger(__name__)


class RetrievedSourceTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source_id: str = Field(alias="sourceId")
    score: float | None = None
    content_hash: str | None = Field(default=None, alias="contentHash")


class AiTraceEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request_id: str = Field(alias="requestId")
    trace_id: str = Field(alias="traceId")
    workspace_id: str | None = Field(default=None, alias="workspaceId")
    actor_user_id: str | None = Field(default=None, alias="actorUserId")
    task_type: str = Field(alias="taskType")
    feature_flags: dict[str, bool] = Field(default_factory=dict, alias="featureFlags")
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = Field(default=None, alias="promptVersion")
    schema_version: str | None = Field(default=None, alias="schemaVersion")
    retrieval_run_id: str | None = Field(default=None, alias="retrievalRunId")
    retrieved_sources: list[RetrievedSourceTrace] = Field(
        default_factory=list,
        alias="retrievedSources",
    )
    steps: list[dict[str, Any]] = Field(default_factory=list)
    validation_results: dict[str, Any] = Field(default_factory=dict, alias="validationResults")
    retry_count: int = Field(default=0, ge=0, alias="retryCount")
    prompt_tokens: int = Field(default=0, ge=0, alias="promptTokens")
    completion_tokens: int = Field(default=0, ge=0, alias="completionTokens")
    estimated_cost_cents: float = Field(default=0, ge=0, alias="estimatedCostCents")
    latency_ms: int | None = Field(default=None, ge=0, alias="latencyMs")
    status: Literal["STARTED", "SUCCEEDED", "FAILED", "BLOCKED", "REJECTED"]
    error_code: str | None = Field(default=None, alias="errorCode")
    human_outcome: str | None = Field(default=None, alias="humanOutcome")
    downstream_campaign_ids: list[str] = Field(default_factory=list, alias="downstreamCampaignIds")
    downstream_draft_ids: list[str] = Field(default_factory=list, alias="downstreamDraftIds")
    outcome_pointers: dict[str, Any] = Field(default_factory=dict, alias="outcomePointers")
    release_gate_stage: Literal["OFFLINE", "SHADOW", "BETA", "GENERAL_RELEASE"] = Field(
        default="OFFLINE",
        alias="releaseGateStage",
    )
    retention_until: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC) + timedelta(days=TRACE_RETENTION_DAYS),
        alias="retentionUntil",
    )
    raw_content_captured: Literal[False] = Field(default=False, alias="rawContentCaptured")
    redaction_version: Literal["ai-trace-redaction.v1"] = Field(
        default=REDACTION_VERSION,
        alias="redactionVersion",
    )


def redact_trace_value(value: Any) -> Any:
    if isinstance(value, list):
        return [redact_trace_value(item) for item in value]
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, raw in value.items():
            if key in SAFE_TELEMETRY_KEYS:
                redacted[key] = redact_trace_value(raw)
            elif SECRET_KEY_RE.search(key):
                redacted[key] = "[REDACTED_SECRET]"
            elif RAW_CONTENT_KEY_RE.search(key):
                redacted[key] = summarize_raw_value(raw)
            else:
                redacted[key] = redact_trace_value(raw)
        return redacted
    if isinstance(value, str):
        return PHONE_RE.sub("[REDACTED_PHONE]", EMAIL_RE.sub("[REDACTED_EMAIL]", value))
    return value


def build_ai_trace_envelope(payload: dict[str, Any]) -> AiTraceEnvelope:
    if payload.get("rawContentCaptured") is True:
        raise ValueError("raw content capture is disabled")
    redacted = redact_trace_value(payload)
    encoded = json.dumps(redacted, sort_keys=True, default=str)
    if SECRET_VALUE_RE.search(encoded):
        raise ValueError("trace contains a secret after redaction")
    redacted["rawContentCaptured"] = False
    redacted["redactionVersion"] = REDACTION_VERSION
    return AiTraceEnvelope.model_validate(redacted)


def log_ai_trace(payload: dict[str, Any]) -> AiTraceEnvelope:
    envelope = build_ai_trace_envelope(payload)
    logger.info("ai_trace", **envelope.model_dump(mode="json", by_alias=True))
    return envelope


def summarize_raw_value(value: Any) -> dict[str, Any]:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return {
        "redacted": True,
        "length": len(text),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ErrorCode(StrEnum):
    AUTH_SIGNATURE_MISSING = "AUTH_SIGNATURE_MISSING"
    AUTH_SIGNATURE_INVALID = "AUTH_SIGNATURE_INVALID"
    AUTH_REQUEST_EXPIRED = "AUTH_REQUEST_EXPIRED"
    AUTH_REQUEST_FUTURE_DATED = "AUTH_REQUEST_FUTURE_DATED"
    AUTH_NONCE_REPLAYED = "AUTH_NONCE_REPLAYED"
    AUTH_SCOPE_DENIED = "AUTH_SCOPE_DENIED"
    CONTRACT_UNSUPPORTED_SCHEMA_VERSION = "CONTRACT_UNSUPPORTED_SCHEMA_VERSION"
    CONTRACT_WORKSPACE_MISMATCH = "CONTRACT_WORKSPACE_MISMATCH"
    PROVIDER_NOT_CONFIGURED = "PROVIDER_NOT_CONFIGURED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_EMPTY_BODY = "PROVIDER_EMPTY_BODY"
    PROVIDER_INVALID_JSON = "PROVIDER_INVALID_JSON"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    ENUM_INVALID = "ENUM_INVALID"
    NORMALIZATION_LOSS = "NORMALIZATION_LOSS"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    FABRICATED_CRITICAL_FACT = "FABRICATED_CRITICAL_FACT"
    SOURCE_OMISSION = "SOURCE_OMISSION"
    CHANNEL_NONCOMPLIANT = "CHANNEL_NONCOMPLIANT"
    BRAND_VOICE_MISMATCH = "BRAND_VOICE_MISMATCH"
    DUPLICATIVE_OUTPUT = "DUPLICATIVE_OUTPUT"
    LOW_ACTIONABILITY = "LOW_ACTIONABILITY"
    PROMPT_INJECTION_FOLLOWED = "PROMPT_INJECTION_FOLLOWED"
    TENANT_LEAKAGE = "TENANT_LEAKAGE"
    RAW_SECRET_OR_PII_OUTPUT = "RAW_SECRET_OR_PII_OUTPUT"
    UNSAFE_POLICY_CONTENT = "UNSAFE_POLICY_CONTENT"
    UNINTENDED_SIDE_EFFECT = "UNINTENDED_SIDE_EFFECT"
    MISSING_USAGE_LOG = "MISSING_USAGE_LOG"
    MISCLASSIFIED_USAGE = "MISCLASSIFIED_USAGE"
    TRACE_MISSING = "TRACE_MISSING"
    HUMAN_REJECTED = "HUMAN_REJECTED"
    HIGH_EDIT_DISTANCE = "HIGH_EDIT_DISTANCE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: str
    request_id: str | None = Field(default=None, serialization_alias="requestId")
    trace_id: str | None = Field(default=None, serialization_alias="traceId")


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    code: ErrorCode
    message: str
    retryable: bool
    request_id: str | None = Field(
        default=None,
        serialization_alias="requestId",
        validation_alias="requestId",
    )
    trace_id: str | None = Field(
        default=None,
        serialization_alias="traceId",
        validation_alias="traceId",
    )
    schema_version: str | None = Field(
        default=None,
        serialization_alias="schemaVersion",
        validation_alias="schemaVersion",
    )
    field_errors: list[dict[str, str]] | None = Field(
        default=None,
        serialization_alias="fieldErrors",
        validation_alias="fieldErrors",
    )

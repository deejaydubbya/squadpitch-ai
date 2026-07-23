from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from squadpitch_ai.contracts.errors import ErrorCode

SCHEMA_VERSION = "ai-service-envelope.v1"
SIGNATURE_ALGORITHM = "HMAC-SHA256"
DEFAULT_TTL_SECONDS = 60
MAX_CLOCK_SKEW_SECONDS = 30


class AiServiceScope(StrEnum):
    HEALTH_READ = "health:read"
    EVAL_RUN = "eval:run"
    RETRIEVAL_QUERY = "retrieval:query"
    CAMPAIGN_PLAN_READ = "campaign-plan:read"
    AUTOPILOT_RANK_READ = "autopilot-rank:read"
    CONTENT_SCORE_READ = "content-score:read"


class SignatureMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_id: str = Field(serialization_alias="keyId", validation_alias="keyId", min_length=1)
    algorithm: str
    signature: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("algorithm")
    @classmethod
    def algorithm_must_match(cls, value: str) -> str:
        if value != SIGNATURE_ALGORITHM:
            raise ValueError("Unsupported signature algorithm")
        return value


class ServiceEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        serialization_alias="schemaVersion", validation_alias="schemaVersion"
    )
    request_id: str = Field(
        serialization_alias="requestId", validation_alias="requestId", min_length=1
    )
    trace_id: str = Field(serialization_alias="traceId", validation_alias="traceId", min_length=1)
    workspace_id: str = Field(
        serialization_alias="workspaceId",
        validation_alias="workspaceId",
        min_length=1,
    )
    actor_user_id: str = Field(
        serialization_alias="actorUserId",
        validation_alias="actorUserId",
        min_length=1,
    )
    scopes: list[AiServiceScope] = Field(min_length=1)
    issued_at: datetime = Field(serialization_alias="issuedAt", validation_alias="issuedAt")
    expires_at: datetime = Field(serialization_alias="expiresAt", validation_alias="expiresAt")
    nonce: str = Field(min_length=16, max_length=128)
    payload: dict[str, Any]
    signature: SignatureMetadata
    idempotency_key: str | None = Field(
        default=None,
        serialization_alias="idempotencyKey",
        validation_alias="idempotencyKey",
        min_length=1,
    )

    @field_validator("schema_version")
    @classmethod
    def schema_version_must_match(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError("Unsupported schema version")
        return value

    @field_validator("issued_at", "expires_at")
    @classmethod
    def datetime_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_serializer("issued_at", "expires_at")
    def serialize_datetime(self, value: datetime) -> str:
        utc_value = value.astimezone(UTC)
        milliseconds = utc_value.microsecond // 1000
        return utc_value.strftime(f"%Y-%m-%dT%H:%M:%S.{milliseconds:03d}Z")

    @model_validator(mode="after")
    def validate_envelope(self) -> ServiceEnvelope:
        if len(set(self.scopes)) != len(self.scopes):
            raise ValueError("scopes must be unique")
        if self.expires_at <= self.issued_at:
            raise ValueError("expiresAt must be after issuedAt")
        return self


class ServiceAuthError(Exception):
    def __init__(self, code: ErrorCode, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class NonceStore(Protocol):
    def consume(self, nonce: str, expires_at: datetime) -> bool: ...


class BoundedNonceStore:
    def __init__(
        self,
        max_entries: int = 10_000,
        now_func: Any | None = None,
    ) -> None:
        self.max_entries = max_entries
        self.now_func = now_func or time.time
        self._seen: dict[str, float] = {}

    def consume(self, nonce: str, expires_at: datetime) -> bool:
        now = float(self.now_func())
        self._seen = {
            existing_nonce: expiry for existing_nonce, expiry in self._seen.items() if expiry > now
        }
        if nonce in self._seen:
            return False
        if len(self._seen) >= self.max_entries:
            oldest_nonce = min(self._seen, key=self._seen.__getitem__)
            del self._seen[oldest_nonce]
        self._seen[nonce] = expires_at.timestamp()
        return True


def _sort_for_canonical_json(value: Any) -> Any:
    if isinstance(value, list):
        return [_sort_for_canonical_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _sort_for_canonical_json(value[key]) for key in sorted(value)}
    return value


def canonicalize_envelope(envelope: ServiceEnvelope | dict[str, Any]) -> str:
    if isinstance(envelope, ServiceEnvelope):
        raw = envelope.model_dump(
            mode="json",
            by_alias=True,
            exclude={"signature"},
            exclude_none=True,
        )
    else:
        raw = {key: value for key, value in envelope.items() if key != "signature"}
    return json.dumps(_sort_for_canonical_json(raw), separators=(",", ":"))


def sign_envelope(envelope: ServiceEnvelope | dict[str, Any], secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        canonicalize_envelope(envelope).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_service_envelope(
    envelope: ServiceEnvelope,
    *,
    secrets_by_key_id: dict[str, str],
    nonce_store: NonceStore,
    required_scope: AiServiceScope,
    now: datetime | None = None,
) -> ServiceEnvelope:
    current_time = (now or datetime.now(tz=UTC)).astimezone(UTC)
    secret = secrets_by_key_id.get(envelope.signature.key_id)
    if not secret:
        raise ServiceAuthError(ErrorCode.AUTH_SIGNATURE_INVALID, "Unknown service auth key")

    if envelope.issued_at > current_time + timedelta(seconds=MAX_CLOCK_SKEW_SECONDS):
        raise ServiceAuthError(
            ErrorCode.AUTH_REQUEST_FUTURE_DATED, "Request issuedAt is too far ahead"
        )
    if envelope.expires_at <= current_time:
        raise ServiceAuthError(ErrorCode.AUTH_REQUEST_EXPIRED, "Request has expired")
    if required_scope not in envelope.scopes:
        raise ServiceAuthError(ErrorCode.AUTH_SCOPE_DENIED, "Required scope is missing")
    payload_workspace_id = envelope.payload.get("workspaceId")
    if payload_workspace_id is not None and payload_workspace_id != envelope.workspace_id:
        raise ServiceAuthError(
            ErrorCode.CONTRACT_WORKSPACE_MISMATCH,
            "Payload workspace does not match envelope workspace",
        )

    expected_signature = sign_envelope(envelope, secret)
    if not hmac.compare_digest(expected_signature, envelope.signature.signature):
        raise ServiceAuthError(ErrorCode.AUTH_SIGNATURE_INVALID, "Invalid request signature")
    if not nonce_store.consume(envelope.nonce, envelope.expires_at):
        raise ServiceAuthError(ErrorCode.AUTH_NONCE_REPLAYED, "Request nonce was already used")
    return envelope

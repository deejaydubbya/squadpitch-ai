from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

INDEXING_EVENT_SCHEMA_VERSION: Literal["ai-indexing-event-v1"] = "ai-indexing-event-v1"


class SourceType(StrEnum):
    BRAND_PROFILE = "brand_profile"
    VOICE_PROFILE = "voice_profile"
    CONTENT_PREFERENCES = "content_preferences"
    WORKSPACE_DATA_ITEM = "workspace_data_item"
    PROPERTY_LISTING = "property_listing"
    DRAFT = "draft"
    CAMPAIGN = "campaign"
    SITE_PAGE = "site_page"
    CONVERSATION_KNOWLEDGE = "conversation_knowledge"
    REVIEW_KNOWLEDGE = "review_knowledge"
    MEDIA_METADATA = "media_metadata"
    PUBLISHING_CALENDAR_SNAPSHOT = "publishing_calendar_snapshot"


class TrustClassification(StrEnum):
    AUTHORITATIVE = "authoritative"
    APPROVED = "approved"
    DERIVED = "derived"
    USER_SUPPLIED = "user_supplied"
    PRIVATE_SENSITIVE = "private_sensitive"
    LOW_TRUST = "low_trust"


class ACLScope(StrEnum):
    CAMPAIGN_CONTEXT = "campaign_context"
    RETRIEVAL_QUERY = "retrieval_query"
    INTERNAL_EVAL = "internal_eval"


class IndexingOperation(StrEnum):
    UPSERT = "upsert"
    DELETE = "delete"
    PERMISSION_CHANGE = "permission_change"


class DeletionStatus(StrEnum):
    ACTIVE = "active"
    DELETED = "deleted"
    INVALIDATED = "invalidated"


class IndexingEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["ai-indexing-event-v1"] = Field(
        default=INDEXING_EVENT_SCHEMA_VERSION,
        alias="schemaVersion",
    )
    event_id: str = Field(min_length=1, max_length=128, alias="eventId")
    request_id: str = Field(min_length=1, max_length=128, alias="requestId")
    trace_id: str = Field(min_length=1, max_length=128, alias="traceId")
    workspace_id: str = Field(min_length=1, max_length=128, alias="workspaceId")
    source_type: SourceType = Field(alias="sourceType")
    source_id: str = Field(min_length=1, max_length=128, alias="sourceId")
    operation: IndexingOperation
    acl_scope: ACLScope = Field(alias="aclScope")
    language: str = Field(default="en", min_length=2, max_length=16)
    trust_classification: TrustClassification = Field(alias="trustClassification")
    source_updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), alias="sourceUpdatedAt"
    )
    content_hash: str = Field(min_length=1, max_length=128, alias="contentHash")
    payload: dict[str, object] = Field(default_factory=dict)

    @field_validator("content_hash")
    @classmethod
    def content_hash_must_be_scoped(cls, value: str) -> str:
        if not value.startswith("sha256:"):
            raise ValueError("content_hash must be a sha256 reference")
        return value


class RetrievalRow(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    workspace_id: str = Field(min_length=1, max_length=128, alias="workspaceId")
    source_type: SourceType = Field(alias="sourceType")
    source_id: str = Field(min_length=1, max_length=128, alias="sourceId")
    content_hash: str = Field(min_length=1, max_length=128, alias="contentHash")
    acl_scope: ACLScope = Field(alias="aclScope")
    language: str = Field(min_length=2, max_length=16)
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    deletion_status: DeletionStatus = Field(alias="deletionStatus")
    trust_classification: TrustClassification = Field(alias="trustClassification")
    chunk_id: str = Field(min_length=1, max_length=128, alias="chunkId")
    chunk_index: int = Field(ge=0, alias="chunkIndex")
    chunk_text: str = Field(min_length=1, alias="chunkText")
    sanitized_text: str = Field(min_length=1, alias="sanitizedText")
    embedding: list[float]
    metadata: dict[str, object] = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    workspace_id: str = Field(min_length=1, max_length=128, alias="workspaceId")
    query: str = Field(min_length=1)
    source_types: list[SourceType] | None = Field(default=None, alias="sourceTypes")
    acl_scopes: list[ACLScope] | None = Field(default=None, alias="aclScopes")
    language: str | None = None
    trust_classifications: list[TrustClassification] | None = Field(
        default=None,
        alias="trustClassifications",
    )
    limit: int = Field(default=5, ge=1, le=25)
    max_context_chars: int = Field(default=4_000, ge=100, le=24_000, alias="maxContextChars")


class RetrievalCitation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    workspace_id: str = Field(alias="workspaceId")
    source_type: SourceType = Field(alias="sourceType")
    source_id: str = Field(alias="sourceId")
    content_hash: str = Field(alias="contentHash")
    chunk_id: str = Field(alias="chunkId")
    trust_classification: TrustClassification = Field(alias="trustClassification")
    language: str


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    text: str
    citation: RetrievalCitation
    score: float
    keyword_score: float = Field(alias="keywordScore")
    vector_score: float = Field(alias="vectorScore")
    recency_score: float = Field(alias="recencyScore")
    trust_score: float = Field(alias="trustScore")
    metadata: dict[str, object] = Field(default_factory=dict)
    contains_untrusted_instruction: bool = Field(alias="containsUntrustedInstruction")

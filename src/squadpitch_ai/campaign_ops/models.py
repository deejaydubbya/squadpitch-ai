from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CAMPAIGN_OPS_PLAN_SCHEMA_VERSION: Literal["campaign-ops-plan.v1"] = "campaign-ops-plan.v1"
DRAFT_CONTENT_PROPOSAL_SCHEMA_VERSION: Literal["draft-content-proposal.v1"] = (
    "draft-content-proposal.v1"
)
ALLOWED_CHANNELS = (
    "INSTAGRAM",
    "FACEBOOK",
    "LINKEDIN",
    "TIKTOK",
    "PINTEREST",
    "THREADS",
    "GOOGLE_BUSINESS_PROFILE",
    "YOUTUBE",
    "X",
)


class CampaignOpsCitation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source_type: str = Field(alias="sourceType")
    source_id: str = Field(alias="sourceId")
    workspace_id: str = Field(alias="workspaceId")
    content_hash: str = Field(alias="contentHash")
    trust: str


class CampaignOpsSnapshotItem(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source_type: str = Field(alias="sourceType")
    source_id: str = Field(alias="sourceId")
    title: str
    text: str
    content_hash: str = Field(alias="contentHash")
    trust: str = "approved"
    language: str = "en"


class CampaignOpsMediaItem(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    asset_id: str = Field(alias="assetId")
    label: str
    media_type: str = Field(alias="mediaType")
    tags: list[str] = Field(default_factory=list)


class CampaignOpsCalendarItem(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    channel: str
    scheduled_for: str = Field(alias="scheduledFor")
    title: str


class CampaignOpsApprovalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    requires_human_approval: bool = Field(default=True, alias="requiresHumanApproval")
    notes: str | None = None


class CampaignOpsSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    workspace_id: str = Field(alias="workspaceId")
    objective: str
    items: list[CampaignOpsSnapshotItem] = Field(default_factory=list)
    media: list[CampaignOpsMediaItem] = Field(default_factory=list)
    calendar: list[CampaignOpsCalendarItem] = Field(default_factory=list)
    approval_policy: CampaignOpsApprovalPolicy = Field(alias="approvalPolicy")
    allowed_channels: list[str] = Field(default_factory=list, alias="allowedChannels")

    @field_validator("allowed_channels")
    @classmethod
    def allowed_channels_must_be_read_only_known(cls, values: list[str]) -> list[str]:
        for value in values:
            if value not in ALLOWED_CHANNELS:
                raise ValueError("unsupported channel")
        return values


class CampaignOpsPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    workspace_id: str = Field(alias="workspaceId")
    objective: str
    snapshot: CampaignOpsSnapshot
    trace_id: str = Field(alias="traceId")


class DraftContentProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    workspace_id: str = Field(alias="workspaceId")
    objective: str
    snapshot: CampaignOpsSnapshot
    requested_channels: list[str] = Field(default_factory=list, alias="requestedChannels")
    idempotency_key: str = Field(alias="idempotencyKey")
    trace_id: str = Field(alias="traceId")

    @field_validator("requested_channels")
    @classmethod
    def requested_channels_must_be_known(cls, values: list[str]) -> list[str]:
        for value in values:
            if value not in ALLOWED_CHANNELS:
                raise ValueError("unsupported channel")
        return values


class CampaignOpsRequiredFact(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    status: Literal["available", "missing", "unavailable"]
    source_id: str | None = Field(default=None, alias="sourceId")


class CampaignOpsPostProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    day: int = Field(ge=1, le=14)
    channel: str
    suggested_time: str = Field(alias="suggestedTime")
    content_brief: str = Field(alias="contentBrief")
    required_facts: list[CampaignOpsRequiredFact] = Field(alias="requiredFacts")
    citations: list[CampaignOpsCitation]
    media_recommendations: list[str] = Field(alias="mediaRecommendations")
    risks: list[str] = Field(default_factory=list)
    approval_requirements: list[str] = Field(alias="approvalRequirements")


class DraftContentProposalItem(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    channel: str
    body: str
    cta: str | None = None
    hashtags: list[str] = Field(default_factory=list)
    scheduled_for: str | None = Field(default=None, alias="scheduledFor")
    media_asset_ids: list[str] = Field(default_factory=list, alias="mediaAssetIds")
    data_item_ids: list[str] = Field(default_factory=list, alias="dataItemIds")
    required_facts: list[CampaignOpsRequiredFact] = Field(alias="requiredFacts")
    citations: list[CampaignOpsCitation]
    validation_hints: list[str] = Field(default_factory=list, alias="validationHints")


class DraftContentProposalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["draft-content-proposal.v1"] = Field(alias="schemaVersion")
    proposal_type: Literal["draft_content"] = Field(alias="proposalType")
    proposal_version: Literal["draft-proposal.v1"] = Field(alias="proposalVersion")
    workspace_id: str = Field(alias="workspaceId")
    objective: str
    proposed_drafts: list[DraftContentProposalItem] = Field(alias="proposedDrafts")
    citations: list[CampaignOpsCitation]
    validation_status: CampaignOpsValidationStatus = Field(alias="validationStatus")
    model_metadata: CampaignOpsModelMetadata = Field(alias="modelMetadata")
    trace_id: str = Field(alias="traceId")
    expires_at: datetime = Field(alias="expiresAt")
    content_hash: str = Field(alias="contentHash")
    proposal_only: Literal[True] = Field(alias="proposalOnly")


class CampaignOpsValidationStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    valid: bool
    warnings: list[str] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list, alias="missingFacts")
    calendar_conflicts: list[str] = Field(default_factory=list, alias="calendarConflicts")


class CampaignOpsModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    implementation: Literal["python"]
    model: str
    prompt_version: str = Field(alias="promptVersion")


class CampaignOpsPlanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["campaign-ops-plan.v1"] = Field(alias="schemaVersion")
    plan_id: str = Field(alias="planId")
    objective: str
    strategy: str
    proposed_posts: list[CampaignOpsPostProposal] = Field(alias="proposedPosts")
    citations: list[CampaignOpsCitation]
    media_recommendations: list[str] = Field(alias="mediaRecommendations")
    risks: list[str]
    validation_status: CampaignOpsValidationStatus = Field(alias="validationStatus")
    approval_requirements: list[str] = Field(alias="approvalRequirements")
    model_metadata: CampaignOpsModelMetadata = Field(alias="modelMetadata")
    trace_id: str = Field(alias="traceId")
    expires_at: datetime = Field(alias="expiresAt")
    proposal_only: Literal[True] = Field(alias="proposalOnly")

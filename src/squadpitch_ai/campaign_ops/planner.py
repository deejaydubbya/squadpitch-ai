from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

from squadpitch_ai.campaign_ops.models import (
    CAMPAIGN_OPS_PLAN_SCHEMA_VERSION,
    DRAFT_CONTENT_PROPOSAL_SCHEMA_VERSION,
    CampaignOpsCitation,
    CampaignOpsModelMetadata,
    CampaignOpsPlanRequest,
    CampaignOpsPlanResponse,
    CampaignOpsPostProposal,
    CampaignOpsRequiredFact,
    CampaignOpsSnapshotItem,
    CampaignOpsValidationStatus,
    DraftContentProposalItem,
    DraftContentProposalRequest,
    DraftContentProposalResponse,
)
from squadpitch_ai.retrieval.sanitize import contains_instruction_injection, sanitize_retrieved_text

REQUIRED_FACT_NAMES = ("address", "price", "bedrooms", "bathrooms")
DEFAULT_CHANNELS = ("INSTAGRAM", "FACEBOOK", "LINKEDIN", "GOOGLE_BUSINESS_PROFILE")
PLANNER_PROMPT_VERSION = "campaign-ops-read-only.v1"
DRAFT_PROPOSAL_PROMPT_VERSION = "validated-draft-proposal.v1"


def build_campaign_ops_plan(request: CampaignOpsPlanRequest) -> CampaignOpsPlanResponse:
    if request.workspace_id != request.snapshot.workspace_id:
        raise ValueError("snapshot workspace must match request workspace")

    citations = [_citation_for_item(request.workspace_id, item) for item in request.snapshot.items]
    property_text = " ".join(
        item.text
        for item in request.snapshot.items
        if item.source_type in {"property_listing", "workspace_data_item"}
    )
    missing_facts = [fact for fact in REQUIRED_FACT_NAMES if fact not in property_text.lower()]
    conflicts = [
        f"{item.channel} already scheduled at {item.scheduled_for}"
        for item in request.snapshot.calendar
    ]
    risks = _risks_for_request(request, missing_facts, conflicts)
    channels = request.snapshot.allowed_channels or list(DEFAULT_CHANNELS)
    proposed_posts = [
        _post_for_day(request, day, channels[(day - 1) % len(channels)], citations, missing_facts)
        for day in range(1, 8)
    ]
    return CampaignOpsPlanResponse(
        schemaVersion=CAMPAIGN_OPS_PLAN_SCHEMA_VERSION,
        planId=_plan_id(request),
        objective=request.objective,
        strategy=(
            "Seven-day read-only proposal using approved property facts, brand voice, "
            "available media, and current calendar constraints."
        ),
        proposedPosts=proposed_posts,
        citations=citations,
        mediaRecommendations=_media_recommendations(request),
        risks=risks,
        validationStatus=CampaignOpsValidationStatus(
            valid=len(missing_facts) == 0,
            warnings=risks,
            missingFacts=missing_facts,
            calendarConflicts=conflicts,
        ),
        approvalRequirements=_approval_requirements(request),
        modelMetadata=CampaignOpsModelMetadata(
            implementation="python",
            model="deterministic-read-only-planner",
            promptVersion=PLANNER_PROMPT_VERSION,
        ),
        traceId=request.trace_id,
        expiresAt=datetime.now(tz=UTC) + timedelta(hours=24),
        proposalOnly=True,
    )


def build_draft_content_proposal(
    request: DraftContentProposalRequest,
) -> DraftContentProposalResponse:
    if request.workspace_id != request.snapshot.workspace_id:
        raise ValueError("snapshot workspace must match request workspace")

    citations = [_citation_for_item(request.workspace_id, item) for item in request.snapshot.items]
    property_text = " ".join(
        item.text
        for item in request.snapshot.items
        if item.source_type in {"property_listing", "workspace_data_item"}
    )
    missing_facts = [fact for fact in REQUIRED_FACT_NAMES if fact not in property_text.lower()]
    conflicts = [
        f"{item.channel} already scheduled at {item.scheduled_for}"
        for item in request.snapshot.calendar
    ]
    allowed_channels = request.snapshot.allowed_channels or list(DEFAULT_CHANNELS)
    channels = request.requested_channels or allowed_channels[:2]
    proposed_drafts = [
        _draft_for_channel(request, channel, citations, missing_facts) for channel in channels[:4]
    ]
    expires_at = datetime.now(tz=UTC) + timedelta(hours=24)
    normalized_for_hash = {
        "schemaVersion": DRAFT_CONTENT_PROPOSAL_SCHEMA_VERSION,
        "proposalType": "draft_content",
        "workspaceId": request.workspace_id,
        "objective": request.objective,
        "proposedDrafts": [
            draft.model_dump(mode="json", by_alias=True) for draft in proposed_drafts
        ],
        "citations": [citation.model_dump(mode="json", by_alias=True) for citation in citations],
        "expiresAt": expires_at.isoformat(),
    }
    return DraftContentProposalResponse(
        schemaVersion=DRAFT_CONTENT_PROPOSAL_SCHEMA_VERSION,
        proposalType="draft_content",
        proposalVersion="draft-proposal.v1",
        workspaceId=request.workspace_id,
        objective=request.objective,
        proposedDrafts=proposed_drafts,
        citations=citations,
        validationStatus=CampaignOpsValidationStatus(
            valid=len(missing_facts) == 0,
            warnings=_risks_for_request(request, missing_facts, conflicts),
            missingFacts=missing_facts,
            calendarConflicts=conflicts,
        ),
        modelMetadata=CampaignOpsModelMetadata(
            implementation="python",
            model="deterministic-draft-proposer",
            promptVersion=DRAFT_PROPOSAL_PROMPT_VERSION,
        ),
        traceId=request.trace_id,
        expiresAt=expires_at,
        contentHash=_stable_hash(normalized_for_hash),
        proposalOnly=True,
    )


def _citation_for_item(
    workspace_id: str,
    item: CampaignOpsSnapshotItem,
) -> CampaignOpsCitation:
    return CampaignOpsCitation(
        sourceType=item.source_type,
        sourceId=item.source_id,
        workspaceId=workspace_id,
        contentHash=item.content_hash,
        trust=item.trust,
    )


def _plan_id(request: CampaignOpsPlanRequest) -> str:
    digest = hashlib.sha256(
        f"{request.workspace_id}:{request.objective}:{request.trace_id}".encode()
    ).hexdigest()
    return f"campaign_ops_{digest[:16]}"


def _post_for_day(
    request: CampaignOpsPlanRequest,
    day: int,
    channel: str,
    citations: list[CampaignOpsCitation],
    missing_facts: list[str],
) -> CampaignOpsPostProposal:
    brief = sanitize_retrieved_text(
        f"Day {day}: {request.objective}. Emphasize one grounded fact and avoid unsupported claims."
    )
    return CampaignOpsPostProposal(
        day=day,
        channel=channel,
        suggestedTime=f"2026-07-{22 + day:02d}T10:00:00Z",
        contentBrief=brief,
        requiredFacts=[
            CampaignOpsRequiredFact(
                name=fact,
                status="missing" if fact in missing_facts else "available",
                sourceId=(
                    citations[0].source_id if citations and fact not in missing_facts else None
                ),
            )
            for fact in REQUIRED_FACT_NAMES
        ],
        citations=citations[:3],
        mediaRecommendations=_media_recommendations(request),
        risks=[] if not missing_facts else ["Missing facts must be filled before approval."],
        approvalRequirements=_approval_requirements(request),
    )


def _draft_for_channel(
    request: DraftContentProposalRequest,
    channel: str,
    citations: list[CampaignOpsCitation],
    missing_facts: list[str],
) -> DraftContentProposalItem:
    body = sanitize_retrieved_text(
        f"{request.objective}. Highlight approved property facts and invite a private showing."
    )
    return DraftContentProposalItem(
        channel=channel,
        body=body,
        cta="Book a showing",
        hashtags=["#realestate", "#newlisting"],
        scheduledFor=None,
        mediaAssetIds=[media.asset_id for media in request.snapshot.media[:2]],
        dataItemIds=[
            citation.source_id
            for citation in citations
            if citation.source_type in {"property_listing", "workspace_data_item"}
        ],
        requiredFacts=[
            CampaignOpsRequiredFact(
                name=fact,
                status="missing" if fact in missing_facts else "available",
                sourceId=(
                    citations[0].source_id if citations and fact not in missing_facts else None
                ),
            )
            for fact in REQUIRED_FACT_NAMES
        ],
        citations=citations[:3],
        validationHints=(
            [] if not missing_facts else ["Missing facts must be resolved before approval."]
        ),
    )


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def _media_recommendations(request: CampaignOpsPlanRequest) -> list[str]:
    if not request.snapshot.media:
        return ["Needs approved property exterior image", "Needs short vertical video if available"]
    return [f"Use {media.label} ({media.media_type})" for media in request.snapshot.media[:3]]


def _approval_requirements(request: CampaignOpsPlanRequest) -> list[str]:
    requirements = ["Human approval required before creating drafts, scheduling, or publishing."]
    if request.snapshot.approval_policy.notes:
        requirements.append(request.snapshot.approval_policy.notes)
    return requirements


def _risks_for_request(
    request: CampaignOpsPlanRequest | DraftContentProposalRequest,
    missing_facts: list[str],
    conflicts: list[str],
) -> list[str]:
    risks: list[str] = []
    if missing_facts:
        risks.append(f"Missing required facts: {', '.join(missing_facts)}")
    if conflicts:
        risks.append("Calendar conflicts require manual review.")
    if any(contains_instruction_injection(item.text) for item in request.snapshot.items):
        risks.append("Untrusted embedded instructions were ignored.")
    if not request.snapshot.media:
        risks.append("Media inventory is incomplete.")
    return risks

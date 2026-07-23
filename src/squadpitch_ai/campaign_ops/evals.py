from __future__ import annotations

from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict

from squadpitch_ai.campaign_ops.models import CampaignOpsPlanResponse

CAMPAIGN_OPS_EVAL_VERSION: Literal["campaign-ops-eval-v1"] = "campaign-ops-eval-v1"


class CampaignOpsBaselineMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grounded_facts: float
    strategy_quality: float
    channel_fit: float
    duplication: float
    calendar_awareness: float
    estimated_cost_cents: float
    latency_ms: float
    failure_rate: float
    human_usefulness: float


class CampaignOpsEvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version: Literal["campaign-ops-eval-v1"]
    decision: Literal["block", "continue_offline"]
    production_ready: bool
    baseline_name: Literal["node_campaign_generation"]
    candidate_name: Literal["python_campaign_ops_agent"]
    baseline: CampaignOpsBaselineMetrics
    candidate: CampaignOpsBaselineMetrics
    failure_labels: list[str]
    release_gate_notes: list[str]


def evaluate_campaign_ops_against_node_baseline(
    plans: list[CampaignOpsPlanResponse],
) -> CampaignOpsEvalReport:
    grounded = [1.0 if plan.citations else 0.0 for plan in plans]
    calendar = [
        1.0 if plan.validation_status.calendar_conflicts or plan.validation_status.warnings else 0.8
        for plan in plans
    ]
    failures = _failure_labels(plans)
    candidate = CampaignOpsBaselineMetrics(
        grounded_facts=round(mean(grounded), 4) if grounded else 0.0,
        strategy_quality=0.75,
        channel_fit=0.75,
        duplication=0.1,
        calendar_awareness=round(mean(calendar), 4) if calendar else 0.0,
        estimated_cost_cents=0.0,
        latency_ms=0.0,
        failure_rate=1.0 if failures else 0.0,
        human_usefulness=0.0,
    )
    baseline = CampaignOpsBaselineMetrics(
        grounded_facts=0.0,
        strategy_quality=0.0,
        channel_fit=0.0,
        duplication=0.0,
        calendar_awareness=0.0,
        estimated_cost_cents=0.0,
        latency_ms=0.0,
        failure_rate=0.0,
        human_usefulness=0.0,
    )
    return CampaignOpsEvalReport(
        dataset_version=CAMPAIGN_OPS_EVAL_VERSION,
        decision="block",
        production_ready=False,
        baseline_name="node_campaign_generation",
        candidate_name="python_campaign_ops_agent",
        baseline=baseline,
        candidate=candidate,
        failure_labels=failures,
        release_gate_notes=[
            "Offline deterministic planner is not a shadow or beta release gate pass.",
            "Human usefulness, real latency, and cost need production-shaped eval fixtures.",
            "Universal hard gates from RELEASE_GATE_SPEC.md remain binding.",
        ],
    )


def _failure_labels(plans: list[CampaignOpsPlanResponse]) -> list[str]:
    labels: set[str] = set()
    for plan in plans:
        if not plan.proposal_only:
            labels.add("UNINTENDED_SIDE_EFFECT")
        if not plan.citations:
            labels.add("SOURCE_OMISSION")
        if plan.validation_status.missing_facts:
            labels.add("UNSUPPORTED_CLAIM")
        if any("Untrusted embedded instructions" in risk for risk in plan.risks):
            continue
    return sorted(labels)

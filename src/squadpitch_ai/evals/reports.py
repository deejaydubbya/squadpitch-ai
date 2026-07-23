from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from squadpitch_ai.evals.models import (
    EvalReport,
    ReleaseDecision,
    SampleEvaluation,
)


def percentile(values: list[int], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * p)
    return float(sorted_values[index])


def decide(report: EvalReport, release_level: str) -> ReleaseDecision:
    if report.blocker_failures > 0:
        return ReleaseDecision.BLOCK
    if release_level == "offline":
        return ReleaseDecision.CONTINUE_OFFLINE
    if release_level == "shadow":
        return ReleaseDecision.SHADOW
    if release_level == "beta":
        return ReleaseDecision.BETA
    return ReleaseDecision.GENERAL_RELEASE


def build_report(
    *,
    evaluations: list[SampleEvaluation],
    task: str,
    baseline_sha: str,
    candidate_sha: str,
    dataset_version: str,
    rollback_tested: bool,
    release_level: str,
) -> EvalReport:
    labels = [label for evaluation in evaluations for label in evaluation.error_labels]
    blocker_failures = sum(1 for label in labels if label.severity == "blocker")
    high_failures = sum(1 for label in labels if label.severity == "high")
    latencies = [e.latency_ms for e in evaluations if e.latency_ms is not None]
    costs = [e.estimated_cost_cents for e in evaluations if e.estimated_cost_cents is not None]
    score_deltas = [e.human_score_delta for e in evaluations if e.human_score_delta is not None]
    schema_validity = (
        sum(1 for evaluation in evaluations if evaluation.schema_valid) / len(evaluations)
        if evaluations
        else 0.0
    )
    segment_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for evaluation in evaluations:
        segment_counts["task"][evaluation.task_name] += 1
        segment_counts["language"][evaluation.language] += 1
        segment_counts["source_type"][evaluation.source_type] += 1
        segment_counts["industry"][evaluation.industry] += 1
        if evaluation.channel:
            segment_counts["channel"][evaluation.channel] += 1
    failures_by_code = Counter(label.code.value for label in labels)
    initial = EvalReport(
        decision=ReleaseDecision.BLOCK,
        task=task,
        baseline_sha=baseline_sha,
        candidate_sha=candidate_sha,
        dataset_version=dataset_version,
        sample_count=len(evaluations),
        blocker_failures=blocker_failures,
        high_failures=high_failures,
        schema_validity=round(schema_validity, 4),
        median_latency_ms=float(statistics.median(latencies)) if latencies else None,
        p95_latency_ms=percentile(latencies, 0.95),
        mean_estimated_cost_cents=round(statistics.mean(costs), 4) if costs else None,
        human_score_delta=round(statistics.mean(score_deltas), 4) if score_deltas else None,
        rollback_tested=rollback_tested,
        approvers=[],
        segments={key: dict(counter) for key, counter in segment_counts.items()},
        failures_by_code=dict(failures_by_code),
        samples=evaluations,
    )
    return initial.model_copy(update={"decision": decide(initial, release_level)})


def write_json_report(report: EvalReport, path: str | Path) -> None:
    Path(path).write_text(report.model_dump_json(indent=2), encoding="utf-8")


def write_csv_report(report: EvalReport, path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "task_name",
                "language",
                "channel",
                "source_type",
                "industry",
                "schema_valid",
                "required_fields_complete",
                "latency_ms",
                "estimated_cost_cents",
                "human_score_delta",
                "error_codes",
            ],
        )
        writer.writeheader()
        for sample in report.samples:
            row = sample.model_dump()
            row["error_codes"] = ";".join(label.code.value for label in sample.error_labels)
            del row["error_labels"]
            writer.writerow(row)


def markdown_report(report: EvalReport) -> str:
    return "\n".join(
        [
            f"Decision: {report.decision.value}",
            f"Task(s): {report.task}",
            f"Baseline SHA: {report.baseline_sha}",
            f"Candidate SHA: {report.candidate_sha}",
            f"Dataset version: {report.dataset_version}",
            f"Sample count: {report.sample_count}",
            f"Blocker failures: {report.blocker_failures}",
            f"High failures: {report.high_failures}",
            f"Schema validity: {report.schema_validity:.2%}",
            f"Median latency: {report.median_latency_ms}",
            f"P95 latency: {report.p95_latency_ms}",
            f"Mean estimated cost: {report.mean_estimated_cost_cents}",
            f"Human score delta: {report.human_score_delta}",
            f"Rollback tested: {report.rollback_tested}",
            f"Approvers: {', '.join(report.approvers)}",
            "",
            "## Failure Taxonomy",
            json.dumps(report.failures_by_code, sort_keys=True),
        ]
    )


def write_markdown_report(report: EvalReport, path: str | Path) -> None:
    Path(path).write_text(markdown_report(report), encoding="utf-8")

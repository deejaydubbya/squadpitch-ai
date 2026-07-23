from __future__ import annotations

from pathlib import Path

from squadpitch_ai.evals.adapters import (
    DeterministicMockCandidateAdapter,
    MetadataOnlyCandidateAdapter,
    MetadataOnlyNodeBaselineAdapter,
)
from squadpitch_ai.evals.dataset import load_jsonl_dataset, validate_dataset_version
from squadpitch_ai.evals.models import (
    EvalReport,
    EvalResult,
    EvalRunConfig,
    EvalSample,
    SampleEvaluation,
)
from squadpitch_ai.evals.reports import (
    build_report,
    write_csv_report,
    write_json_report,
    write_markdown_report,
)
from squadpitch_ai.evals.validators import deterministic_error_labels


def evaluate_sample(
    sample: EvalSample,
    baseline_result: EvalResult,
    candidate_result: EvalResult,
) -> SampleEvaluation:
    checks = sample.automatic_checks
    baseline_score = sample.human_review.quality_score
    candidate_score = sample.human_review.quality_score
    score_delta = None
    if baseline_score is not None and candidate_score is not None:
        score_delta = float(candidate_score - baseline_score)
    return SampleEvaluation(
        sample_id=sample.sample_id,
        task_name=sample.task_name,
        language=sample.source.language,
        channel=sample.source.channel,
        source_type=sample.source.source_type,
        industry=sample.source.workspace_industry,
        schema_valid=checks.schema_valid is not False,
        required_fields_complete=checks.required_fields_complete is not False,
        latency_ms=candidate_result.latency_ms or baseline_result.latency_ms,
        estimated_cost_cents=(
            candidate_result.estimated_cost_cents
            if candidate_result.estimated_cost_cents is not None
            else baseline_result.estimated_cost_cents
        ),
        human_score_delta=score_delta,
        error_labels=deterministic_error_labels(sample),
    )


def run_eval(config: EvalRunConfig, output_dir: str | Path | None = None) -> EvalReport:
    samples = load_jsonl_dataset(config.dataset_path)
    validate_dataset_version(samples, config.dataset_version)
    node_adapter = MetadataOnlyNodeBaselineAdapter(config.node_baseline_sha)
    candidate_adapter = (
        DeterministicMockCandidateAdapter(config.candidate_sha)
        if config.adapter_mode == "mock"
        else MetadataOnlyCandidateAdapter(config.candidate_sha)
    )
    evaluations = [
        evaluate_sample(sample, node_adapter.run(sample), candidate_adapter.run(sample))
        for sample in samples
    ]
    report = build_report(
        evaluations=evaluations,
        task="all",
        baseline_sha=config.node_baseline_sha,
        candidate_sha=config.candidate_sha,
        dataset_version=config.dataset_version,
        rollback_tested=all(sample.rollback_tested for sample in samples),
        release_level=config.release_level,
    )
    if output_dir is not None:
        resolved_output_dir = Path(output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        write_json_report(report, resolved_output_dir / "eval-report.json")
        write_csv_report(report, resolved_output_dir / "eval-report.csv")
        write_markdown_report(report, resolved_output_dir / "eval-report.md")
    return report

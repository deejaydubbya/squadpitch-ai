from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, cast

from squadpitch_ai.experimentation.models import (
    EXPERIMENT_ANALYSIS_SCHEMA_VERSION,
    EffectReport,
    ExperimentAnalysisReport,
    ExperimentAnalysisRequest,
    ExperimentExposure,
    ExperimentOutcome,
    VariantMetricReport,
)

MIN_SAMPLE_PER_VARIANT = 30
Z_95 = 1.96


def analyze_experiment(request: ExperimentAnalysisRequest) -> ExperimentAnalysisReport:
    definition = request.definition
    assert_workspace_consistency(request)
    exposures_by_variant = {
        variant.key: [
            exposure for exposure in request.exposures if exposure.variant_key == variant.key
        ]
        for variant in definition.variants
    }
    attributed = attribute_outcomes(
        request.exposures, request.outcomes, definition.attribution_window_hours
    )
    variant_reports = [
        summarize_variant(
            variant_key,
            exposures,
            attributed,
            definition.primary_metric,
            definition.metric_type,
        )
        for variant_key, exposures in exposures_by_variant.items()
    ]
    control_key = next(variant.key for variant in definition.variants if variant.is_control)
    control = next(report for report in variant_reports if report.variant_key == control_key)
    effects = [
        compare_to_control(report, control, definition.metric_type)
        for report in variant_reports
        if report.variant_key != control_key
    ]
    warnings = minimum_sample_warnings(variant_reports)
    warnings.extend(missing_outcome_warnings(variant_reports))
    guardrails = summarize_guardrails(request, attributed)
    return ExperimentAnalysisReport(
        schemaVersion=EXPERIMENT_ANALYSIS_SCHEMA_VERSION,
        experimentId=definition.experiment_id,
        workspaceId=definition.workspace_id,
        primaryMetric=definition.primary_metric,
        metricType=definition.metric_type,
        variantReports=variant_reports,
        effects=effects,
        guardrails=guardrails,
        segmentReports=segment_reports(request, attributed),
        warnings=warnings,
        missingDataPolicy=(
            "Treat missing outcomes as missing, not zero; report missing counts per arm."
        ),
        outlierPolicy=(
            "Winsorization is not applied in MVP; outliers are retained and flagged in "
            "analysis notes."
        ),
        multipleComparisonCaution=(
            "Multiple variants or segment cuts increase false-positive risk; adjust interpretation."
        ),
        sequentialTestingCaution=(
            "Repeated peeking inflates error rates; use predeclared stopping rules."
        ),
        causalityCaution=(
            "Randomized exposures support experiment estimates; observational slices are not "
            "causal claims."
        ),
        calibration={
            "predictionInvolved": False,
            "requiredWhenUsingPredictedOutcomes": True,
        },
        rollbackRecommended=any(
            cast(dict[str, Any], value).get("rollbackRecommended") is True
            for value in guardrails.values()
        ),
        traceId=request.trace_id,
    )


def assert_workspace_consistency(request: ExperimentAnalysisRequest) -> None:
    workspace_id = request.definition.workspace_id
    for exposure in request.exposures:
        if (
            exposure.workspace_id != workspace_id
            or exposure.experiment_id != request.definition.experiment_id
        ):
            raise ValueError("cross-workspace or cross-experiment exposure contamination")
    for outcome in request.outcomes:
        if (
            outcome.workspace_id != workspace_id
            or outcome.experiment_id != request.definition.experiment_id
        ):
            raise ValueError("cross-workspace or cross-experiment outcome contamination")


def attribute_outcomes(
    exposures: list[ExperimentExposure],
    outcomes: list[ExperimentOutcome],
    window_hours: int,
) -> dict[str, list[ExperimentOutcome]]:
    outcomes_by_entity: dict[tuple[str, str], list[ExperimentOutcome]] = defaultdict(list)
    for outcome in outcomes:
        outcomes_by_entity[(outcome.entity_type, outcome.entity_id)].append(outcome)
    attributed: dict[str, list[ExperimentOutcome]] = {}
    for exposure in exposures:
        window_seconds = window_hours * 60 * 60
        attributed[exposure.exposure_id] = [
            outcome
            for outcome in outcomes_by_entity[(exposure.entity_type, exposure.entity_id)]
            if 0 <= (outcome.observed_at - exposure.exposed_at).total_seconds() <= window_seconds
        ]
    return attributed


def summarize_variant(
    variant_key: str,
    exposures: list[ExperimentExposure],
    attributed: dict[str, list[ExperimentOutcome]],
    metric: str,
    metric_type: str,
) -> VariantMetricReport:
    values = [
        outcome.value
        for exposure in exposures
        for outcome in attributed.get(exposure.exposure_id, [])
        if outcome.metric == metric
    ]
    missing = max(0, len(exposures) - len(values))
    mean = math.fsum(values) / len(values) if values else 0.0
    variance = sample_variance(values)
    ci = (
        proportion_ci(mean, len(values))
        if metric_type == "proportion"
        else mean_ci(mean, variance, len(values))
    )
    return VariantMetricReport(
        variantKey=variant_key,
        sampleSize=len(exposures),
        observedCount=len(values),
        missingCount=missing,
        mean=round(mean, 6),
        variance=round(variance, 6),
        confidenceInterval=(round(ci[0], 6), round(ci[1], 6)),
    )


def compare_to_control(
    treatment: VariantMetricReport,
    control: VariantMetricReport,
    metric_type: str,
) -> EffectReport:
    effect = treatment.mean - control.mean
    relative = None if control.mean == 0 else effect / control.mean
    if metric_type == "proportion":
        standard_error = math.sqrt(
            treatment.mean * (1 - treatment.mean) / max(1, treatment.observed_count)
            + control.mean * (1 - control.mean) / max(1, control.observed_count)
        )
        method = "normal_approximation_two_proportion_difference"
    else:
        standard_error = math.sqrt(
            treatment.variance / max(1, treatment.observed_count)
            + control.variance / max(1, control.observed_count)
        )
        method = "welch_mean_difference_normal_approximation"
    ci = (effect - Z_95 * standard_error, effect + Z_95 * standard_error)
    return EffectReport(
        treatmentVariant=treatment.variant_key,
        controlVariant=control.variant_key,
        absoluteEffect=round(effect, 6),
        relativeEffect=None if relative is None else round(relative, 6),
        confidenceInterval=(round(ci[0], 6), round(ci[1], 6)),
        method=method,
    )


def sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = math.fsum(values) / len(values)
    return math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)


def proportion_ci(p_hat: float, observed_count: int) -> tuple[float, float]:
    if observed_count == 0:
        return (0.0, 0.0)
    margin = Z_95 * math.sqrt(p_hat * (1 - p_hat) / observed_count)
    return (max(0.0, p_hat - margin), min(1.0, p_hat + margin))


def mean_ci(mean: float, variance: float, observed_count: int) -> tuple[float, float]:
    if observed_count == 0:
        return (0.0, 0.0)
    margin = Z_95 * math.sqrt(variance / observed_count)
    return (mean - margin, mean + margin)


def minimum_sample_warnings(reports: list[VariantMetricReport]) -> list[str]:
    return [
        (
            f"Variant {report.variant_key} has sample size {report.sample_size}; "
            f"minimum guidance is {MIN_SAMPLE_PER_VARIANT}."
        )
        for report in reports
        if report.sample_size < MIN_SAMPLE_PER_VARIANT
    ]


def missing_outcome_warnings(reports: list[VariantMetricReport]) -> list[str]:
    return [
        f"Variant {report.variant_key} has {report.missing_count} missing primary outcomes."
        for report in reports
        if report.missing_count > 0
    ]


def summarize_guardrails(
    request: ExperimentAnalysisRequest,
    attributed: dict[str, list[ExperimentOutcome]],
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for metric in request.definition.guardrail_metrics:
        values = [
            outcome.value
            for outcomes in attributed.values()
            for outcome in outcomes
            if outcome.metric == metric
        ]
        mean = math.fsum(values) / len(values) if values else 0.0
        summary[metric] = {
            "observedCount": len(values),
            "mean": round(mean, 6),
            "rollbackRecommended": metric.endswith("_failure_rate") and mean > 0.05,
        }
    return summary


def segment_reports(
    request: ExperimentAnalysisRequest,
    attributed: dict[str, list[ExperimentOutcome]],
) -> dict[str, object]:
    reports: dict[str, object] = {}
    for segment in request.definition.segments:
        buckets: dict[str, int] = defaultdict(int)
        for exposure in request.exposures:
            key = exposure.segments.get(segment, "unknown")
            buckets[key] += len(
                [
                    outcome
                    for outcome in attributed[exposure.exposure_id]
                    if outcome.metric == request.definition.primary_metric
                ]
            )
        reports[segment] = dict(buckets)
    return reports

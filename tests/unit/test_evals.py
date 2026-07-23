from __future__ import annotations

import json
from pathlib import Path

import pytest

from squadpitch_ai.contracts.errors import ErrorCode
from squadpitch_ai.evals.dataset import DatasetValidationError, load_jsonl_dataset
from squadpitch_ai.evals.judges import DisabledJudge, JudgeUnavailableError
from squadpitch_ai.evals.models import ErrorLabel, EvalRunConfig
from squadpitch_ai.evals.reports import markdown_report
from squadpitch_ai.evals.runner import run_eval
from squadpitch_ai.evals.validators import deterministic_error_labels

DATASET = Path("datasets/eval/seed_ai_eval_sample_v1.jsonl")


def test_jsonl_schema_validation() -> None:
    samples = load_jsonl_dataset(DATASET)

    assert len(samples) == 50
    assert samples[0].schema_version == "ai-eval-sample-v1"
    assert samples[0].input.fixture_ref.startswith("private://eval-fixtures/")


def test_dataset_version_compatibility() -> None:
    with pytest.raises(DatasetValidationError):
        run_eval(
            EvalRunConfig(
                run_id="bad-version",
                dataset_path=str(DATASET),
                dataset_version="wrong-version",
                node_baseline_sha="node",
                candidate_sha="python",
            )
        )


def test_reproducibility() -> None:
    config = EvalRunConfig(
        run_id="repro",
        dataset_path=str(DATASET),
        dataset_version="seed-2026-07-22",
        node_baseline_sha="node",
        candidate_sha="python",
    )

    assert run_eval(config).model_dump() == run_eval(config).model_dump()


def test_metric_calculations_and_segmented_reporting() -> None:
    report = run_eval(
        EvalRunConfig(
            run_id="metrics",
            dataset_path=str(DATASET),
            dataset_version="seed-2026-07-22",
            node_baseline_sha="node",
            candidate_sha="python",
        )
    )

    assert report.sample_count == 50
    assert report.schema_validity < 1
    assert report.median_latency_ms is not None
    assert report.p95_latency_ms is not None
    assert report.mean_estimated_cost_cents is not None
    assert "task" in report.segments
    assert report.segments["task"]["content_draft_generation"] >= 1
    assert report.segments["language"]["en"] >= 1


def test_judge_failure() -> None:
    sample = load_jsonl_dataset(DATASET)[0]

    with pytest.raises(JudgeUnavailableError):
        DisabledJudge().score(sample)


def test_missing_expected_fields_maps_to_schema_invalid() -> None:
    sample = next(
        s for s in load_jsonl_dataset(DATASET) if s.sample_id == "sites_page_generation_037"
    )

    labels = deterministic_error_labels(sample)

    assert ErrorCode.SCHEMA_INVALID in {label.code for label in labels}


def test_privacy_rules(tmp_path: Path) -> None:
    raw = json.loads(DATASET.read_text(encoding="utf-8").splitlines()[0])
    raw["input"]["fixture_ref"] = "file:///tmp/raw-customer.json"
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(DatasetValidationError):
        load_jsonl_dataset(bad)


def test_error_taxonomy_mapping() -> None:
    label = ErrorLabel(
        code=ErrorCode.TENANT_LEAKAGE,
        severity="blocker",
        dimension="tenant_isolation",
        source_field="context_refs",
        artifact_field="artifact",
        notes="Safe note",
    )

    assert label.code == ErrorCode.TENANT_LEAKAGE
    with pytest.raises(ValueError):
        ErrorLabel(
            code=ErrorCode.TENANT_LEAKAGE,
            severity="medium",
            dimension="tenant_isolation",
            source_field="context_refs",
            artifact_field="artifact",
            notes="Wrong severity",
        )


def test_release_report_generation(tmp_path: Path) -> None:
    report = run_eval(
        EvalRunConfig(
            run_id="report",
            dataset_path=str(DATASET),
            dataset_version="seed-2026-07-22",
            node_baseline_sha="node",
            candidate_sha="python",
        ),
        output_dir=tmp_path,
    )

    assert (tmp_path / "eval-report.json").exists()
    assert (tmp_path / "eval-report.csv").exists()
    assert (tmp_path / "eval-report.md").exists()
    assert "Decision:" in markdown_report(report)


def test_node_adapter_mocks() -> None:
    sample = load_jsonl_dataset(DATASET)[0]
    report = run_eval(
        EvalRunConfig(
            run_id="adapter",
            dataset_path=str(DATASET),
            dataset_version="seed-2026-07-22",
            node_baseline_sha="node-sha",
            candidate_sha="python-sha",
            adapter_mode="mock",
        )
    )

    assert sample.baseline_result.implementation == "node"
    assert report.baseline_sha == "node-sha"
    assert report.candidate_sha == "python-sha"


def test_blocker_prevents_shadow_decision() -> None:
    report = run_eval(
        EvalRunConfig(
            run_id="shadow",
            dataset_path=str(DATASET),
            dataset_version="seed-2026-07-22",
            node_baseline_sha="node",
            candidate_sha="python",
            release_level="shadow",
        )
    )

    assert report.blocker_failures > 0
    assert report.decision == "block"

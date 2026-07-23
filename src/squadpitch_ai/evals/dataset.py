from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from squadpitch_ai.evals.models import EVAL_SAMPLE_SCHEMA_VERSION, EvalSample


class DatasetValidationError(Exception):
    pass


def load_jsonl_dataset(path: str | Path) -> list[EvalSample]:
    samples: list[EvalSample] = []
    dataset_path = Path(path)
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if raw.get("schema_version") != EVAL_SAMPLE_SCHEMA_VERSION:
                raise DatasetValidationError(
                    f"line {line_number}: unsupported schema_version {raw.get('schema_version')}"
                )
            samples.append(EvalSample.model_validate(raw))
        except (json.JSONDecodeError, ValidationError, DatasetValidationError) as exc:
            raise DatasetValidationError(f"{dataset_path}:{line_number}: {exc}") from exc
    if not samples:
        raise DatasetValidationError(f"{dataset_path}: dataset is empty")
    versions = {sample.dataset_version for sample in samples}
    if len(versions) != 1:
        raise DatasetValidationError("dataset must contain exactly one dataset_version")
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise DatasetValidationError("dataset contains duplicate sample_id values")
    return samples


def validate_dataset_version(samples: list[EvalSample], expected_version: str) -> None:
    actual = {sample.dataset_version for sample in samples}
    if actual != {expected_version}:
        raise DatasetValidationError(
            f"expected dataset_version {expected_version}; got {', '.join(sorted(actual))}"
        )

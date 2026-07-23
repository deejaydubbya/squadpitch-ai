from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from squadpitch_ai.brand_quality.models import QUALITY_LABELS, BrandQualityDatasetExample

DATASET_SCHEMA_VERSION: Literal["brand-quality-dataset.v1"] = "brand-quality-dataset.v1"
LEAKAGE_BANNED_FIELDS = {
    "rawPrompt",
    "rawOutput",
    "email",
    "phone",
    "publishedAt",
    "conversionCount",
    "futureEngagement",
}


class BrandQualityDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["brand-quality-dataset.v1"] = Field(alias="schemaVersion")
    dataset_version: str = Field(alias="datasetVersion")
    examples: list[BrandQualityDatasetExample]
    label_guide: dict[str, object] = Field(alias="labelGuide")
    split_strategy: dict[str, object] = Field(alias="splitStrategy")
    leakage_controls: dict[str, object] = Field(alias="leakageControls")
    class_balance: dict[str, object] = Field(alias="classBalance")
    privacy_policy: dict[str, object] = Field(alias="privacyPolicy")


def build_brand_quality_dataset(rows: list[dict[str, Any]]) -> BrandQualityDataset:
    assert_no_leakage_fields(rows)
    examples = [BrandQualityDatasetExample.model_validate(row) for row in rows]
    examples.sort(key=lambda example: (example.created_at, example.example_id))
    return BrandQualityDataset(
        schemaVersion=DATASET_SCHEMA_VERSION,
        datasetVersion=dataset_version_for(examples),
        examples=examples,
        labelGuide={
            "labels": list(QUALITY_LABELS),
            "scale": "0 or 1 per dimension for current supervised seed set",
            "interAnnotator": (
                "Resolve disagreements by dimension-specific examples and reviewer notes"
            ),
        },
        splitStrategy={"type": "time_aware", "train": "70%", "validation": "15%", "test": "15%"},
        leakageControls={
            "bannedFields": sorted(LEAKAGE_BANNED_FIELDS),
            "rawPrompts": False,
            "rawOutputs": False,
            "postTreatmentMetrics": False,
        },
        classBalance=class_balance(examples),
        privacyPolicy={
            "sources": [
                "synthetic examples",
                "explicitly approved sanitized edits",
                "accepted/rejected draft metadata",
                "structured user corrections",
                "brand constraints",
            ],
            "syntheticPortfolioExportOnly": True,
            "piiMinimized": True,
        },
    )


def split_dataset(dataset: BrandQualityDataset) -> dict[str, list[BrandQualityDatasetExample]]:
    ordered = sorted(dataset.examples, key=lambda example: (example.created_at, example.example_id))
    train_end = int(len(ordered) * 0.7)
    validation_end = int(len(ordered) * 0.85)
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


def assert_no_leakage_fields(rows: list[dict[str, Any]]) -> None:
    leaked = sorted({key for row in rows for key in row if key in LEAKAGE_BANNED_FIELDS})
    if leaked:
        raise ValueError(f"leakage fields are not allowed: {', '.join(leaked)}")


def dataset_version_for(examples: list[BrandQualityDatasetExample]) -> str:
    encoded = json.dumps(
        [
            {
                "exampleId": example.example_id,
                "workspaceId": example.workspace_id,
                "createdAt": example.created_at.isoformat(),
                "labels": example.labels,
            }
            for example in examples
        ],
        sort_keys=True,
    )
    return "brand-quality-dataset-" + hashlib.sha256(encoded.encode()).hexdigest()[:12]


def class_balance(examples: list[BrandQualityDatasetExample]) -> dict[str, object]:
    return {
        label: {
            "positive": sum(1 for example in examples if example.labels.get(label) == 1),
            "negative": sum(1 for example in examples if example.labels.get(label) == 0),
        }
        for label in QUALITY_LABELS
    }

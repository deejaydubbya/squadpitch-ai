from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from squadpitch_ai.autopilot_ranker.features import FEATURE_NAMES, assert_no_target_leakage
from squadpitch_ai.autopilot_ranker.models import AutopilotRankingCandidate

DATASET_SCHEMA_VERSION: Literal["autopilot-ranking-dataset.v1"] = "autopilot-ranking-dataset.v1"
POSITIVE_LABELS = {"APPROVED", "PUBLISHED", "EXCEEDED_BASELINE_ENGAGEMENT", "PRODUCED_INQUIRY"}
NEGATIVE_LABELS = {"DISMISSED", "EXPIRED", "REJECTED"}


class TrainingExample(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    candidate: AutopilotRankingCandidate
    label: Literal[0, 1]
    split_key: str = Field(alias="splitKey")
    event_time: datetime = Field(alias="eventTime")


class TrainingDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["autopilot-ranking-dataset.v1"] = Field(alias="schemaVersion")
    dataset_version: str = Field(alias="datasetVersion")
    feature_schema_version: str = Field(alias="featureSchemaVersion")
    examples: list[TrainingExample]
    label_policy: dict[str, object] = Field(alias="labelPolicy")
    leakage_analysis: dict[str, object] = Field(alias="leakageAnalysis")
    missing_data_policy: dict[str, object] = Field(alias="missingDataPolicy")
    split_strategy: dict[str, object] = Field(alias="splitStrategy")
    privacy: dict[str, object]


def build_training_dataset(rows: list[dict[str, Any]]) -> TrainingDataset:
    assert_no_target_leakage(FEATURE_NAMES)
    examples: list[TrainingExample] = []
    for row in rows:
        label_name = str(row.get("label", "")).upper()
        if label_name not in POSITIVE_LABELS and label_name not in NEGATIVE_LABELS:
            continue
        candidate = AutopilotRankingCandidate.model_validate(row["candidate"])
        examples.append(
            TrainingExample(
                candidate=candidate,
                label=1 if label_name in POSITIVE_LABELS else 0,
                splitKey=f"{candidate.workspace_id}:{candidate.candidate_id}",
                eventTime=candidate.detected_at,
            )
        )
    examples.sort(key=lambda example: (example.event_time, example.split_key))
    return TrainingDataset(
        schemaVersion=DATASET_SCHEMA_VERSION,
        datasetVersion=dataset_version_for(examples),
        featureSchemaVersion="autopilot-ranking-features.v1",
        examples=examples,
        labelPolicy={
            "positive": sorted(POSITIVE_LABELS),
            "negative": sorted(NEGATIVE_LABELS),
            "ambiguousRows": "dropped",
        },
        leakageAnalysis={
            "bannedFeatures": [
                "approved",
                "dismissed",
                "published",
                "conversion",
                "generatedDraftIds",
                "decidedAt",
                "publishedAt",
            ],
            "result": "pass",
        },
        missingDataPolicy={
            "numeric": "median/default imputation at scoring time",
            "categorical": "unknown bucket or one-hot zero",
        },
        splitStrategy={
            "type": "time_aware",
            "train": "oldest 70%",
            "validation": "next 15%",
            "test": "newest 15%",
        },
        privacy={
            "rawPrompts": False,
            "rawOutputs": False,
            "workspaceScoped": True,
            "syntheticPortfolioExportOnly": True,
        },
    )


def split_dataset(dataset: TrainingDataset) -> dict[str, list[TrainingExample]]:
    ordered = sorted(dataset.examples, key=lambda example: (example.event_time, example.split_key))
    n = len(ordered)
    train_end = int(n * 0.7)
    validation_end = int(n * 0.85)
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


def dataset_version_for(examples: list[TrainingExample]) -> str:
    encoded = json.dumps(
        [
            {
                "candidateId": example.candidate.candidate_id,
                "workspaceId": example.candidate.workspace_id,
                "label": example.label,
                "eventTime": example.event_time.isoformat(),
            }
            for example in examples
        ],
        sort_keys=True,
    )
    return "autopilot-ranking-dataset-" + hashlib.sha256(encoded.encode()).hexdigest()[:12]

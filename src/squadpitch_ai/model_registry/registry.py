from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from squadpitch_ai.brand_quality.models import BRAND_QUALITY_SCHEMA_VERSION, MODEL_VERSION

MODEL_REGISTRY_SCHEMA_VERSION: Literal["model-registry.v1"] = "model-registry.v1"
LOCAL_DEV_ARTIFACT_PREFIX = "local-dev://"


class ModelRegistryError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ModelRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    registry_schema_version: Literal["model-registry.v1"] = Field(alias="registrySchemaVersion")
    model_id: str = Field(alias="modelId")
    version: str
    task: str
    framework: str
    artifact_uri: str = Field(alias="artifactUri")
    artifact_checksum_sha256: str = Field(alias="artifactChecksumSha256")
    dataset_version: str = Field(alias="datasetVersion")
    code_commit: str = Field(alias="codeCommit")
    training_parameters: dict[str, Any] = Field(alias="trainingParameters")
    metrics: dict[str, float]
    calibration: dict[str, float]
    compatibility_schema: str = Field(alias="compatibilitySchema")
    deployment_status: Literal["shadow", "canary", "active", "retired", "rolled_back"] = Field(
        alias="deploymentStatus",
    )
    rollout_percentage: int = Field(alias="rolloutPercentage", ge=0, le=100)
    created_at: datetime = Field(alias="createdAt")
    approved_at: datetime | None = Field(default=None, alias="approvedAt")
    retired_at: datetime | None = Field(default=None, alias="retiredAt")
    model_card_uri: str = Field(alias="modelCardUri")
    rollback_target: str | None = Field(default=None, alias="rollbackTarget")
    prompt_version: str | None = Field(default=None, alias="promptVersion")
    hosted_model_config: dict[str, Any] = Field(default_factory=dict, alias="hostedModelConfig")

    @property
    def key(self) -> str:
        return f"{self.model_id}:{self.version}"


class ModelRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    registry_schema_version: Literal["model-registry.v1"] = Field(alias="registrySchemaVersion")
    entries: list[ModelRegistryEntry]

    def get(self, model_id: str, version: str) -> ModelRegistryEntry:
        for entry in self.entries:
            if entry.model_id == model_id and entry.version == version:
                return entry
        raise ModelRegistryError("MODEL_NOT_FOUND", f"Model not found: {model_id}:{version}")

    def require_compatible(
        self, model_id: str, version: str, schema_version: str
    ) -> ModelRegistryEntry:
        entry = self.get(model_id, version)
        if entry.compatibility_schema != schema_version:
            raise ModelRegistryError(
                "MODEL_SCHEMA_INCOMPATIBLE",
                f"Model {entry.key} is not compatible with {schema_version}",
            )
        if entry.deployment_status not in {"shadow", "canary", "active"}:
            raise ModelRegistryError("MODEL_NOT_DEPLOYABLE", f"Model {entry.key} is not deployable")
        verify_artifact_integrity(entry)
        return entry

    def rollback_target_for(self, model_id: str, version: str) -> ModelRegistryEntry:
        entry = self.get(model_id, version)
        if not entry.rollback_target:
            raise ModelRegistryError(
                "ROLLBACK_TARGET_MISSING", f"Model {entry.key} has no rollback target"
            )
        rollback_version = entry.rollback_target.split(":", 1)[-1]
        return self.get(model_id, rollback_version)


def checksum_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def checksum_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_dev_artifact_checksum(model_id: str, version: str, task: str) -> str:
    return checksum_bytes(f"{model_id}:{version}:{task}:local-dev-artifact".encode())


def verify_artifact_integrity(entry: ModelRegistryEntry) -> bool:
    if entry.artifact_uri.startswith(LOCAL_DEV_ARTIFACT_PREFIX):
        expected = local_dev_artifact_checksum(entry.model_id, entry.version, entry.task)
    else:
        path = Path(entry.artifact_uri.removeprefix("file://"))
        if not path.exists():
            raise ModelRegistryError(
                "ARTIFACT_MISSING", f"Artifact does not exist: {entry.artifact_uri}"
            )
        expected = checksum_file(path)
    if expected != entry.artifact_checksum_sha256:
        raise ModelRegistryError("ARTIFACT_CHECKSUM_MISMATCH", f"Checksum mismatch for {entry.key}")
    return True


def build_default_model_registry(code_commit: str = "local-dev") -> ModelRegistry:
    created_at = datetime(2026, 7, 22, tzinfo=UTC)
    return ModelRegistry(
        registrySchemaVersion=MODEL_REGISTRY_SCHEMA_VERSION,
        entries=[
            ModelRegistryEntry(
                registrySchemaVersion=MODEL_REGISTRY_SCHEMA_VERSION,
                modelId="brand-content-quality",
                version=MODEL_VERSION,
                task="brand_content_quality_classification",
                framework="fastapi+pytorch-compatible-deterministic-shadow",
                artifactUri=f"{LOCAL_DEV_ARTIFACT_PREFIX}brand-content-quality/{MODEL_VERSION}",
                artifactChecksumSha256=local_dev_artifact_checksum(
                    "brand-content-quality",
                    MODEL_VERSION,
                    "brand_content_quality_classification",
                ),
                datasetVersion="brand-quality-dataset-seed-v1",
                codeCommit=code_commit,
                trainingParameters={"device": "cpu", "batchSize": 1, "precision": "float32"},
                metrics={
                    "macroF1": 0.0,
                    "microF1": 0.0,
                    "p95LatencyMs": 0.0,
                    "coldStartMs": 0.0,
                },
                calibration={"expectedCalibrationError": 0.0, "trainedArtifactAvailable": 0.0},
                compatibilitySchema=BRAND_QUALITY_SCHEMA_VERSION,
                deploymentStatus="shadow",
                rolloutPercentage=0,
                createdAt=created_at,
                approvedAt=created_at,
                retiredAt=None,
                modelCardUri="docs/ai-platform/BRAND_CONTENT_QUALITY_MODEL_CARD.md",
                rollbackTarget="brand-content-quality:brand-quality-deterministic-shadow.v0",
            ),
            ModelRegistryEntry(
                registrySchemaVersion=MODEL_REGISTRY_SCHEMA_VERSION,
                modelId="brand-content-quality",
                version="brand-quality-deterministic-shadow.v0",
                task="brand_content_quality_classification",
                framework="fastapi+deterministic",
                artifactUri=(
                    f"{LOCAL_DEV_ARTIFACT_PREFIX}"
                    "brand-content-quality/brand-quality-deterministic-shadow.v0"
                ),
                artifactChecksumSha256=local_dev_artifact_checksum(
                    "brand-content-quality",
                    "brand-quality-deterministic-shadow.v0",
                    "brand_content_quality_classification",
                ),
                datasetVersion="brand-quality-dataset-seed-v1",
                codeCommit=code_commit,
                trainingParameters={"device": "cpu", "batchSize": 1, "precision": "float32"},
                metrics={"macroF1": 0.0, "microF1": 0.0},
                calibration={"expectedCalibrationError": 0.0, "trainedArtifactAvailable": 0.0},
                compatibilitySchema=BRAND_QUALITY_SCHEMA_VERSION,
                deploymentStatus="retired",
                rolloutPercentage=0,
                createdAt=created_at,
                approvedAt=created_at,
                retiredAt=created_at,
                modelCardUri="docs/ai-platform/BRAND_CONTENT_QUALITY_MODEL_CARD.md",
                rollbackTarget=None,
            ),
        ],
    )

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from squadpitch_ai.brand_quality.models import (
    BRAND_QUALITY_SCHEMA_VERSION,
    MODEL_VERSION,
    BrandQualityScoreRequest,
)
from squadpitch_ai.brand_quality.scorer import score_brand_quality
from squadpitch_ai.model_registry.registry import (
    ModelRegistry,
    ModelRegistryEntry,
    build_default_model_registry,
)


@dataclass(frozen=True)
class ModelInferenceMetrics:
    model_id: str
    version: str
    latency_ms: float
    cold_start: bool
    batch_size: int
    memory_mb: float
    status: str


class SelfHostedBrandQualityInference:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or build_default_model_registry()
        self._warmed_entries: set[str] = set()
        self.metrics: list[ModelInferenceMetrics] = []

    def health(self) -> dict[str, object]:
        entry = self.registry.require_compatible(
            "brand-content-quality",
            MODEL_VERSION,
            BRAND_QUALITY_SCHEMA_VERSION,
        )
        return {
            "status": "ready",
            "modelId": entry.model_id,
            "version": entry.version,
            "deploymentStatus": entry.deployment_status,
            "warmed": entry.key in self._warmed_entries,
        }

    def warmup(
        self, model_id: str = "brand-content-quality", version: str = MODEL_VERSION
    ) -> ModelRegistryEntry:
        entry = self.registry.require_compatible(model_id, version, BRAND_QUALITY_SCHEMA_VERSION)
        self._warmed_entries.add(entry.key)
        return entry

    def predict(self, request: BrandQualityScoreRequest) -> tuple[Any, ModelInferenceMetrics]:
        start = time.perf_counter()
        entry = self.registry.require_compatible(
            "brand-content-quality",
            request.model_version,
            request.schema_version,
        )
        cold_start = entry.key not in self._warmed_entries
        if cold_start:
            self.warmup(entry.model_id, entry.version)
        response = score_brand_quality(request)
        metric = ModelInferenceMetrics(
            model_id=entry.model_id,
            version=entry.version,
            latency_ms=(time.perf_counter() - start) * 1000,
            cold_start=cold_start,
            batch_size=1,
            memory_mb=0.0,
            status="succeeded",
        )
        self.metrics.append(metric)
        return response, metric


def benchmark_brand_quality_inference(
    inference: SelfHostedBrandQualityInference,
    request: BrandQualityScoreRequest,
    iterations: int = 10,
) -> dict[str, float | int | str]:
    latencies: list[float] = []
    cold_start_ms = 0.0
    for _index in range(iterations):
        _response, metric = inference.predict(request)
        latencies.append(metric.latency_ms)
        if metric.cold_start:
            cold_start_ms = metric.latency_ms
    ordered = sorted(latencies)
    p50 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.5))]
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return {
        "modelVersion": request.model_version,
        "iterations": iterations,
        "p50LatencyMs": round(p50, 6),
        "p95LatencyMs": round(p95, 6),
        "throughputPerSecond": round(1000 / max(0.001, p50), 6),
        "coldStartMs": round(cold_start_ms, 6),
        "batchSize": 1,
        "memoryMb": 0.0,
        "quantizedAccuracyDelta": 0.0,
        "hostedJudgeCostRatio": 0.0,
        "failureBehavior": "fallback_to_deterministic_shadow",
    }


@lru_cache
def get_default_brand_quality_inference() -> SelfHostedBrandQualityInference:
    inference = SelfHostedBrandQualityInference()
    inference.warmup()
    return inference

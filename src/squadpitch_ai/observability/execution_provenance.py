from __future__ import annotations

import os
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def execution_provenance(
    *,
    operation: str,
    implementation: str,
    trace_id: str,
    latency_ms: float,
    inference_mode: str,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
    model: str | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Safe metadata describing the implementation that produced a response."""
    value: dict[str, Any] = {
        "service": "squadpitch-ai",
        "implementation": implementation,
        "fallbackUsed": fallback_used,
        "traceId": trace_id,
        "latencyMs": max(0, round(latency_ms)),
        "inferenceMode": inference_mode,
    }
    optional = {
        "fallbackReason": fallback_reason,
        "serviceVersion": os.getenv("SP_AI_BUILD_SHA") or os.getenv("FLY_IMAGE_REF"),
        "model": model,
        "modelVersion": model_version,
    }
    value.update({key: item for key, item in optional.items() if item})
    logger.info("ai.execution.completed", operation=operation, **value)
    return value

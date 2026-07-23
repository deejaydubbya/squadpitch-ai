from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

AI_RELIABILITY_SLOS: dict[str, dict[str, float | str]] = {
    "python_api_availability": {"target": 0.995, "window": "30d"},
    "worker_success_rate": {"target": 0.99, "window": "7d"},
    "retrieval_latency_p95_ms": {"target": 250.0, "window": "1h"},
    "agent_plan_latency_p95_ms": {"target": 2000.0, "window": "1h"},
    "content_scoring_latency_p95_ms": {"target": 500.0, "window": "1h"},
    "eval_job_completion_rate": {"target": 0.98, "window": "24h"},
    "queue_delay_p95_ms": {"target": 30000.0, "window": "1h"},
    "trace_completeness_rate": {"target": 0.995, "window": "24h"},
    "cost_budget_compliance_rate": {"target": 0.99, "window": "30d"},
}


class CircuitBreakerOpen(RuntimeError):
    pass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    base_delay_seconds: float
    max_delay_seconds: float
    retry_on: frozenset[str]
    safe_operations_only: bool = True

    def should_retry(self, *, attempt: int, error_code: str, operation_safe: bool) -> bool:
        if self.safe_operations_only and not operation_safe:
            return False
        return attempt < self.max_attempts and error_code in self.retry_on


def backoff_delay_seconds(
    attempt: int,
    *,
    base_delay_seconds: float = 0.1,
    max_delay_seconds: float = 5.0,
    jitter_seed: str = "",
) -> float:
    exponential = min(max_delay_seconds, base_delay_seconds * (2 ** max(0, attempt - 1)))
    digest = hashlib.sha256(f"{jitter_seed}:{attempt}".encode()).hexdigest()
    jitter = int(digest[:8], 16) / 0xFFFFFFFF
    return float(round(exponential * (0.5 + jitter), 6))


@dataclass
class CircuitBreaker:
    failure_threshold: int
    recovery_seconds: float
    opened_at: float | None = None
    failure_count: int = 0

    def allow(self, now: float | None = None) -> bool:
        current = now if now is not None else time.monotonic()
        if self.opened_at is None:
            return True
        return current - self.opened_at >= self.recovery_seconds

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened_at = None

    def record_failure(self, now: float | None = None) -> None:
        current = now if now is not None else time.monotonic()
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.opened_at = current

    def guard(self, fn: Callable[[], Any], now: float | None = None) -> Any:
        if not self.allow(now):
            raise CircuitBreakerOpen("Circuit breaker is open")
        try:
            result = fn()
        except Exception:
            self.record_failure(now)
            raise
        self.record_success()
        return result


@dataclass
class Bulkhead:
    max_in_flight: int
    in_flight: int = 0

    def acquire(self) -> bool:
        if self.in_flight >= self.max_in_flight:
            return False
        self.in_flight += 1
        return True

    def release(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)


@dataclass
class WorkspaceQuota:
    max_operations: int
    window_seconds: int
    _events: dict[str, list[float]] = field(default_factory=dict)

    def allow(self, workspace_id: str, now: float | None = None) -> bool:
        current = now if now is not None else time.time()
        retained = [
            timestamp
            for timestamp in self._events.get(workspace_id, [])
            if current - timestamp <= self.window_seconds
        ]
        if len(retained) >= self.max_operations:
            self._events[workspace_id] = retained
            return False
        retained.append(current)
        self._events[workspace_id] = retained
        return True


def classify_failure_drill(
    scenario: str,
) -> dict[str, str | bool]:
    mapping: dict[str, dict[str, str | bool]] = {
        "python_unavailable": {
            "expectedBehavior": "Node falls back or returns typed provider-unavailable error.",
            "retryable": True,
            "rollback": "Disable Python-backed feature flags.",
        },
        "redis_unavailable": {
            "expectedBehavior": (
                "Pause enqueueing, surface queue backlog alert, preserve sync fallbacks."
            ),
            "retryable": True,
            "rollback": "Disable queued AI ingestion and workers.",
        },
        "postgres_unavailable": {
            "expectedBehavior": (
                "Readiness fails closed and product writes stop before AI side effects."
            ),
            "retryable": True,
            "rollback": "Keep feature flags disabled until database recovers.",
        },
        "provider_timeout": {
            "expectedBehavior": "Retry only safe read-only operations, then degrade gracefully.",
            "retryable": True,
            "rollback": "Route to deterministic fallback or Node baseline.",
        },
        "invalid_model_artifact": {
            "expectedBehavior": (
                "Checksum check fails readiness and model registry rollback target is used."
            ),
            "retryable": False,
            "rollback": "Pin previous model version.",
        },
        "schema_mismatch": {
            "expectedBehavior": "Reject request with schema error before product action.",
            "retryable": False,
            "rollback": "Restore previous contract version.",
        },
        "clock_skew": {
            "expectedBehavior": "Signed request rejected as expired or future dated.",
            "retryable": False,
            "rollback": "Fix NTP before re-enabling calls.",
        },
        "replay_attack": {
            "expectedBehavior": "Nonce replay is rejected.",
            "retryable": False,
            "rollback": "Rotate service auth secret if replay source is unknown.",
        },
        "duplicate_job": {
            "expectedBehavior": "Idempotency key suppresses duplicate side effects.",
            "retryable": False,
            "rollback": "Disable enqueueing and inspect dedupe key.",
        },
        "trace_backend_unavailable": {
            "expectedBehavior": (
                "Work can complete but trace completeness SLO is breached and alerted."
            ),
            "retryable": True,
            "rollback": "Do not promote release gates until traces recover.",
        },
    }
    return mapping.get(
        scenario,
        {
            "expectedBehavior": "Classify before retrying.",
            "retryable": False,
            "rollback": "Disable affected feature flag.",
        },
    )


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil((pct / 100) * len(ordered)) - 1)
    return ordered[index]


def load_test_summary(
    samples: list[dict[str, float | int | str]],
) -> dict[str, float | int | None]:
    latencies = [float(sample["latencyMs"]) for sample in samples if "latencyMs" in sample]
    errors = [sample for sample in samples if sample.get("status") == "error"]
    duration_ms = max(
        1.0, max((float(sample.get("elapsedMs", 0)) for sample in samples), default=1.0)
    )
    return {
        "sampleCount": len(samples),
        "p50LatencyMs": percentile(latencies, 50),
        "p95LatencyMs": percentile(latencies, 95),
        "p99LatencyMs": percentile(latencies, 99),
        "throughputPerSecond": round(len(samples) / (duration_ms / 1000), 6),
        "errorRate": 0.0 if not samples else len(errors) / len(samples),
        "meanLatencyMs": None if not latencies else round(statistics.fmean(latencies), 6),
    }

from __future__ import annotations

import pytest

from squadpitch_ai.reliability import (
    AI_RELIABILITY_SLOS,
    Bulkhead,
    CircuitBreaker,
    CircuitBreakerOpen,
    RetryPolicy,
    WorkspaceQuota,
    backoff_delay_seconds,
    classify_failure_drill,
    load_test_summary,
)


def test_slo_catalog_covers_required_ai_platform_targets() -> None:
    assert "python_api_availability" in AI_RELIABILITY_SLOS
    assert "worker_success_rate" in AI_RELIABILITY_SLOS
    assert "retrieval_latency_p95_ms" in AI_RELIABILITY_SLOS
    assert "agent_plan_latency_p95_ms" in AI_RELIABILITY_SLOS
    assert "content_scoring_latency_p95_ms" in AI_RELIABILITY_SLOS
    assert "eval_job_completion_rate" in AI_RELIABILITY_SLOS
    assert "queue_delay_p95_ms" in AI_RELIABILITY_SLOS
    assert "trace_completeness_rate" in AI_RELIABILITY_SLOS
    assert "cost_budget_compliance_rate" in AI_RELIABILITY_SLOS


def test_retry_policy_and_jitter_are_safe_and_deterministic() -> None:
    policy = RetryPolicy(
        max_attempts=3,
        base_delay_seconds=0.1,
        max_delay_seconds=1,
        retry_on=frozenset({"PROVIDER_TIMEOUT"}),
    )

    assert policy.should_retry(attempt=1, error_code="PROVIDER_TIMEOUT", operation_safe=True)
    assert not policy.should_retry(attempt=3, error_code="PROVIDER_TIMEOUT", operation_safe=True)
    assert not policy.should_retry(attempt=1, error_code="PROVIDER_TIMEOUT", operation_safe=False)
    assert backoff_delay_seconds(2, jitter_seed="workspace-a") == backoff_delay_seconds(
        2,
        jitter_seed="workspace-a",
    )


def test_circuit_breaker_bulkhead_and_quota() -> None:
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=10)
    breaker.record_failure(now=100)
    assert breaker.allow(now=101)
    breaker.record_failure(now=102)
    assert not breaker.allow(now=103)
    assert breaker.allow(now=113)
    with pytest.raises(CircuitBreakerOpen):
        breaker.guard(lambda: "blocked", now=103)

    bulkhead = Bulkhead(max_in_flight=1)
    assert bulkhead.acquire()
    assert not bulkhead.acquire()
    bulkhead.release()
    assert bulkhead.acquire()

    quota = WorkspaceQuota(max_operations=2, window_seconds=60)
    assert quota.allow("workspace-a", now=1)
    assert quota.allow("workspace-a", now=2)
    assert not quota.allow("workspace-a", now=3)
    assert quota.allow("workspace-a", now=100)


def test_failure_drills_and_load_summary() -> None:
    drill = classify_failure_drill("invalid_model_artifact")
    assert drill["retryable"] is False
    assert "rollback" in drill

    summary = load_test_summary(
        [
            {"latencyMs": 10, "elapsedMs": 100, "status": "ok"},
            {"latencyMs": 30, "elapsedMs": 200, "status": "error"},
            {"latencyMs": 20, "elapsedMs": 300, "status": "ok"},
        ]
    )
    assert summary["p50LatencyMs"] == 20
    assert summary["p95LatencyMs"] == 30
    assert summary["errorRate"] == pytest.approx(1 / 3)

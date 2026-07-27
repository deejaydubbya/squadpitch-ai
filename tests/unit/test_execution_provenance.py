from pytest import MonkeyPatch

from squadpitch_ai.observability.execution_provenance import execution_provenance


def test_execution_provenance_reports_configured_build(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("SP_AI_BUILD_SHA", "abc123")

    value = execution_provenance(
        operation="campaign_ops_plan",
        implementation="campaign_ops_v1",
        trace_id="trace-1",
        latency_ms=12.6,
        inference_mode="deterministic",
    )

    assert value == {
        "service": "squadpitch-ai",
        "implementation": "campaign_ops_v1",
        "fallbackUsed": False,
        "traceId": "trace-1",
        "latencyMs": 13,
        "inferenceMode": "deterministic",
        "serviceVersion": "abc123",
    }


def test_execution_provenance_truthfully_reports_internal_fallback() -> None:
    value = execution_provenance(
        operation="brand_quality_score",
        implementation="deterministic_brand_quality_v1",
        trace_id="trace-2",
        latency_ms=4,
        inference_mode="deterministic",
        fallback_used=True,
        fallback_reason="model_unavailable",
        model="brand-quality",
    )

    assert value["fallbackUsed"] is True
    assert value["fallbackReason"] == "model_unavailable"
    assert value["model"] == "brand-quality"
    assert "modelVersion" not in value

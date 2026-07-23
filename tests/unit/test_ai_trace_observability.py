from __future__ import annotations

import pytest

from squadpitch_ai.observability import build_ai_trace_envelope, redact_trace_value


def base_trace(**overrides: object) -> dict[str, object]:
    trace: dict[str, object] = {
        "requestId": "req-1",
        "traceId": "trace-1",
        "workspaceId": "workspace-1",
        "actorUserId": "user-1",
        "taskType": "draft_content",
        "featureFlags": {"ai_action_proposals_enabled": True},
        "provider": "openai",
        "model": "gpt-4o-mini",
        "promptVersion": "draft.v1",
        "schemaVersion": "draft-content-proposal.v1",
        "retrievalRunId": "retrieval-1",
        "retrievedSources": [{"sourceId": "property-1", "score": 0.9}],
        "steps": [{"name": "retrieval", "prompt": "Use jane@example.com facts only"}],
        "validationResults": {"valid": True, "output": "raw output"},
        "retryCount": 1,
        "promptTokens": 100,
        "completionTokens": 50,
        "estimatedCostCents": 4.2,
        "latencyMs": 250,
        "status": "SUCCEEDED",
        "humanOutcome": "approved",
        "downstreamCampaignIds": ["campaign-1"],
        "downstreamDraftIds": ["draft-1"],
        "outcomePointers": {"proposalId": "proposal-1"},
        "releaseGateStage": "SHADOW",
    }
    trace.update(overrides)
    return trace


def test_ai_trace_envelope_is_complete_and_propagates_ids() -> None:
    envelope = build_ai_trace_envelope(base_trace())

    assert envelope.request_id == "req-1"
    assert envelope.trace_id == "trace-1"
    assert envelope.workspace_id == "workspace-1"
    assert envelope.task_type == "draft_content"
    assert envelope.retrieval_run_id == "retrieval-1"
    assert envelope.raw_content_captured is False
    assert envelope.redaction_version == "ai-trace-redaction.v1"


def test_ai_trace_redacts_raw_content_and_pii() -> None:
    redacted = redact_trace_value(
        {
            "prompt": "Email jane@example.com and call 919-555-1212.",
            "authorization": "Bearer abcdefghijklmnop",
            "sourceId": "property-1",
        }
    )

    assert redacted["prompt"]["redacted"] is True
    assert redacted["authorization"] == "[REDACTED_SECRET]"
    assert redacted["sourceId"] == "property-1"


def test_ai_trace_does_not_store_secret_values() -> None:
    envelope = build_ai_trace_envelope(
        base_trace(steps=[{"name": "bad", "note": "sk-secret1234567890"}])
    )

    serialized = envelope.model_dump_json(by_alias=True)
    assert "sk-secret1234567890" not in serialized
    assert "[REDACTED_PHONE]" in serialized


def test_ai_trace_never_accepts_raw_content_capture() -> None:
    with pytest.raises(ValueError):
        build_ai_trace_envelope(base_trace(rawContentCaptured=True))

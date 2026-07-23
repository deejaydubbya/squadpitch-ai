from __future__ import annotations

from typing import Protocol

from squadpitch_ai.evals.models import EvalResult, EvalSample


class EvalAdapter(Protocol):
    def run(self, sample: EvalSample) -> EvalResult: ...


class MetadataOnlyNodeBaselineAdapter:
    def __init__(self, code_sha: str) -> None:
        self.code_sha = code_sha

    def run(self, sample: EvalSample) -> EvalResult:
        return sample.baseline_result.model_copy(update={"code_sha": self.code_sha})


class MetadataOnlyCandidateAdapter:
    def __init__(self, code_sha: str) -> None:
        self.code_sha = code_sha

    def run(self, sample: EvalSample) -> EvalResult:
        return sample.candidate_result.model_copy(update={"code_sha": self.code_sha})


class DeterministicMockCandidateAdapter:
    def __init__(self, code_sha: str) -> None:
        self.code_sha = code_sha

    def run(self, sample: EvalSample) -> EvalResult:
        latency = sample.candidate_result.latency_ms
        if latency is None:
            latency = sample.baseline_result.latency_ms or 1000
        return sample.candidate_result.model_copy(
            update={
                "code_sha": self.code_sha,
                "model": sample.candidate_result.model or "deterministic-eval-candidate",
                "provider": sample.candidate_result.provider or "offline",
                "latency_ms": latency,
                "prompt_tokens": sample.candidate_result.prompt_tokens or 0,
                "completion_tokens": sample.candidate_result.completion_tokens or 0,
                "estimated_cost_cents": sample.candidate_result.estimated_cost_cents or 0.0,
            }
        )

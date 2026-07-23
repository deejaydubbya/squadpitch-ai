from __future__ import annotations

from typing import Protocol

from squadpitch_ai.evals.models import EvalSample


class JudgeUnavailableError(Exception):
    pass


class ModelAssistedJudge(Protocol):
    def score(self, sample: EvalSample) -> int: ...


class DisabledJudge:
    def score(self, sample: EvalSample) -> int:
        raise JudgeUnavailableError("model-assisted judge disabled")


class MockJudge:
    def score(self, sample: EvalSample) -> int:
        return sample.human_review.quality_score or 3

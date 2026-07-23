from __future__ import annotations

import math
import re

from squadpitch_ai.brand_quality.models import (
    BRAND_QUALITY_SCHEMA_VERSION,
    MODEL_VERSION,
    QUALITY_LABELS,
    BrandQualityDimensionScore,
    BrandQualityScoreRequest,
    BrandQualityScoreResponse,
    RiskLabel,
)

PROMO_WORDS = {"limited", "exclusive", "deal", "offer", "act now", "guaranteed"}
SUPPORTED_LANGUAGES = {"en", "es"}
CHANNEL_LIMITS = {
    "X": 280,
    "THREADS": 500,
    "INSTAGRAM": 2200,
    "FACEBOOK": 3000,
    "LINKEDIN": 3000,
}


def score_brand_quality(request: BrandQualityScoreRequest) -> BrandQualityScoreResponse:
    if request.model_version != MODEL_VERSION:
        raise ValueError(f"unsupported model version: {request.model_version}")
    text = request.sanitized_text.strip()
    lower = text.lower()
    word_count = len(re.findall(r"\w+", text))
    channel_limit = CHANNEL_LIMITS.get(request.channel.upper(), 2200)

    banned_hits = [phrase for phrase in request.banned_phrases if phrase.lower() in lower]
    brand_terms = [term.lower() for term in request.brand_constraints if term.strip()]
    brand_match = (
        0.5 if not brand_terms else sum(term in lower for term in brand_terms) / len(brand_terms)
    )
    promo_score = min(1.0, sum(word in lower for word in PROMO_WORDS) / 3)
    verbosity_score = min(1.0, max(0, word_count - 80) / 120)
    channel_risk = 1.0 if len(text) > channel_limit else 0.0
    language_risk = 0.0 if request.language in SUPPORTED_LANGUAGES else 1.0
    review_score = max(
        1.0 - brand_match,
        promo_score,
        verbosity_score,
        1.0 if banned_hits else 0.0,
        channel_risk,
        language_risk,
    )
    dimensions = [
        dimension(
            "brand_voice_match", 1.0 - brand_match, "Lower brand-term coverage.", invert=True
        ),
        dimension("excessive_promotion", promo_score, "Promotional language density."),
        dimension("excessive_verbosity", verbosity_score, "Length exceeds concise content target."),
        dimension(
            "prohibited_phrase", 1.0 if banned_hits else 0.0, f"Banned phrases: {banned_hits}"
        ),
        dimension("channel_suitability", channel_risk, f"Channel limit {channel_limit} chars."),
        dimension("unsupported_language_risk", language_risk, f"Language={request.language}."),
        dimension("needs_human_review", review_score, "Maximum policy/quality risk."),
    ]
    categories = [score.label for score in dimensions if score.risk != "low"]
    return BrandQualityScoreResponse(
        schemaVersion=BRAND_QUALITY_SCHEMA_VERSION,
        workspaceId=request.workspace_id,
        contentId=request.content_id,
        modelVersion=MODEL_VERSION,
        modelFamily="deterministic_shadow",
        scores=dimensions,
        needsHumanReview=review_score >= 0.5,
        categories=categories,
        calibration={
            "method": 0.0,
            "expectedCalibrationError": 0.0,
            "trainedArtifactAvailable": 0.0,
        },
        explanations=[score.explanation for score in dimensions if score.risk != "low"],
        traceId=request.trace_id,
        proposalOnly=True,
    )


def dimension(
    label: str, risk_score: float, explanation: str, *, invert: bool = False
) -> BrandQualityDimensionScore:
    if label not in QUALITY_LABELS:
        raise ValueError(f"unknown label: {label}")
    clamped = max(0.0, min(1.0, risk_score))
    score = 1.0 - clamped if invert else clamped
    return BrandQualityDimensionScore(
        label=label,
        score=round(score, 6),
        risk=risk_label(clamped),
        explanation=explanation,
    )


def risk_label(value: float) -> RiskLabel:
    if value >= 0.66:
        return "high"
    if value >= 0.33:
        return "medium"
    return "low"


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))

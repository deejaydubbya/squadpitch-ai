from __future__ import annotations

from squadpitch_ai.autopilot_ranker.models import AutopilotRankingCandidate

FEATURE_SCHEMA_VERSION = "autopilot-ranking-features.v1"
FEATURE_NAMES = (
    "bias",
    "heuristic_score",
    "trigger_new_listing",
    "trigger_price_drop",
    "trigger_open_house",
    "trigger_just_sold",
    "trigger_stale_listing",
    "listing_age_days",
    "price_change_percent",
    "days_since_last_post",
    "media_available",
    "historical_approval_rate",
    "historical_engagement_rate",
    "hour_of_day",
    "day_of_week",
    "recent_audience_engagement",
)
LEAKAGE_BANNED_FEATURES = {
    "approved",
    "dismissed",
    "published",
    "exceededBaselineEngagement",
    "producedInquiry",
    "conversion",
    "label",
    "decidedAt",
    "publishedAt",
    "generatedDraftIds",
}


def vectorize_candidate(candidate: AutopilotRankingCandidate) -> tuple[list[float], list[str]]:
    missing: list[str] = []

    def value(name: str, raw: float | int | None, default: float) -> float:
        if raw is None:
            missing.append(name)
            return default
        return float(raw)

    trigger = candidate.trigger_type.upper()
    return [
        1.0,
        candidate.heuristic_score,
        1.0 if trigger == "NEW_LISTING" else 0.0,
        1.0 if trigger == "PRICE_DROP" else 0.0,
        1.0 if trigger == "OPEN_HOUSE" else 0.0,
        1.0 if trigger == "JUST_SOLD" else 0.0,
        1.0 if trigger == "STALE_LISTING" else 0.0,
        min(value("listingAgeDays", candidate.listing_age_days, 30.0), 365.0) / 365.0,
        value("priceChangePercent", candidate.price_change_percent, 0.0) / 100.0,
        min(value("daysSinceLastPost", candidate.days_since_last_post, 14.0), 90.0) / 90.0,
        1.0 if candidate.media_available else 0.0,
        value("historicalApprovalRate", candidate.historical_approval_rate, 0.5),
        value("historicalEngagementRate", candidate.historical_engagement_rate, 0.0),
        value("hourOfDay", candidate.hour_of_day, 12.0) / 23.0,
        value("dayOfWeek", candidate.day_of_week, 3.0) / 6.0,
        value("recentAudienceEngagement", candidate.recent_audience_engagement, 0.0),
    ], missing


def assert_no_target_leakage(feature_names: list[str] | tuple[str, ...]) -> None:
    leaked = sorted(set(feature_names) & LEAKAGE_BANNED_FEATURES)
    if leaked:
        raise ValueError(f"target leakage features are not allowed: {', '.join(leaked)}")

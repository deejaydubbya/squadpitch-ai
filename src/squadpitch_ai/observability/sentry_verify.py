import os

import sentry_sdk


def main() -> None:
    dsn = os.getenv("SP_AI_SENTRY_DSN")
    if not dsn:
        raise SystemExit(
            "Sentry is not configured. Set SP_AI_SENTRY_DSN for this operator command."
        )
    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv("SP_AI_SENTRY_ENVIRONMENT", "production"),
        release=(
            os.getenv("SP_AI_SENTRY_RELEASE")
            or os.getenv("SP_AI_BUILD_SHA")
            or os.getenv("FLY_IMAGE_REF")
        ),
        traces_sample_rate=0.0,
        send_default_pii=False,
    )
    event_id = sentry_sdk.capture_exception(
        RuntimeError("Squadpitch AI production-readiness verification"),
        tags={"synthetic": "true", "source": "production-readiness", "service": "squadpitch-ai"},
    )
    delivered = sentry_sdk.flush(timeout=5.0)
    status = "submitted" if delivered else "timed out"
    print(f"Synthetic Sentry event {status}; event ID: {event_id}")
    raise SystemExit(0 if delivered else 1)

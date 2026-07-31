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
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("synthetic", "true")
        scope.set_tag("source", "production-readiness")
        scope.set_tag("service", "squadpitch-ai")
        event_id = sentry_sdk.capture_exception(
            RuntimeError("Squadpitch AI production-readiness verification")
        )
    sentry_sdk.flush(timeout=5.0)
    print(f"Synthetic Sentry event submitted; event ID: {event_id}")

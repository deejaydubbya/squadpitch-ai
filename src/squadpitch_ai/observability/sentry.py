from __future__ import annotations

import os
from typing import Any, cast

import sentry_sdk
from sentry_sdk.types import Event, Hint

from squadpitch_ai.core.config import Settings

_SENSITIVE_KEY_PARTS = (
    "authorization",
    "cookie",
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "phone",
    "email",
    "message",
    "body",
    "content",
)


def _redact(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return "[Truncated]"
    if isinstance(value, list):
        return [_redact(item, depth + 1) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: "[Filtered]"
        if any(part in str(key).lower() for part in _SENSITIVE_KEY_PARTS)
        else _redact(item, depth + 1)
        for key, item in value.items()
    }


def before_send(event: Event, _hint: Hint) -> Event | None:
    sanitized = cast(Event, _redact(event))
    request = sanitized.get("request")
    if isinstance(request, dict):
        request_url = request.get("url")
        if isinstance(request_url, str):
            request["url"] = request_url.split("?", maxsplit=1)[0]
        request.pop("headers", None)
        request.pop("cookies", None)
        request.pop("data", None)
    user = sanitized.get("user")
    if isinstance(user, dict):
        user.pop("email", None)
        user.pop("ip_address", None)
        user.pop("username", None)
    return sanitized


def init_sentry(settings: Settings, *, service: str = "squadpitch-ai") -> bool:
    if not settings.sentry_dsn:
        return False
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment or settings.app_env,
        release=(
            settings.sentry_release or os.getenv("SP_AI_BUILD_SHA") or os.getenv("FLY_IMAGE_REF")
        ),
        traces_sample_rate=settings.sentry_traces_sample_rate,
        send_default_pii=False,
        before_send=before_send,
    )
    sentry_sdk.set_tag("service", service)
    return True


def capture_exception(error: BaseException, **safe_context: str | None) -> str | None:
    with sentry_sdk.push_scope() as scope:
        for key, value in safe_context.items():
            if value is not None:
                scope.set_tag(key, value)
        event_id = sentry_sdk.capture_exception(error)
    return str(event_id) if event_id else None

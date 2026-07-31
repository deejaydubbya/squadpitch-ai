from typing import Any, cast

from sentry_sdk.types import Event

from squadpitch_ai.core.config import Settings
from squadpitch_ai.observability.sentry import before_send, init_sentry


def test_sentry_disabled_without_dsn() -> None:
    assert init_sentry(Settings(app_env="test")) is False


def test_before_send_redacts_credentials_content_and_pii() -> None:
    event: dict[str, Any] = {
        "request": {
            "headers": {"authorization": "Bearer secret"},
            "cookies": {"session": "secret"},
            "data": "customer content",
            "url": "/route?code=oauth-secret",
        },
        "user": {"id": "user-id", "email": "person@example.com", "ip_address": "127.0.0.1"},
        "extra": {"workspace_id": "workspace-id", "provider": "openai", "access_token": "secret"},
    }
    sanitized = before_send(cast(Event, event), {})
    assert sanitized is not None
    assert sanitized["request"] == {"url": "/route"}
    assert sanitized["user"] == {"id": "user-id"}
    assert sanitized["extra"]["access_token"] == "[Filtered]"
    serialized = str(sanitized)
    assert "Bearer secret" not in serialized
    assert "person@example.com" not in serialized
    assert "oauth-secret" not in serialized

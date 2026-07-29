import pytest
from pydantic import ValidationError

from squadpitch_ai.core.config import Settings


def test_non_production_settings_keep_safe_local_defaults() -> None:
    settings = Settings(app_env="test")
    assert settings.postgres_dsn is None
    assert settings.redis_url is None


def test_production_requires_core_dependencies_and_service_auth() -> None:
    with pytest.raises(ValidationError) as error:
        Settings(app_env="production")

    message = str(error.value)
    assert "SP_AI_POSTGRES_DSN" in message
    assert "SP_AI_REDIS_URL" in message
    assert "SP_AI_SERVICE_AUTH_SECRETS" in message


def test_production_accepts_complete_core_configuration() -> None:
    settings = Settings(
        app_env="production",
        postgres_dsn="postgresql://db.internal/squadpitch",
        redis_url="redis://redis.internal:6379",
        service_auth_secrets="primary:not-a-real-secret",
    )
    assert settings.service_auth_secrets_by_key_id == {"primary": "not-a-real-secret"}

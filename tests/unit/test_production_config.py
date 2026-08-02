from pathlib import Path

import pytest
from pydantic import ValidationError

from squadpitch_ai.core.config import Settings


def test_non_production_settings_keep_safe_local_defaults() -> None:
    settings = Settings(app_env="test")
    assert settings.postgres_dsn is None
    assert settings.redis_url is None


def test_production_requires_service_auth() -> None:
    with pytest.raises(ValidationError) as error:
        Settings(app_env="production")

    message = str(error.value)
    assert "SP_AI_SERVICE_AUTH_SECRETS" in message


def test_production_accepts_service_auth_without_unused_dependencies() -> None:
    settings = Settings(
        app_env="production",
        service_auth_secrets="primary:not-a-real-secret",
    )
    assert settings.postgres_dsn is None
    assert settings.redis_url is None
    assert settings.service_auth_secrets_by_key_id == {"primary": "not-a-real-secret"}


def test_explicit_malformed_bom_prefixed_env_is_still_rejected(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("\ufeffSP_AI_SENTRY_TRACES_SAMPLE_RATE=0.5\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Settings(_env_file=env_file)  # type: ignore[call-arg]  # Pydantic runtime kwarg


def test_explicit_unknown_runtime_setting_is_rejected(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SP_AI_NOT_A_REAL_SETTING=value\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Settings(_env_file=env_file)  # type: ignore[call-arg]  # Pydantic runtime kwarg

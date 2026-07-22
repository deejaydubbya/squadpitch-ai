import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SP_AI_",
        env_file=".env",
        extra="forbid",
        case_sensitive=False,
    )

    app_env: Literal["development", "test", "staging", "production"] = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8080")), ge=1, le=65535)
    postgres_dsn: str | None = None
    redis_url: str | None = None
    otel_service_name: str = "squadpitch-ai"
    request_timeout_seconds: PositiveInt = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()

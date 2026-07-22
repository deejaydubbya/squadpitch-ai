from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException

from squadpitch_ai.api.middleware import RequestContextMiddleware
from squadpitch_ai.contracts.errors import ErrorCode, ErrorDetail, ErrorEnvelope
from squadpitch_ai.core.config import Settings, get_settings
from squadpitch_ai.core.dependencies import DependencyRegistry, build_dependency_registry
from squadpitch_ai.observability.logging import configure_logging

logger = structlog.get_logger(__name__)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    service: str


class ReadinessDependencyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    ready: bool


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    dependencies: list[ReadinessDependencyResponse]


def error_response(
    request: Request,
    status_code: int,
    code: ErrorCode,
    message: str,
) -> JSONResponse:
    envelope = ErrorEnvelope(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=getattr(request.state, "request_id", None),
            trace_id=getattr(request.state, "trace_id", None),
        ),
    )
    return JSONResponse(
        status_code=status_code,
        content=envelope.model_dump(by_alias=True),
    )


def create_app(
    settings: Settings | None = None,
    registry: DependencyRegistry | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    dependency_registry = registry or build_dependency_registry()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(resolved_settings.log_level)
        app.state.settings = resolved_settings
        app.state.dependency_registry = dependency_registry
        logger.info(
            "api_startup",
            requestId=None,
            traceId=None,
            taskName=None,
            taskVersion=None,
            schemaVersion=None,
            errorCode=None,
            latencyMs=None,
        )
        yield
        logger.info(
            "api_shutdown",
            requestId=None,
            traceId=None,
            taskName=None,
            taskVersion=None,
            schemaVersion=None,
            errorCode=None,
            latencyMs=None,
        )

    app = FastAPI(
        title="SquadPitch AI",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(RequestContextMiddleware)
    FastAPIInstrumentor.instrument_app(app)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return error_response(
            request,
            exc.status_code,
            ErrorCode.INTERNAL_ERROR,
            str(exc.detail),
        )

    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        return error_response(
            request,
            exc.status_code,
            ErrorCode.INTERNAL_ERROR,
            str(exc.detail),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return error_response(request, 422, ErrorCode.SCHEMA_INVALID, "Request validation failed")

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unhandled_exception",
            requestId=getattr(request.state, "request_id", None),
            traceId=getattr(request.state, "trace_id", None),
            taskName=None,
            taskVersion=None,
            schemaVersion=None,
            errorCode=ErrorCode.INTERNAL_ERROR.value,
            latencyMs=None,
        )
        return error_response(request, 500, ErrorCode.INTERNAL_ERROR, "Internal server error")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", service="squadpitch-ai")

    @app.get("/ready", response_model=ReadinessResponse)
    async def ready() -> ReadinessResponse:
        statuses = await dependency_registry.readiness()
        return ReadinessResponse(
            status="ready" if all(status.ready for status in statuses) else "not_ready",
            dependencies=[
                ReadinessDependencyResponse(name=status.name, ready=status.ready)
                for status in statuses
            ],
        )

    @app.get("/__test/error", include_in_schema=False)
    async def test_error() -> None:
        if resolved_settings.app_env != "test":
            raise HTTPException(status_code=404, detail="Not found")
        raise HTTPException(status_code=418, detail="Test error")

    return app


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "squadpitch_ai.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    main()

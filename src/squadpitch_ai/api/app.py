import socket
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from squadpitch_ai.api.middleware import RequestContextMiddleware
from squadpitch_ai.autopilot_ranker import AutopilotRankingRequest, rank_autopilot_opportunities
from squadpitch_ai.brand_quality import BrandQualityScoreRequest
from squadpitch_ai.campaign_ops import (
    CampaignOpsPlanRequest,
    DraftContentProposalRequest,
    build_campaign_ops_plan,
    build_draft_content_proposal,
)
from squadpitch_ai.contracts.errors import ErrorCode, ErrorEnvelope
from squadpitch_ai.contracts.service_envelope import (
    AiServiceScope,
    BoundedNonceStore,
    ServiceAuthError,
    ServiceEnvelope,
    verify_service_envelope,
)
from squadpitch_ai.core.config import Settings, get_settings
from squadpitch_ai.core.dependencies import DependencyRegistry, build_dependency_registry
from squadpitch_ai.experimentation import ExperimentAnalysisRequest, analyze_experiment
from squadpitch_ai.model_registry import ModelRegistryError, get_default_brand_quality_inference
from squadpitch_ai.observability.execution_provenance import execution_provenance
from squadpitch_ai.observability.logging import configure_logging
from squadpitch_ai.observability.sentry import capture_exception, init_sentry
from squadpitch_ai.retrieval.api import RetrievalQueryRequest, execute_retrieval_query

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
    retryable: bool = False,
    field_errors: list[dict[str, str]] | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    trace_id = getattr(request.state, "trace_id", None)
    schema_version = getattr(request.state, "schema_version", None)
    envelope = ErrorEnvelope(
        code=code,
        message=message,
        retryable=retryable,
        request_id=request_id,
        trace_id=trace_id,
        schema_version=schema_version,
        field_errors=field_errors,
    )
    return JSONResponse(
        status_code=status_code,
        content=envelope.model_dump(by_alias=True, exclude_none=True),
    )


def status_for_error_code(code: ErrorCode) -> int:
    if code in {
        ErrorCode.AUTH_SIGNATURE_MISSING,
        ErrorCode.AUTH_SIGNATURE_INVALID,
        ErrorCode.AUTH_REQUEST_EXPIRED,
        ErrorCode.AUTH_REQUEST_FUTURE_DATED,
        ErrorCode.AUTH_NONCE_REPLAYED,
        ErrorCode.AUTH_SCOPE_DENIED,
    }:
        return 401 if code != ErrorCode.AUTH_SCOPE_DENIED else 403
    if code in {
        ErrorCode.CONTRACT_UNSUPPORTED_SCHEMA_VERSION,
        ErrorCode.CONTRACT_WORKSPACE_MISMATCH,
        ErrorCode.SCHEMA_INVALID,
    }:
        return 422
    if code in {
        ErrorCode.PROVIDER_UNAVAILABLE,
        ErrorCode.PROVIDER_TIMEOUT,
    }:
        return 503
    return 500


def parse_service_envelope(body: dict[str, Any], request: Request) -> ServiceEnvelope:
    if "signature" not in body:
        request.state.request_id = body.get("requestId")
        request.state.trace_id = body.get("traceId")
        request.state.schema_version = body.get("schemaVersion")
        raise ServiceAuthError(ErrorCode.AUTH_SIGNATURE_MISSING, "Signature metadata is required")
    try:
        envelope = ServiceEnvelope.model_validate(body)
    except ValidationError as exc:
        request.state.request_id = body.get("requestId")
        request.state.trace_id = body.get("traceId")
        request.state.schema_version = body.get("schemaVersion")
        unsupported_schema = any(
            "Unsupported schema version" in str(error.get("msg", "")) for error in exc.errors()
        )
        if unsupported_schema:
            raise ServiceAuthError(
                ErrorCode.CONTRACT_UNSUPPORTED_SCHEMA_VERSION,
                "Unsupported schema version",
            ) from exc
        raise ServiceAuthError(ErrorCode.SCHEMA_INVALID, "Request validation failed") from exc
    request.state.request_id = envelope.request_id
    request.state.trace_id = envelope.trace_id
    request.state.schema_version = envelope.schema_version
    return envelope


def create_app(
    settings: Settings | None = None,
    registry: DependencyRegistry | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    dependency_registry = registry or build_dependency_registry(resolved_settings)
    init_sentry(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(resolved_settings.log_level)
        app.state.settings = resolved_settings
        app.state.dependency_registry = dependency_registry
        app.state.nonce_store = BoundedNonceStore(
            max_entries=resolved_settings.service_auth_nonce_store_max_entries,
        )
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
        field_errors = [
            {"field": ".".join(str(part) for part in err["loc"]), "message": err["msg"]}
            for err in exc.errors()
        ]
        return error_response(
            request,
            422,
            ErrorCode.SCHEMA_INVALID,
            "Request validation failed",
            field_errors=field_errors,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        capture_exception(
            exc,
            request_id=getattr(request.state, "request_id", None),
            trace_id=getattr(request.state, "trace_id", None),
            route=request.url.path,
        )
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

    @app.post("/v1/health/check", response_model=HealthResponse)
    async def signed_health(
        body: dict[str, Any], request: Request
    ) -> HealthResponse | JSONResponse:
        try:
            envelope = parse_service_envelope(body, request)
            verify_service_envelope(
                envelope,
                secrets_by_key_id=resolved_settings.service_auth_secrets_by_key_id,
                nonce_store=request.app.state.nonce_store,
                required_scope=AiServiceScope.HEALTH_READ,
            )
        except ServiceAuthError as exc:
            return error_response(
                request,
                status_for_error_code(exc.code),
                exc.code,
                str(exc),
                retryable=exc.retryable,
            )
        return HealthResponse(status="ok", service="squadpitch-ai")

    @app.post("/v1/eval/run", include_in_schema=False, response_model=None)
    async def signed_eval_probe(
        body: dict[str, Any], request: Request
    ) -> dict[str, object] | JSONResponse:
        try:
            envelope = parse_service_envelope(body, request)
            verify_service_envelope(
                envelope,
                secrets_by_key_id=resolved_settings.service_auth_secrets_by_key_id,
                nonce_store=request.app.state.nonce_store,
                required_scope=AiServiceScope.EVAL_RUN,
            )
        except ServiceAuthError as exc:
            return error_response(
                request,
                status_for_error_code(exc.code),
                exc.code,
                str(exc),
                retryable=exc.retryable,
            )
        return {
            "status": "accepted",
            "enabled": False,
            "requestId": envelope.request_id,
            "traceId": envelope.trace_id,
            "schemaVersion": envelope.schema_version,
        }

    @app.post("/v1/experiments/analyze", include_in_schema=False, response_model=None)
    async def signed_experiment_analysis(
        body: dict[str, Any], request: Request
    ) -> dict[str, object] | JSONResponse:
        started_at = time.perf_counter()
        try:
            envelope = parse_service_envelope(body, request)
            verify_service_envelope(
                envelope,
                secrets_by_key_id=resolved_settings.service_auth_secrets_by_key_id,
                nonce_store=request.app.state.nonce_store,
                required_scope=AiServiceScope.EVAL_RUN,
            )
            analysis_request = ExperimentAnalysisRequest.model_validate(
                {
                    "schemaVersion": envelope.payload.get("schemaVersion"),
                    "definition": envelope.payload.get("definition"),
                    "exposures": envelope.payload.get("exposures", []),
                    "outcomes": envelope.payload.get("outcomes", []),
                    "traceId": envelope.trace_id,
                }
            )
            report = analyze_experiment(analysis_request)
        except ServiceAuthError as exc:
            return error_response(
                request,
                status_for_error_code(exc.code),
                exc.code,
                str(exc),
                retryable=exc.retryable,
            )
        except (ValidationError, ValueError) as exc:
            return error_response(
                request,
                422,
                ErrorCode.SCHEMA_INVALID,
                str(exc),
                retryable=False,
            )
        response = cast(dict[str, object], report.model_dump(mode="json", by_alias=True))
        response["provenance"] = execution_provenance(
            operation="experiment_analysis",
            implementation="deterministic_experiment_analysis_v1",
            trace_id=envelope.trace_id,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            inference_mode="deterministic",
        )
        return response

    @app.post("/v1/retrieval/query", include_in_schema=False, response_model=None)
    async def signed_retrieval_query(
        body: dict[str, Any], request: Request
    ) -> dict[str, object] | JSONResponse:
        started_at = time.perf_counter()
        try:
            envelope = parse_service_envelope(body, request)
            verify_service_envelope(
                envelope,
                secrets_by_key_id=resolved_settings.service_auth_secrets_by_key_id,
                nonce_store=request.app.state.nonce_store,
                required_scope=AiServiceScope.RETRIEVAL_QUERY,
            )
            retrieval_request = RetrievalQueryRequest.model_validate(
                {
                    **envelope.payload,
                    "workspaceId": envelope.workspace_id,
                    "traceId": envelope.trace_id,
                }
            )
            result = execute_retrieval_query(retrieval_request)
        except ServiceAuthError as exc:
            return error_response(
                request,
                status_for_error_code(exc.code),
                exc.code,
                str(exc),
                retryable=exc.retryable,
            )
        except (ValidationError, ValueError) as exc:
            return error_response(
                request,
                422,
                ErrorCode.SCHEMA_INVALID,
                str(exc),
                retryable=False,
            )
        response = cast(dict[str, object], result.model_dump(mode="json", by_alias=True))
        response["provenance"] = execution_provenance(
            operation="retrieval_query",
            implementation="hybrid_retrieval_v1",
            trace_id=envelope.trace_id,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            inference_mode="deterministic_embedding_hybrid",
        )
        logger.info(
            "retrieval_query_completed",
            traceId=envelope.trace_id,
            workspaceId=envelope.workspace_id,
            resultCount=result.result_count,
            topK=result.top_k,
            empty=result.empty,
            schemaVersion=result.schema_version,
        )
        return response

    @app.post("/v1/campaign-ops/plan", include_in_schema=False, response_model=None)
    async def signed_campaign_ops_plan(
        body: dict[str, Any], request: Request
    ) -> dict[str, object] | JSONResponse:
        started_at = time.perf_counter()
        try:
            envelope = parse_service_envelope(body, request)
            verify_service_envelope(
                envelope,
                secrets_by_key_id=resolved_settings.service_auth_secrets_by_key_id,
                nonce_store=request.app.state.nonce_store,
                required_scope=AiServiceScope.CAMPAIGN_PLAN_READ,
            )
            plan_request = CampaignOpsPlanRequest.model_validate(
                {
                    "workspaceId": envelope.workspace_id,
                    "objective": envelope.payload.get("objective"),
                    "snapshot": envelope.payload.get("snapshot"),
                    "traceId": envelope.trace_id,
                }
            )
            plan = build_campaign_ops_plan(plan_request)
        except ServiceAuthError as exc:
            return error_response(
                request,
                status_for_error_code(exc.code),
                exc.code,
                str(exc),
                retryable=exc.retryable,
            )
        except (ValidationError, ValueError) as exc:
            return error_response(
                request,
                422,
                ErrorCode.SCHEMA_INVALID,
                str(exc),
                retryable=False,
            )
        response = plan.model_dump(mode="json", by_alias=True)
        response["provenance"] = execution_provenance(
            operation="campaign_ops_plan",
            implementation="campaign_ops_v1",
            trace_id=envelope.trace_id,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            inference_mode="deterministic",
        )
        return response

    @app.post("/v1/campaign-ops/draft-proposal", include_in_schema=False, response_model=None)
    async def signed_draft_content_proposal(
        body: dict[str, Any], request: Request
    ) -> dict[str, object] | JSONResponse:
        started_at = time.perf_counter()
        try:
            envelope = parse_service_envelope(body, request)
            verify_service_envelope(
                envelope,
                secrets_by_key_id=resolved_settings.service_auth_secrets_by_key_id,
                nonce_store=request.app.state.nonce_store,
                required_scope=AiServiceScope.CAMPAIGN_PLAN_READ,
            )
            proposal_request = DraftContentProposalRequest.model_validate(
                {
                    "workspaceId": envelope.workspace_id,
                    "objective": envelope.payload.get("objective"),
                    "snapshot": envelope.payload.get("snapshot"),
                    "requestedChannels": envelope.payload.get("requestedChannels", []),
                    "idempotencyKey": envelope.payload.get("idempotencyKey"),
                    "traceId": envelope.trace_id,
                }
            )
            proposal = build_draft_content_proposal(proposal_request)
        except ServiceAuthError as exc:
            return error_response(
                request,
                status_for_error_code(exc.code),
                exc.code,
                str(exc),
                retryable=exc.retryable,
            )
        except (ValidationError, ValueError) as exc:
            return error_response(
                request,
                422,
                ErrorCode.SCHEMA_INVALID,
                str(exc),
                retryable=False,
            )
        response = proposal.model_dump(mode="json", by_alias=True)
        response["provenance"] = execution_provenance(
            operation="draft_content_proposal",
            implementation="draft_content_proposal_v1",
            trace_id=envelope.trace_id,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            inference_mode="deterministic",
        )
        return response

    @app.post("/v1/autopilot/rank", include_in_schema=False, response_model=None)
    async def signed_autopilot_rank(
        body: dict[str, Any], request: Request
    ) -> dict[str, object] | JSONResponse:
        started_at = time.perf_counter()
        try:
            envelope = parse_service_envelope(body, request)
            verify_service_envelope(
                envelope,
                secrets_by_key_id=resolved_settings.service_auth_secrets_by_key_id,
                nonce_store=request.app.state.nonce_store,
                required_scope=AiServiceScope.AUTOPILOT_RANK_READ,
            )
            ranking_request = AutopilotRankingRequest.model_validate(
                {
                    "schemaVersion": envelope.payload.get("schemaVersion"),
                    "workspaceId": envelope.workspace_id,
                    "candidates": envelope.payload.get("candidates"),
                    "modelVersion": envelope.payload.get("modelVersion"),
                    "shadowMode": envelope.payload.get("shadowMode", True),
                    "traceId": envelope.trace_id,
                }
            )
            result = rank_autopilot_opportunities(ranking_request)
        except ServiceAuthError as exc:
            return error_response(
                request,
                status_for_error_code(exc.code),
                exc.code,
                str(exc),
                retryable=exc.retryable,
            )
        except (ValidationError, ValueError) as exc:
            return error_response(
                request,
                422,
                ErrorCode.SCHEMA_INVALID,
                str(exc),
                retryable=False,
            )
        response = cast(dict[str, object], result.model_dump(mode="json", by_alias=True))
        response["provenance"] = execution_provenance(
            operation="autopilot_rank",
            implementation="autopilot_logistic_ranker_v1",
            trace_id=envelope.trace_id,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            inference_mode="logistic_regression",
            model="autopilot-ranker",
            model_version=str(result.model_metadata.get("modelVersion")),
        )
        return response

    @app.post("/v1/content-quality/score", include_in_schema=False, response_model=None)
    async def signed_content_quality_score(
        body: dict[str, Any], request: Request
    ) -> dict[str, object] | JSONResponse:
        started_at = time.perf_counter()
        try:
            envelope = parse_service_envelope(body, request)
            verify_service_envelope(
                envelope,
                secrets_by_key_id=resolved_settings.service_auth_secrets_by_key_id,
                nonce_store=request.app.state.nonce_store,
                required_scope=AiServiceScope.CONTENT_SCORE_READ,
            )
            score_request = BrandQualityScoreRequest.model_validate(
                {
                    "schemaVersion": envelope.payload.get("schemaVersion"),
                    "workspaceId": envelope.workspace_id,
                    "contentId": envelope.payload.get("contentId"),
                    "sanitizedText": envelope.payload.get("sanitizedText"),
                    "channel": envelope.payload.get("channel"),
                    "industry": envelope.payload.get("industry", "real_estate"),
                    "brandConstraints": envelope.payload.get("brandConstraints", []),
                    "bannedPhrases": envelope.payload.get("bannedPhrases", []),
                    "language": envelope.payload.get("language", "en"),
                    "modelVersion": envelope.payload.get("modelVersion"),
                    "traceId": envelope.trace_id,
                }
            )
            result, metric = get_default_brand_quality_inference().predict(score_request)
        except ServiceAuthError as exc:
            return error_response(
                request,
                status_for_error_code(exc.code),
                exc.code,
                str(exc),
                retryable=exc.retryable,
            )
        except ModelRegistryError as exc:
            return error_response(
                request,
                503 if exc.code in {"ARTIFACT_MISSING", "ARTIFACT_CHECKSUM_MISMATCH"} else 422,
                ErrorCode.PROVIDER_UNAVAILABLE
                if exc.code in {"ARTIFACT_MISSING", "ARTIFACT_CHECKSUM_MISMATCH"}
                else ErrorCode.SCHEMA_INVALID,
                str(exc),
                retryable=exc.code in {"ARTIFACT_MISSING", "ARTIFACT_CHECKSUM_MISMATCH"},
            )
        except (ValidationError, ValueError) as exc:
            return error_response(
                request,
                422,
                ErrorCode.SCHEMA_INVALID,
                str(exc),
                retryable=False,
            )
        logger.info(
            "model_inference",
            requestId=getattr(request.state, "request_id", None),
            traceId=getattr(request.state, "trace_id", None),
            taskName="brand_content_quality",
            taskVersion=metric.version,
            schemaVersion=result.schema_version,
            errorCode=None,
            latencyMs=round(metric.latency_ms),
            modelId=metric.model_id,
            coldStart=metric.cold_start,
            batchSize=metric.batch_size,
        )
        response = cast(dict[str, object], result.model_dump(mode="json", by_alias=True))
        response["provenance"] = execution_provenance(
            operation="brand_quality_score",
            implementation="deterministic_brand_quality_v1",
            trace_id=envelope.trace_id,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            inference_mode="deterministic",
        )
        return response

    @app.post("/v1/models/registry/health", include_in_schema=False, response_model=None)
    async def signed_model_registry_health(
        body: dict[str, Any], request: Request
    ) -> dict[str, object] | JSONResponse:
        try:
            envelope = parse_service_envelope(body, request)
            verify_service_envelope(
                envelope,
                secrets_by_key_id=resolved_settings.service_auth_secrets_by_key_id,
                nonce_store=request.app.state.nonce_store,
                required_scope=AiServiceScope.HEALTH_READ,
            )
            health = get_default_brand_quality_inference().health()
        except ServiceAuthError as exc:
            return error_response(
                request,
                status_for_error_code(exc.code),
                exc.code,
                str(exc),
                retryable=exc.retryable,
            )
        except ModelRegistryError as exc:
            return error_response(
                request,
                503,
                ErrorCode.PROVIDER_UNAVAILABLE,
                str(exc),
                retryable=True,
            )
        return health

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
    if settings.api_host == "::":
        server_socket = create_dual_stack_socket(settings.api_port)
        config = uvicorn.Config(
            "squadpitch_ai.api.app:create_app",
            factory=True,
            host=settings.api_host,
            port=settings.api_port,
        )
        uvicorn.Server(config).run(sockets=[server_socket])
        return
    uvicorn.run(
        "squadpitch_ai.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
    )


def create_dual_stack_socket(port: int) -> socket.socket:
    """Bind one explicit IPv6 socket that also accepts IPv4-mapped traffic."""
    server_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        server_socket.bind(("::", port))
        server_socket.set_inheritable(True)
        return server_socket
    except BaseException:
        server_socket.close()
        raise


if __name__ == "__main__":
    main()

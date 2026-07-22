import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

REQUEST_ID_HEADER = "x-request-id"
TRACE_ID_HEADER = "x-trace-id"

logger = structlog.get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started_at = time.perf_counter()
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        trace_id = request.headers.get(TRACE_ID_HEADER) or request_id
        request.state.request_id = request_id
        request.state.trace_id = trace_id

        response = await call_next(request)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[TRACE_ID_HEADER] = trace_id

        logger.info(
            "request_complete",
            requestId=request_id,
            traceId=trace_id,
            taskName=None,
            taskVersion=None,
            schemaVersion=None,
            errorCode=None,
            latencyMs=latency_ms,
            method=request.method,
            path=request.url.path,
            statusCode=response.status_code,
        )
        return response

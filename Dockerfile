FROM ghcr.io/astral-sh/uv:0.5.4-python3.12-bookworm-slim AS runtime

ARG SP_AI_BUILD_SHA

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    SP_AI_BUILD_SHA=${SP_AI_BUILD_SHA}

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY apps ./apps

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["python", "-m", "squadpitch_ai.api.app"]

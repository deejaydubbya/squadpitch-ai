# SquadPitch AI

Python AI platform foundation for future SquadPitch intelligence-plane work.

[![CI](https://github.com/deejaydubbya/squadpitch-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/deejaydubbya/squadpitch-ai/actions/workflows/ci.yml)
[![Deploy](https://github.com/deejaydubbya/squadpitch-ai/actions/workflows/deploy.yml/badge.svg)](https://github.com/deejaydubbya/squadpitch-ai/actions/workflows/deploy.yml)

This project currently provides:

- FastAPI API process with `/health` and `/ready`.
- Background worker entry point with graceful shutdown.
- Structured JSON logging with request and trace identifiers.
- Stable JSON error envelope.
- Pydantic v2 settings validation.
- Placeholder dependency interfaces for future Postgres and Redis checks.
- Fly.io process configuration for API and worker apps.

It does not route production AI traffic, consume production queues, create a Python database schema, or call production models.

## Architecture

The repository contains one deployable Python project with two process entry points:

- API: `squadpitch-ai`, a FastAPI service for internal health/readiness and future signed service calls.
- Worker: `squadpitch-ai-worker`, a background process with a placeholder job registry.

The browser must not call this service. Node AI generation remains the production path, and the Node `ai_platform_enabled` feature flag remains disabled by default.

## Local Setup

Install `uv`, then run:

```powershell
uv sync --frozen --dev
```

## Test Commands

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run coverage run -m pytest
uv run coverage report
uv run pip-audit
```

## Runtime

API:

```powershell
uv run python -m squadpitch_ai.api.app
```

Worker:

```powershell
uv run python -m squadpitch_ai.worker.app
```

## Fly Deployment

Fly apps:

- API: `squadpitch-ai`
- Worker: `squadpitch-ai-worker`

Configs:

- API: `infrastructure/fly/squadpitch-ai.toml`
- Worker: `infrastructure/fly/squadpitch-ai-worker.toml`

The API exposes `/health` and `/ready`. The worker does not expose an HTTP service.

## CI/CD

`CI` runs on pull requests to `main`, pushes to `main`, and manual dispatch. It runs:

- `uv sync --frozen --dev`
- `uv run pytest`
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run mypy .`
- coverage with the configured threshold
- `uv run pip-audit`
- Docker build validation

`Deploy` runs after successful `CI` on `main` and can also be run manually for `all`, `api`, or `worker`. Normal deployment order is API first, then worker after API verification.

Required GitHub secret names:

- `FLY_API_TOKEN_API` for the `production-api` environment, or fallback `FLY_API_TOKEN`
- `FLY_API_TOKEN_WORKER` for the `production-worker` environment, or fallback `FLY_API_TOKEN`

Do not store secret values in source files.

## Rollback Overview

Redeploy a previous commit:

```powershell
git checkout <commit-sha>
fly deploy --config infrastructure/fly/squadpitch-ai.toml --app squadpitch-ai --remote-only
fly deploy --config infrastructure/fly/squadpitch-ai-worker.toml --app squadpitch-ai-worker --remote-only
```

Scale down:

```powershell
fly scale count app=0 --app squadpitch-ai --yes
fly scale count worker=0 --app squadpitch-ai-worker --yes
```

Keep Node integration disabled by leaving `ai_platform_enabled=false` and not setting any production behavior flag.

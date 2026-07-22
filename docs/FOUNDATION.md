# Python AI Platform Foundation

This milestone establishes the Python service boundary accepted by ADR-001 through ADR-004:

- Node remains the production control plane.
- Python exposes health/readiness and worker process foundations only.
- No production AI traffic is routed to Python.
- No Python-owned database schema is introduced.
- Postgres and Redis are represented as future dependency interfaces, not mandatory health dependencies.

Structured logs support `requestId`, `traceId`, `taskName`, `taskVersion`, `schemaVersion`, `errorCode`, and `latencyMs`. Health checks may leave task fields absent.

from pydantic import BaseModel, ConfigDict, Field


class AiLogContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = Field(default=None, serialization_alias="requestId")
    trace_id: str | None = Field(default=None, serialization_alias="traceId")
    task_name: str | None = Field(default=None, serialization_alias="taskName")
    task_version: str | None = Field(default=None, serialization_alias="taskVersion")
    schema_version: str | None = Field(default=None, serialization_alias="schemaVersion")
    error_code: str | None = Field(default=None, serialization_alias="errorCode")
    latency_ms: int | None = Field(default=None, serialization_alias="latencyMs")

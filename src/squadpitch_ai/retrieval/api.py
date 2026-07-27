from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from squadpitch_ai.retrieval.models import (
    ACLScope,
    IndexingEvent,
    RetrievalQuery,
    RetrievalResult,
    SourceType,
    TrustClassification,
)
from squadpitch_ai.retrieval.retriever import HybridRetriever
from squadpitch_ai.retrieval.store import InMemoryRetrievalStore


class RetrievalQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["retrieval-query.v1"] = Field(
        default="retrieval-query.v1", alias="schemaVersion"
    )
    workspace_id: str = Field(min_length=1, max_length=128, alias="workspaceId")
    query: str = Field(min_length=1, max_length=2_000)
    purpose: Literal["campaign_context", "content_generation", "verification"]
    top_k: int = Field(default=5, ge=1, le=25, alias="topK")
    source_types: list[SourceType] | None = Field(default=None, alias="sourceTypes")
    acl_scopes: list[ACLScope] | None = Field(default=None, alias="aclScopes")
    language: str | None = Field(default=None, min_length=2, max_length=16)
    trust_classifications: list[TrustClassification] | None = Field(
        default=None, alias="trustClassifications"
    )
    max_context_chars: int = Field(default=4_000, ge=100, le=24_000, alias="maxContextChars")
    indexing_events: list[IndexingEvent] = Field(
        default_factory=list, max_length=100, alias="indexingEvents"
    )
    trace_id: str = Field(min_length=1, max_length=128, alias="traceId")

    @model_validator(mode="after")
    def indexing_events_must_match_workspace(self) -> RetrievalQueryRequest:
        if any(event.workspace_id != self.workspace_id for event in self.indexing_events):
            raise ValueError("Indexing event workspace does not match query workspace")
        return self


class RetrievalQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["retrieval-query-response.v1"] = Field(
        default="retrieval-query-response.v1", alias="schemaVersion"
    )
    workspace_id: str = Field(alias="workspaceId")
    purpose: str
    results: list[RetrievalResult]
    result_count: int = Field(ge=0, alias="resultCount")
    empty: bool
    top_k: int = Field(ge=1, le=25, alias="topK")
    trace_id: str = Field(alias="traceId")


def execute_retrieval_query(request: RetrievalQueryRequest) -> RetrievalQueryResponse:
    store = InMemoryRetrievalStore()
    for event in request.indexing_events:
        store.apply_event(event)
    results = HybridRetriever(store).search(
        RetrievalQuery(
            workspaceId=request.workspace_id,
            query=request.query,
            sourceTypes=request.source_types,
            aclScopes=request.acl_scopes,
            language=request.language,
            trustClassifications=request.trust_classifications,
            limit=request.top_k,
            maxContextChars=request.max_context_chars,
        )
    )
    if any(result.citation.workspace_id != request.workspace_id for result in results):
        raise ValueError("Retrieval result crossed workspace boundary")
    return RetrievalQueryResponse(
        workspaceId=request.workspace_id,
        purpose=request.purpose,
        results=results,
        resultCount=len(results),
        empty=len(results) == 0,
        topK=request.top_k,
        traceId=request.trace_id,
    )

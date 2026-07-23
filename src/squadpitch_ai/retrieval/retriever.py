from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from squadpitch_ai.retrieval.embeddings import cosine_similarity, embed_text, tokenize
from squadpitch_ai.retrieval.models import (
    ACLScope,
    RetrievalCitation,
    RetrievalQuery,
    RetrievalResult,
    RetrievalRow,
    SourceType,
    TrustClassification,
)
from squadpitch_ai.retrieval.sanitize import contains_instruction_injection
from squadpitch_ai.retrieval.store import InMemoryRetrievalStore, RetrievalSecurityError

TRUST_WEIGHTS: dict[TrustClassification, float] = {
    TrustClassification.AUTHORITATIVE: 1.0,
    TrustClassification.APPROVED: 0.9,
    TrustClassification.DERIVED: 0.65,
    TrustClassification.USER_SUPPLIED: 0.5,
    TrustClassification.PRIVATE_SENSITIVE: 0.35,
    TrustClassification.LOW_TRUST: 0.2,
}


class Reranker(Protocol):
    def rerank(
        self, query: RetrievalQuery, results: list[RetrievalResult]
    ) -> list[RetrievalResult]: ...


class HybridRetriever:
    def __init__(self, store: InMemoryRetrievalStore, reranker: Reranker | None = None) -> None:
        self._store = store
        self._reranker = reranker

    def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        if not query.workspace_id:
            raise RetrievalSecurityError("workspace_id is required for retrieval")

        query_vector = embed_text(query.query)
        query_tokens = tokenize(query.query)
        scored: list[RetrievalResult] = []

        for row in self._candidate_rows(query):
            keyword_score = self._keyword_score(query_tokens, row.sanitized_text)
            vector_score = cosine_similarity(query_vector, row.embedding)
            if keyword_score == 0.0 and vector_score < 0.55:
                continue
            recency_score = self._recency_score(row.updated_at)
            trust_score = TRUST_WEIGHTS[row.trust_classification]
            fused_score = (
                (0.45 * keyword_score)
                + (0.35 * vector_score)
                + (0.10 * recency_score)
                + (0.10 * trust_score)
            )
            if fused_score <= 0.0:
                continue
            scored.append(
                RetrievalResult(
                    text=row.sanitized_text,
                    citation=RetrievalCitation(
                        workspaceId=row.workspace_id,
                        sourceType=row.source_type,
                        sourceId=row.source_id,
                        contentHash=row.content_hash,
                        chunkId=row.chunk_id,
                        trustClassification=row.trust_classification,
                        language=row.language,
                    ),
                    score=fused_score,
                    keywordScore=keyword_score,
                    vectorScore=vector_score,
                    recencyScore=recency_score,
                    trustScore=trust_score,
                    metadata=row.metadata,
                    containsUntrustedInstruction=contains_instruction_injection(row.chunk_text)
                    or row.metadata.get("sourceContainsUntrustedInstruction") is True,
                ),
            )

        ordered = sorted(scored, key=lambda result: result.score, reverse=True)
        if self._reranker is not None:
            ordered = self._reranker.rerank(query, ordered)
        return self._fit_context_budget(ordered[: query.limit], query.max_context_chars)

    def _candidate_rows(self, query: RetrievalQuery) -> list[RetrievalRow]:
        rows = self._store.active_rows(query.workspace_id)
        if query.source_types is not None:
            rows = [row for row in rows if row.source_type in query.source_types]
        if query.acl_scopes is not None:
            rows = [row for row in rows if row.acl_scope in query.acl_scopes]
        if query.language is not None:
            rows = [row for row in rows if row.language == query.language]
        if query.trust_classifications is not None:
            rows = [row for row in rows if row.trust_classification in query.trust_classifications]
        return rows

    @staticmethod
    def _keyword_score(query_tokens: list[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        text_tokens = set(tokenize(text))
        matches = sum(1 for token in query_tokens if token in text_tokens)
        return matches / len(query_tokens)

    @staticmethod
    def _recency_score(updated_at: datetime) -> float:
        age_days = max((datetime.now(UTC) - updated_at).total_seconds() / 86_400, 0.0)
        return max(0.0, 1.0 - (age_days / 365.0))

    @staticmethod
    def _fit_context_budget(
        results: list[RetrievalResult],
        max_context_chars: int,
    ) -> list[RetrievalResult]:
        selected: list[RetrievalResult] = []
        used = 0
        for result in results:
            next_used = used + len(result.text)
            if next_used > max_context_chars:
                continue
            selected.append(result)
            used = next_used
        return selected


def build_real_estate_campaign_context(
    store: InMemoryRetrievalStore,
    workspace_id: str,
    property_source_id: str | None = None,
    language: str = "en",
    max_context_chars: int = 4_000,
) -> list[RetrievalResult]:
    query_text = "real estate campaign property facts brand voice approved examples calendar"
    if property_source_id:
        query_text = f"{query_text} {property_source_id}"
    return HybridRetriever(store).search(
        RetrievalQuery(
            workspaceId=workspace_id,
            query=query_text,
            sourceTypes=[
                SourceType.PROPERTY_LISTING,
                SourceType.BRAND_PROFILE,
                SourceType.VOICE_PROFILE,
                SourceType.DRAFT,
                SourceType.PUBLISHING_CALENDAR_SNAPSHOT,
            ],
            aclScopes=[ACLScope.CAMPAIGN_CONTEXT],
            language=language,
            limit=10,
            maxContextChars=max_context_chars,
        ),
    )

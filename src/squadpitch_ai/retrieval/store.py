from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime

from squadpitch_ai.retrieval.embeddings import embed_text
from squadpitch_ai.retrieval.models import (
    DeletionStatus,
    IndexingEvent,
    IndexingOperation,
    RetrievalRow,
    SourceType,
)
from squadpitch_ai.retrieval.sanitize import (
    contains_instruction_injection,
    normalize_source_text,
    sanitize_retrieved_text,
)

CHUNK_SPLIT_PATTERN = re.compile(r"(?:\n\s*){2,}|(?<=[.!?])\s+")
MAX_CHUNK_CHARS = 900


class RetrievalSecurityError(ValueError):
    pass


def content_hash_for_text(text: str) -> str:
    digest = hashlib.sha256(normalize_source_text(text).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _payload_text(payload: dict[str, object]) -> str:
    value = payload.get("approvedText") or payload.get("text") or payload.get("content")
    if isinstance(value, str):
        return value
    facts = payload.get("facts")
    if isinstance(facts, dict):
        return " ".join(f"{key}: {value}" for key, value in sorted(facts.items()))
    return ""


def _chunk_text(text: str) -> list[str]:
    chunks: list[str] = []
    seen_hashes: set[str] = set()
    for raw_part in CHUNK_SPLIT_PATTERN.split(text):
        part = normalize_source_text(raw_part)
        if not part:
            continue
        while len(part) > MAX_CHUNK_CHARS:
            chunks.append(part[:MAX_CHUNK_CHARS].strip())
            part = part[MAX_CHUNK_CHARS:].strip()
        chunks.append(part)

    deduped: list[str] = []
    for chunk in chunks:
        chunk_hash = content_hash_for_text(chunk)
        if chunk_hash in seen_hashes:
            continue
        seen_hashes.add(chunk_hash)
        deduped.append(chunk)
    return deduped


class InMemoryRetrievalStore:
    def __init__(self) -> None:
        self._rows: dict[str, RetrievalRow] = {}
        self._processed_events: set[str] = set()

    def apply_event(self, event: IndexingEvent) -> list[RetrievalRow]:
        if not event.workspace_id:
            raise RetrievalSecurityError("workspace_id is required for indexing")
        if event.event_id in self._processed_events:
            return self.active_rows_for_source(
                event.workspace_id,
                event.source_type,
                event.source_id,
            )

        if event.operation in {IndexingOperation.DELETE, IndexingOperation.PERMISSION_CHANGE}:
            status = (
                DeletionStatus.DELETED
                if event.operation == IndexingOperation.DELETE
                else DeletionStatus.INVALIDATED
            )
            self.invalidate_source(event.workspace_id, event.source_type, event.source_id, status)
            self._processed_events.add(event.event_id)
            return []

        text = _payload_text(event.payload)
        if not text:
            self._processed_events.add(event.event_id)
            return []

        source_contains_untrusted_instruction = contains_instruction_injection(text)
        self.invalidate_source(
            event.workspace_id,
            event.source_type,
            event.source_id,
            DeletionStatus.INVALIDATED,
            except_content_hash=event.content_hash,
        )

        now = datetime.now(UTC)
        created: list[RetrievalRow] = []
        for index, chunk in enumerate(_chunk_text(text)):
            sanitized = sanitize_retrieved_text(chunk)
            chunk_hash = content_hash_for_text(f"{event.workspace_id}:{event.source_id}:{chunk}")
            chunk_hash_prefix = chunk_hash.removeprefix("sha256:")[:16]
            chunk_id = f"{event.source_type.value}:{event.source_id}:{chunk_hash_prefix}"
            if chunk_id in self._rows and self._rows[chunk_id].content_hash == event.content_hash:
                continue

            row = RetrievalRow(
                workspaceId=event.workspace_id,
                sourceType=event.source_type,
                sourceId=event.source_id,
                contentHash=event.content_hash,
                aclScope=event.acl_scope,
                language=event.language,
                createdAt=now,
                updatedAt=event.source_updated_at,
                deletionStatus=DeletionStatus.ACTIVE,
                trustClassification=event.trust_classification,
                chunkId=chunk_id,
                chunkIndex=index,
                chunkText=chunk,
                sanitizedText=sanitized,
                embedding=embed_text(sanitized),
                metadata={
                    key: value
                    for key, value in event.payload.items()
                    if key not in {"approvedText", "text", "content"}
                }
                | {"sourceContainsUntrustedInstruction": source_contains_untrusted_instruction},
            )
            self._rows[chunk_id] = row
            created.append(row)

        self._processed_events.add(event.event_id)
        return created

    def invalidate_source(
        self,
        workspace_id: str,
        source_type: SourceType,
        source_id: str,
        deletion_status: DeletionStatus,
        except_content_hash: str | None = None,
    ) -> None:
        for chunk_id, row in list(self._rows.items()):
            if (
                row.workspace_id == workspace_id
                and row.source_type == source_type
                and row.source_id == source_id
                and row.deletion_status == DeletionStatus.ACTIVE
                and row.content_hash != except_content_hash
            ):
                self._rows[chunk_id] = row.model_copy(
                    update={
                        "deletion_status": deletion_status,
                        "updated_at": datetime.now(UTC),
                    },
                )

    def active_rows(self, workspace_id: str) -> list[RetrievalRow]:
        if not workspace_id:
            raise RetrievalSecurityError("workspace_id is required for retrieval")
        return [
            row
            for row in self._rows.values()
            if row.workspace_id == workspace_id and row.deletion_status == DeletionStatus.ACTIVE
        ]

    def active_rows_for_source(
        self,
        workspace_id: str,
        source_type: SourceType,
        source_id: str,
    ) -> list[RetrievalRow]:
        return [
            row
            for row in self.active_rows(workspace_id)
            if row.source_type == source_type and row.source_id == source_id
        ]

    def all_rows(self) -> list[RetrievalRow]:
        return list(self._rows.values())

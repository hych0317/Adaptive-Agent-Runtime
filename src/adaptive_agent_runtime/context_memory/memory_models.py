"""Conditional, evidence-backed Memory models."""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping
from uuid import UUID, uuid4

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from adaptive_agent_runtime.context_memory.json_types import (
    ContextMemoryModel,
    ImmutableJsonObject,
    ImmutableJsonValue,
    utc_now,
)


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    CONFLICTED = "conflicted"
    RETIRED = "retired"


class MemorySensitivity(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class MemoryScope(ContextMemoryModel):
    """Authority scope used before Memory can enter an Agent projection."""

    tenant_id: str = Field(default="default", min_length=1)
    project_id: str = Field(default="default", min_length=1)
    agent_scope: str = Field(default="shared", min_length=1)


class MemoryEvolutionType(StrEnum):
    SUPPORT = "support"
    MODIFY = "modify"
    EXTEND = "extend"
    CONFLICT = "conflict"


class MemoryCondition(ContextMemoryModel):
    facts: ImmutableJsonObject = Field(default_factory=dict)
    required_tags: tuple[str, ...] = ()
    description: str | None = None

    @model_validator(mode="after")
    def validate_condition(self) -> MemoryCondition:
        if len(set(self.required_tags)) != len(self.required_tags):
            raise ValueError("memory condition tags must be unique")
        return self

    def matches(
        self,
        facts: Mapping[str, JsonValue],
        tags: tuple[str, ...],
    ) -> bool:
        if not set(self.required_tags).issubset(tags):
            return False
        return all(
            key in facts and facts[key] == value
            for key, value in self.facts.items()
        )


class MemoryEvidence(ContextMemoryModel):
    evidence_id: UUID = Field(default_factory=uuid4)
    source_context_id: UUID | None = None
    source_reference: str | None = None
    note: str = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0.0, le=1.0)
    observed_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_source(self) -> MemoryEvidence:
        if self.source_context_id is None and self.source_reference is None:
            raise ValueError("memory evidence requires a source reference")
        return self


class MemoryConflict(ContextMemoryModel):
    candidate_id: UUID
    content: ImmutableJsonValue
    condition: MemoryCondition
    evidence: tuple[MemoryEvidence, ...] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    recorded_at: AwareDatetime = Field(default_factory=utc_now)


class MemoryCandidate(ContextMemoryModel):
    candidate_id: UUID = Field(default_factory=uuid4)
    memory_key: str = Field(min_length=1)
    content: ImmutableJsonValue
    condition: MemoryCondition
    evidence: tuple[MemoryEvidence, ...] = Field(min_length=1)
    confidence: float = Field(gt=0.0, le=1.0)
    evolution: MemoryEvolutionType
    target_memory_id: UUID | None = None
    scope: MemoryScope = Field(default_factory=MemoryScope)
    sensitivity: MemorySensitivity = MemorySensitivity.INTERNAL
    expires_at: AwareDatetime | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_candidate(self) -> MemoryCandidate:
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("candidate evidence ids must be unique")
        if self.evolution is MemoryEvolutionType.EXTEND:
            if self.target_memory_id is not None:
                raise ValueError("extend candidates cannot target an existing memory")
        elif self.target_memory_id is None:
            raise ValueError(
                f"{self.evolution.value} candidates require target_memory_id"
            )
        return self


class MemoryUnit(ContextMemoryModel):
    memory_id: UUID = Field(default_factory=uuid4)
    memory_key: str = Field(min_length=1)
    content: ImmutableJsonValue
    condition: MemoryCondition
    evidence: tuple[MemoryEvidence, ...] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    scope: MemoryScope = Field(default_factory=MemoryScope)
    sensitivity: MemorySensitivity = MemorySensitivity.INTERNAL
    expires_at: AwareDatetime | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    conflicts: tuple[MemoryConflict, ...] = ()
    revision: int = Field(default=0, ge=0)
    last_candidate_id: UUID | None = None
    last_candidate_fingerprint: str | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_memory(self) -> MemoryUnit:
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("memory evidence ids must be unique")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        if self.status is MemoryStatus.CONFLICTED and not self.conflicts:
            raise ValueError("a conflicted memory requires conflict records")
        if self.status is MemoryStatus.ACTIVE and self.conflicts:
            raise ValueError("an active memory cannot contain unresolved conflicts")
        if (self.last_candidate_id is None) != (
            self.last_candidate_fingerprint is None
        ):
            raise ValueError(
                "candidate id and candidate fingerprint must be recorded together"
            )
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("memory expiry must be later than creation")
        return self


class MemoryUpdateResult(ContextMemoryModel):
    evolution: MemoryEvolutionType
    memory: MemoryUnit
    previous_memory: MemoryUnit | None = None
    candidate_id: UUID


class MemoryBatchWrite(ContextMemoryModel):
    memory: MemoryUnit
    expected_revision: int | None = Field(default=None, ge=0)


class MemoryBatchCommitReceipt(ContextMemoryModel):
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    memory_count: int = Field(ge=0)
    committed_at: AwareDatetime = Field(default_factory=utc_now)


class MemoryRecallQuery(ContextMemoryModel):
    facts: ImmutableJsonObject = Field(default_factory=dict)
    tags: tuple[str, ...] = ()
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    limit: int = Field(default=10, ge=1)
    include_conflicted: bool = False

    @model_validator(mode="after")
    def validate_query(self) -> MemoryRecallQuery:
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("memory recall tags must be unique")
        return self

"""Runtime-owned models for governed Agent-proposed Memory extraction."""

from __future__ import annotations

import hashlib
import json
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, JsonValue, model_validator

from adaptive_agent_runtime.context_memory.json_types import ContextMemoryModel
from adaptive_agent_runtime.context_memory.memory_models import (
    MemoryCandidate,
    MemoryUpdateResult,
)


MEMORY_EXTRACTION_DECISION_TYPE = "memory.candidate_extraction"
MEMORY_CANDIDATES_APPLY_OPERATION = "memory.write"

_MEMORY_DECISION_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "adaptive-agent-runtime/context-memory/extraction-decision",
)


def stable_memory_extraction_id(*parts: object) -> UUID:
    return uuid5(_MEMORY_DECISION_NAMESPACE, "|".join(str(item) for item in parts))


def memory_decision_fingerprint(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif isinstance(value, (list, tuple)):
        value = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in value
        ]
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class MemoryExtractionExecutionPolicy(ContextMemoryModel):
    timeout_seconds: float = Field(default=20.0, gt=0.0)
    max_agent_calls: int = Field(default=1, ge=1, le=1)
    max_revision_count: int = Field(default=0, ge=0, le=0)
    max_candidates: int = Field(default=8, ge=1)


class MemoryEvidenceBinding(ContextMemoryModel):
    reference_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    source_reference: str = Field(min_length=1)
    source_context_id: UUID | None = None
    reliability: float = Field(default=1.0, gt=0.0, le=1.0)


class ExistingMemoryBinding(ContextMemoryModel):
    reference_id: str = Field(min_length=1)
    memory_id: UUID
    revision: int = Field(ge=0)
    memory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class MemoryExtractionDecisionPayload(ContextMemoryModel):
    """Runtime-owned bindings plus a pre-authorized semantic projection."""

    agent_input: JsonValue
    evidence: tuple[MemoryEvidenceBinding, ...] = Field(min_length=1)
    existing_memories: tuple[ExistingMemoryBinding, ...] = ()
    state_revision: int = Field(ge=0)
    execution_policy: MemoryExtractionExecutionPolicy = Field(
        default_factory=MemoryExtractionExecutionPolicy
    )

    @model_validator(mode="after")
    def validate_bindings(self) -> MemoryExtractionDecisionPayload:
        evidence_refs = tuple(item.reference_id for item in self.evidence)
        memory_refs = tuple(item.reference_id for item in self.existing_memories)
        memory_ids = tuple(item.memory_id for item in self.existing_memories)
        if len(set(evidence_refs)) != len(evidence_refs):
            raise ValueError("Memory evidence references must be unique")
        if len(set(memory_refs)) != len(memory_refs) or len(set(memory_ids)) != len(memory_ids):
            raise ValueError("existing Memory bindings must be one-to-one")
        return self

    def memory_id_for(self, reference_id: str) -> UUID:
        for item in self.existing_memories:
            if item.reference_id == reference_id:
                return item.memory_id
        raise ValueError("Memory proposal targets an unknown Runtime Memory")


class MemoryExtractionEffect(ContextMemoryModel):
    candidates: tuple[MemoryCandidate, ...] = ()
    candidate_batch_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_batch(self) -> MemoryExtractionEffect:
        if self.candidate_batch_fingerprint != memory_decision_fingerprint(
            self.candidates
        ):
            raise ValueError("Memory candidate batch fingerprint does not match")
        return self


class MemoryExtractionDecisionOutcome(ContextMemoryModel):
    request_id: UUID
    effect: MemoryExtractionEffect
    updates: tuple[MemoryUpdateResult, ...]

    @model_validator(mode="after")
    def validate_updates(self) -> MemoryExtractionDecisionOutcome:
        if tuple(item.candidate_id for item in self.updates) != tuple(
            item.candidate_id for item in self.effect.candidates
        ):
            raise ValueError("Memory updates differ from the governed effect")
        return self

"""Governed, immutable Memory Recall decision contracts and candidate resolution."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid5

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from adaptive_agent_runtime.context_memory.contracts import MemoryStore
from adaptive_agent_runtime.context_memory.json_types import (
    ContextMemoryModel,
    ImmutableJsonObject,
    ImmutableJsonValue,
    estimate_tokens,
    utc_now,
)
from adaptive_agent_runtime.context_memory.memory_models import (
    MemoryScope,
    MemorySensitivity,
    MemoryStatus,
    MemoryUnit,
)
from adaptive_agent_runtime.context_memory.experience import (
    ExperienceExecutionOutcome,
    ExperienceMetadataStore,
)
from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.governance.models import GovernanceTarget, RuntimeCommitPermit


MEMORY_RECALL_DECISION_TYPE = "memory.recall"
MEMORY_RECALL_COMMIT_OPERATION = "memory.recall.commit"
_RECALL_NAMESPACE = UUID("326e3fc4-aa08-46c0-a72f-15dc8ce8a8e4")
_SECRET_KEYS = frozenset(
    {"api_key", "authorization", "credential", "password", "secret", "token"}
)


class MemoryConfidenceBand(StrEnum):
    MEDIUM = "medium"
    HIGH = "high"


class MemoryRecallCandidate(ContextMemoryModel):
    """Sanitized candidate visible to the Recall Agent."""

    candidate_ref: str = Field(min_length=1)
    sanitized_content: ImmutableJsonValue
    memory_category: str = Field(min_length=1)
    confidence_band: MemoryConfidenceBand
    provenance_summary: tuple[str, ...] = Field(min_length=1)
    estimated_tokens: int = Field(ge=1)


class MemoryRecallCandidateBinding(ContextMemoryModel):
    """Runtime-only binding from an opaque candidate to one Memory snapshot."""

    candidate: MemoryRecallCandidate
    memory_id: UUID
    memory_revision: int = Field(ge=0)
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope: MemoryScope
    sensitivity: MemorySensitivity


class MemoryRecallRequest(ContextMemoryModel):
    goal: str = Field(min_length=1)
    scope: MemoryScope
    facts: ImmutableJsonObject = Field(default_factory=dict)
    tags: tuple[str, ...] = ()
    candidates: tuple[MemoryRecallCandidateBinding, ...] = ()
    candidate_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_selected_items: int = Field(default=3, ge=1)
    max_selected_tokens: int = Field(default=512, ge=1)

    @model_validator(mode="after")
    def validate_candidates(self) -> MemoryRecallRequest:
        refs = tuple(item.candidate.candidate_ref for item in self.candidates)
        if len(set(refs)) != len(refs):
            raise ValueError("Memory Recall candidate refs must be unique")
        if self.candidate_set_fingerprint != decision_fingerprint(self.candidates):
            raise ValueError("Memory Recall candidate-set fingerprint is invalid")
        return self


class MemoryRecallAgentRequest(ContextMemoryModel):
    """The complete and only candidate projection visible to the Recall Agent."""

    goal: str = Field(min_length=1)
    candidates: tuple[MemoryRecallCandidate, ...] = Field(min_length=1)
    max_selected_items: int = Field(ge=1)
    max_selected_tokens: int = Field(ge=1)


class MemoryRecallDraft(ContextMemoryModel):
    """Authority-free Agent selection containing only opaque references."""

    selected_candidate_refs: tuple[str, ...]
    selection_reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_refs(self) -> MemoryRecallDraft:
        if len(set(self.selected_candidate_refs)) != len(
            self.selected_candidate_refs
        ):
            raise ValueError("Memory Recall selection contains duplicate refs")
        return self


class MemoryRecallSourceSnapshot(ContextMemoryModel):
    candidate_ref: str = Field(min_length=1)
    memory_id: UUID
    memory_revision: int = Field(ge=0)
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    sanitized_content: ImmutableJsonValue
    memory_category: str = Field(min_length=1)
    confidence_band: MemoryConfidenceBand
    provenance_summary: tuple[str, ...] = Field(min_length=1)
    estimated_tokens: int = Field(ge=1)
    scope: MemoryScope
    sensitivity: MemorySensitivity


class MemoryRecallEffect(ContextMemoryModel):
    recall_decision_id: UUID
    run_id: UUID
    task_id: UUID
    scope: MemoryScope
    sources: tuple[MemoryRecallSourceSnapshot, ...] = Field(min_length=1)
    max_items: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    candidate_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_budget_and_scope(self) -> MemoryRecallEffect:
        if len(self.sources) > self.max_items:
            raise ValueError("Recall Effect exceeds item budget")
        if sum(item.estimated_tokens for item in self.sources) > self.max_tokens:
            raise ValueError("Recall Effect exceeds token budget")
        if any(
            item.scope.tenant_id != self.scope.tenant_id
            or item.scope.project_id != self.scope.project_id
            or item.scope.agent_scope not in {"shared", self.scope.agent_scope}
            for item in self.sources
        ):
            raise ValueError("Recall Effect contains an unauthorized Memory scope")
        return self


class MemoryRecallBundleItem(ContextMemoryModel):
    source_memory_ref: str = Field(min_length=1)
    source_memory_version: int = Field(ge=0)
    content: ImmutableJsonValue
    memory_category: str = Field(min_length=1)
    provenance: tuple[str, ...] = Field(min_length=1)


class MemoryRecallBundle(ContextMemoryModel):
    bundle_id: UUID
    recall_decision_id: UUID
    run_id: UUID
    task_id: UUID
    items: tuple[MemoryRecallBundleItem, ...] = Field(min_length=1)
    scope: MemoryScope
    max_items: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    used_tokens: int = Field(ge=1)
    candidate_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: tuple[str, ...] = Field(min_length=1)
    committed_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_budget(self) -> MemoryRecallBundle:
        if len(self.items) > self.max_items:
            raise ValueError("Recall Bundle exceeds item budget")
        if self.used_tokens > self.max_tokens:
            raise ValueError("Recall Bundle exceeds token budget")
        return self


class MemoryRecallCommitReceipt(ContextMemoryModel):
    bundle_id: UUID
    recall_decision_id: UUID
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_count: int = Field(ge=1)
    committed_at: AwareDatetime = Field(default_factory=utc_now)


class MemoryRecallEligibilityPolicy(ContextMemoryModel):
    scope: MemoryScope
    allowed_agent_scopes: tuple[str, ...] = ("shared", "planner")
    allowed_sensitivities: tuple[MemorySensitivity, ...] = (
        MemorySensitivity.PUBLIC,
        MemorySensitivity.INTERNAL,
    )
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    max_candidates: int = Field(default=8, ge=1)
    max_candidate_tokens: int = Field(default=1024, ge=1)
    require_verified_experience: bool = False
    exclude_failed_experience: bool = True
    exclude_inconsistent_experience: bool = True


class MemoryRecallBundleStore(Protocol):
    async def commit(
        self,
        effect: MemoryRecallEffect,
        *,
        effect_fingerprint: str,
        permit: RuntimeCommitPermit | None,
        target: GovernanceTarget | None,
        subject_fingerprint: str | None,
    ) -> MemoryRecallBundle: ...

    async def load_by_effect(self, effect_fingerprint: str) -> MemoryRecallBundle | None: ...


class RuntimeMemoryCandidateResolver:
    """Deterministic metadata/keyword resolver before Agent projection."""

    module_id = "memory.recall.candidate_resolver"

    def __init__(
        self,
        store: MemoryStore,
        experience_store: ExperienceMetadataStore | None = None,
    ) -> None:
        self._store = store
        self._experience_store = experience_store

    async def resolve(
        self,
        *,
        request_id: UUID,
        goal: str,
        facts: Mapping[str, JsonValue],
        tags: tuple[str, ...],
        policy: MemoryRecallEligibilityPolicy,
        now: datetime | None = None,
    ) -> tuple[MemoryRecallCandidateBinding, ...]:
        observed_at = now or datetime.now(timezone.utc)
        eligible: list[tuple[int, MemoryUnit, tuple[str, ...]]] = []
        keywords = {item for item in _terms(goal) if len(item) > 1}
        for memory in await self._store.list_all():
            if memory.status is not MemoryStatus.ACTIVE:
                continue
            if memory.expires_at is not None and memory.expires_at <= observed_at:
                continue
            if memory.confidence < policy.min_confidence:
                continue
            if memory.scope.tenant_id != policy.scope.tenant_id:
                continue
            if memory.scope.project_id != policy.scope.project_id:
                continue
            if memory.scope.agent_scope not in policy.allowed_agent_scopes:
                continue
            if memory.sensitivity not in policy.allowed_sensitivities:
                continue
            if not memory.condition.matches(facts, tags):
                continue
            if not memory.evidence or any(
                item.source_context_id is None and item.source_reference is None
                for item in memory.evidence
            ):
                continue
            experience_provenance: tuple[str, ...] = ()
            if self._experience_store is not None:
                experiences = await self._experience_store.list_for_memory(
                    memory.memory_id
                )
                outcomes = {item.execution_outcome for item in experiences}
                if (
                    policy.exclude_inconsistent_experience
                    and len(outcomes) > 1
                ):
                    continue
                if experiences:
                    latest = max(
                        experiences,
                        key=lambda item: item.committed_at,
                    )
                    if (
                        policy.exclude_failed_experience
                        and latest.execution_outcome
                        is ExperienceExecutionOutcome.FAILED
                    ):
                        continue
                    experience_provenance = (
                        "experience:"
                        f"{latest.execution_outcome.value}:"
                        f"{latest.committed_at.isoformat()}",
                    )
                elif policy.require_verified_experience:
                    continue
            searchable = " ".join(
                (memory.memory_key, str(memory.content), memory.condition.description or "")
            ).lower()
            score = sum(1 for keyword in keywords if keyword in searchable)
            eligible.append((score, memory, experience_provenance))
        eligible.sort(
            key=lambda pair: (
                -pair[0],
                -pair[1].confidence,
                -pair[1].updated_at.timestamp(),
                str(pair[1].memory_id),
            )
        )
        result: list[MemoryRecallCandidateBinding] = []
        used_tokens = 0
        for _, memory, experience_provenance in eligible:
            content = _sanitize(memory.content)
            tokens = estimate_tokens(content)
            if result and used_tokens + tokens > policy.max_candidate_tokens:
                continue
            if tokens > policy.max_candidate_tokens:
                continue
            source_fingerprint = decision_fingerprint(memory)
            candidate_ref = "candidate:" + str(
                uuid5(
                    request_id,
                    f"{memory.memory_id}:{memory.revision}:{source_fingerprint}",
                )
            )
            provenance = tuple(
                (item.source_reference or f"context:{item.source_context_id}")
                for item in memory.evidence[:3]
            ) + experience_provenance
            category = memory.memory_key.rsplit(".", 1)[-1]
            projected = MemoryRecallCandidate(
                candidate_ref=candidate_ref,
                sanitized_content=content,
                memory_category=category,
                confidence_band=(
                    MemoryConfidenceBand.HIGH
                    if memory.confidence >= 0.85
                    else MemoryConfidenceBand.MEDIUM
                ),
                provenance_summary=provenance,
                estimated_tokens=tokens,
            )
            result.append(
                MemoryRecallCandidateBinding(
                    candidate=projected,
                    memory_id=memory.memory_id,
                    memory_revision=memory.revision,
                    source_fingerprint=source_fingerprint,
                    scope=memory.scope,
                    sensitivity=memory.sensitivity,
                )
            )
            used_tokens += tokens
            if len(result) >= policy.max_candidates:
                break
        return tuple(result)


def stable_recall_bundle_id(effect_fingerprint: str) -> UUID:
    return uuid5(_RECALL_NAMESPACE, effect_fingerprint)


def _terms(value: str) -> tuple[str, ...]:
    normalized = "".join(character.lower() if character.isalnum() else " " for character in value)
    return tuple(dict.fromkeys(normalized.split()))


def _sanitize(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]" if str(key).lower() in _SECRET_KEYS else _sanitize(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_sanitize(item) for item in value]
    return value

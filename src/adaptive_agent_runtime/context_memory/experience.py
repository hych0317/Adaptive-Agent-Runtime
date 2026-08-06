"""Append-only, Runtime-evidenced Experience Metadata contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol
from uuid import UUID, NAMESPACE_URL, uuid5

from pydantic import AwareDatetime, Field, model_validator

from adaptive_agent_runtime.context_memory.json_types import (
    ContextMemoryModel,
    ImmutableJsonObject,
    utc_now,
)
from adaptive_agent_runtime.governance.models import (
    GovernanceTarget,
    RuntimeCommitPermit,
)


EXPERIENCE_ASSESSMENT_DECISION_TYPE = "experience.assessment"
EXPERIENCE_METADATA_COMMIT_OPERATION = "experience.metadata.commit"


class ExperienceExecutionOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ExperienceArtifactReference(ContextMemoryModel):
    node_id: UUID
    artifact_type: str = Field(min_length=1)
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_decision_id: UUID


class ExperienceEvaluationReference(ContextMemoryModel):
    evaluation_id: UUID
    evaluator_id: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    verdict: str = Field(min_length=1)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    evaluation_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExperienceMemorySource(ContextMemoryModel):
    """Runtime-only binding; it is never projected to the Assessment Agent."""

    memory_id: UUID
    revision: int = Field(ge=0)
    memory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeExecutionObservation(ContextMemoryModel):
    state_revision: int = Field(ge=0)
    state_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_status: str = Field(min_length=1)
    step_count: int = Field(ge=0)
    output_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    artifact_verified: bool
    evaluation_available: bool


class ExperienceAssessmentRequest(ContextMemoryModel):
    execution_id: UUID
    run_id: UUID
    task_id: UUID
    related_decision_ids: tuple[UUID, ...] = Field(min_length=1)
    source_artifacts: tuple[ExperienceArtifactReference, ...] = Field(min_length=1)
    evaluation_refs: tuple[ExperienceEvaluationReference, ...] = Field(min_length=1)
    source_memories: tuple[ExperienceMemorySource, ...] = ()
    runtime_observation: RuntimeExecutionObservation
    execution_outcome: ExperienceExecutionOutcome
    success_signal: bool
    failure_signal: bool

    @model_validator(mode="after")
    def validate_runtime_evidence(self) -> ExperienceAssessmentRequest:
        if self.success_signal == self.failure_signal:
            raise ValueError("Experience requires exactly one Runtime outcome signal")
        if self.execution_outcome is ExperienceExecutionOutcome.SUCCEEDED:
            if not self.success_signal or self.failure_signal:
                raise ValueError("successful Experience requires Runtime success")
        elif not self.failure_signal or self.success_signal:
            raise ValueError("failed Experience requires Runtime failure")
        if not self.runtime_observation.evaluation_available:
            raise ValueError("Experience requires an available Evaluation")
        if not self.runtime_observation.artifact_verified:
            raise ValueError("Experience requires a verified Artifact")
        if len(set(self.related_decision_ids)) != len(self.related_decision_ids):
            raise ValueError("related Decision ids must be unique")
        if len({item.evaluation_id for item in self.evaluation_refs}) != len(
            self.evaluation_refs
        ):
            raise ValueError("Evaluation references must be unique")
        if len({item.memory_id for item in self.source_memories}) != len(
            self.source_memories
        ):
            raise ValueError("source Memory references must be unique")
        return self


class ExperienceAssessmentAgentRequest(ContextMemoryModel):
    """Deny-by-default projection containing no Runtime or Memory identities."""

    execution_summary: ImmutableJsonObject
    evaluation_summary: tuple[ImmutableJsonObject, ...] = Field(min_length=1)
    artifact_summary: tuple[ImmutableJsonObject, ...] = Field(min_length=1)


class ExperienceAssessmentDraft(ContextMemoryModel):
    """Authority-free semantic opinion produced by an Assessment Agent."""

    observed_pattern: str = Field(min_length=1)
    possible_relevance: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class ExperienceMetadataEffect(ContextMemoryModel):
    experience_id: UUID
    source_execution_id: UUID
    source_run_id: UUID
    source_task_id: UUID
    source_decision_ids: tuple[UUID, ...] = Field(min_length=1)
    source_artifacts: tuple[ExperienceArtifactReference, ...] = Field(min_length=1)
    source_evaluations: tuple[ExperienceEvaluationReference, ...] = Field(min_length=1)
    source_memories: tuple[ExperienceMemorySource, ...] = ()
    runtime_observation: RuntimeExecutionObservation
    execution_outcome: ExperienceExecutionOutcome
    success_signal: bool
    failure_signal: bool
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    agent_assessment_summary: ImmutableJsonObject
    proposal_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_effect(self) -> ExperienceMetadataEffect:
        if self.success_signal == self.failure_signal:
            raise ValueError("Experience Effect requires one Runtime outcome signal")
        if self.execution_outcome is ExperienceExecutionOutcome.SUCCEEDED:
            if not self.success_signal:
                raise ValueError("successful Experience Effect lacks success signal")
        elif not self.failure_signal:
            raise ValueError("failed Experience Effect lacks failure signal")
        return self


class ExperienceMetadata(ContextMemoryModel):
    experience_id: UUID
    version: int = Field(ge=1)
    source_execution_id: UUID
    source_run_id: UUID
    source_task_id: UUID
    source_decision_ids: tuple[UUID, ...] = Field(min_length=1)
    source_artifacts: tuple[ExperienceArtifactReference, ...] = Field(min_length=1)
    source_evaluations: tuple[ExperienceEvaluationReference, ...] = Field(min_length=1)
    source_memory_refs: tuple[UUID, ...] = ()
    runtime_observation: RuntimeExecutionObservation
    execution_outcome: ExperienceExecutionOutcome
    success_signal: bool
    failure_signal: bool
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    agent_assessment_summary: ImmutableJsonObject
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    committed_at: AwareDatetime = Field(default_factory=utc_now)


class ExperienceMetadataCommitReceipt(ContextMemoryModel):
    experience_id: UUID
    version: int = Field(ge=1)
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    committed_at: AwareDatetime = Field(default_factory=utc_now)


class ExperienceMetadataStore(Protocol):
    async def commit(
        self,
        effect: ExperienceMetadataEffect,
        *,
        effect_fingerprint: str,
        permit: RuntimeCommitPermit | None,
        target: GovernanceTarget | None,
        subject_fingerprint: str | None,
    ) -> ExperienceMetadata: ...

    async def load_by_effect(
        self,
        effect_fingerprint: str,
    ) -> ExperienceMetadata | None: ...

    async def load_receipt(
        self,
        effect_fingerprint: str,
    ) -> ExperienceMetadataCommitReceipt | None: ...

    async def list_for_run(self, run_id: UUID) -> tuple[ExperienceMetadata, ...]: ...

    async def list_for_memory(
        self,
        memory_id: UUID,
    ) -> tuple[ExperienceMetadata, ...]: ...


def stable_experience_id(request_id: UUID) -> UUID:
    return uuid5(NAMESPACE_URL, f"adaptive-agent-runtime:experience:{request_id}")


def stable_experience_request_id(run_id: UUID, assessment_revision: int = 1) -> UUID:
    if assessment_revision < 1:
        raise ValueError("Experience assessment revision must be positive")
    return uuid5(
        NAMESPACE_URL,
        (
            "adaptive-agent-runtime:experience-assessment:"
            f"{run_id}:{assessment_revision}"
        ),
    )

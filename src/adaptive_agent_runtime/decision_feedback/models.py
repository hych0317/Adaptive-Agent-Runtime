"""Strongly typed contracts for governed Decision outcome feedback."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol
from uuid import UUID, NAMESPACE_URL, uuid5

from pydantic import AwareDatetime, Field, model_validator

from adaptive_agent_runtime.context_memory.json_types import (
    ContextMemoryModel,
    utc_now,
)
from adaptive_agent_runtime.governance.models import (
    GovernanceTarget,
    RuntimeCommitPermit,
)


DECISION_FEEDBACK_DECISION_TYPE = "decision.outcome_feedback"
DECISION_FEEDBACK_COMMIT_OPERATION = "decision.feedback.commit"


class FeedbackRuntimeOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class DecisionFeedbackAttributionType(StrEnum):
    ASSOCIATED = "associated"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DecisionFeedbackSubjectReference(ContextMemoryModel):
    decision_id: UUID
    decision_type: str = Field(min_length=1)
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class DecisionFeedbackEvaluationReference(ContextMemoryModel):
    report_id: UUID
    trace_id: UUID
    run_id: UUID
    task_id: UUID
    outcome_evaluation_id: UUID
    outcome_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    verdict: str = Field(min_length=1)
    score: float | None = Field(default=None, ge=0.0, le=1.0)


class DecisionFeedbackArtifactReference(ContextMemoryModel):
    node_id: UUID
    artifact_type: str = Field(min_length=1)
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_decision_id: UUID


class DecisionFeedbackExperienceReference(ContextMemoryModel):
    experience_id: UUID
    version: int = Field(ge=1)
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class DecisionFeedbackRecallReference(ContextMemoryModel):
    recall_decision_id: UUID
    recall_effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_id: UUID
    bundle_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    planning_decision_id: UUID
    planning_effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class DecisionFeedbackRuntimeObservation(ContextMemoryModel):
    state_revision: int = Field(ge=0)
    state_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_status: str = Field(min_length=1)
    failure_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    artifact_committed: bool


class DecisionFeedbackRequest(ContextMemoryModel):
    source_run_id: UUID
    source_task_id: UUID
    subject: DecisionFeedbackSubjectReference
    planning_subject: DecisionFeedbackSubjectReference
    evaluation: DecisionFeedbackEvaluationReference
    experience: DecisionFeedbackExperienceReference
    artifact: DecisionFeedbackArtifactReference
    runtime_observation: DecisionFeedbackRuntimeObservation
    runtime_outcome: FeedbackRuntimeOutcome
    recall: DecisionFeedbackRecallReference | None = None

    @model_validator(mode="after")
    def validate_subject(self) -> DecisionFeedbackRequest:
        if self.subject.decision_type == "memory.recall" and self.recall is None:
            raise ValueError("Recall Feedback requires a committed Recall binding")
        if self.subject.decision_type != "memory.recall" and self.recall is not None:
            raise ValueError("Only Recall Feedback may carry a Recall binding")
        if self.evaluation.run_id != self.source_run_id:
            raise ValueError("Feedback Evaluation belongs to another run")
        if self.evaluation.task_id != self.source_task_id:
            raise ValueError("Feedback Evaluation belongs to another task")
        return self


class DecisionFeedbackDraft(ContextMemoryModel):
    """Deterministic association proposal; it has no mutation authority."""

    attribution_type: DecisionFeedbackAttributionType
    summary: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)


class DecisionFeedbackEffect(ContextMemoryModel):
    feedback_id: UUID
    source_run_id: UUID
    source_task_id: UUID
    subject: DecisionFeedbackSubjectReference
    planning_subject: DecisionFeedbackSubjectReference
    evaluation: DecisionFeedbackEvaluationReference
    experience: DecisionFeedbackExperienceReference
    artifact: DecisionFeedbackArtifactReference
    runtime_observation: DecisionFeedbackRuntimeObservation
    runtime_outcome: FeedbackRuntimeOutcome
    evaluation_verdict: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    attribution_type: DecisionFeedbackAttributionType
    recall: DecisionFeedbackRecallReference | None = None
    proposal_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class DecisionFeedbackRecord(ContextMemoryModel):
    feedback_id: UUID
    version: int = Field(ge=1)
    source_run_id: UUID
    source_task_id: UUID
    subject_decision_id: UUID
    subject_decision_type: str = Field(min_length=1)
    subject_effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    planning_decision_id: UUID
    planning_effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_ref: DecisionFeedbackEvaluationReference
    experience_metadata_ref: DecisionFeedbackExperienceReference
    artifact_ref: DecisionFeedbackArtifactReference
    runtime_outcome: FeedbackRuntimeOutcome
    evaluation_verdict: str = Field(min_length=1)
    failure_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    attribution_type: DecisionFeedbackAttributionType
    recall_ref: DecisionFeedbackRecallReference | None = None
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: AwareDatetime = Field(default_factory=utc_now)


class DecisionFeedbackCommitReceipt(ContextMemoryModel):
    feedback_id: UUID
    version: int = Field(ge=1)
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    committed_at: AwareDatetime = Field(default_factory=utc_now)


class DecisionFeedbackStore(Protocol):
    async def find_applied_subject(
        self, run_id: UUID, decision_type: str
    ) -> DecisionFeedbackSubjectReference | None: ...

    async def commit(
        self,
        effect: DecisionFeedbackEffect,
        *,
        effect_fingerprint: str,
        permit: RuntimeCommitPermit | None,
        target: GovernanceTarget | None,
        subject_fingerprint: str | None,
    ) -> DecisionFeedbackRecord: ...

    async def load_by_effect(
        self, effect_fingerprint: str
    ) -> DecisionFeedbackRecord | None: ...

    async def load_receipt(
        self, effect_fingerprint: str
    ) -> DecisionFeedbackCommitReceipt | None: ...

    async def list_for_run(
        self, run_id: UUID
    ) -> tuple[DecisionFeedbackRecord, ...]: ...


def stable_feedback_request_id(run_id: UUID, subject_decision_id: UUID) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        f"adaptive-agent-runtime:decision-feedback:{run_id}:{subject_decision_id}",
    )


def stable_feedback_id(request_id: UUID) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        f"adaptive-agent-runtime:decision-feedback-record:{request_id}",
    )

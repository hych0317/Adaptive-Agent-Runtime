"""Strongly typed contracts for governed, cross-run learning observations."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID, NAMESPACE_URL, uuid5

from pydantic import AwareDatetime, Field, model_validator

from adaptive_agent_runtime.context_memory import MemoryScope
from adaptive_agent_runtime.context_memory.json_types import (
    ContextMemoryModel,
    ImmutableJsonObject,
    utc_now,
)
from adaptive_agent_runtime.decision_feedback import (
    DecisionFeedbackArtifactReference,
    DecisionFeedbackAttributionType,
    DecisionFeedbackEvaluationReference,
    DecisionFeedbackExperienceReference,
    FeedbackRuntimeOutcome,
)
from adaptive_agent_runtime.governance.models import (
    GovernanceTarget,
    RuntimeCommitPermit,
)


EXPERIENCE_LEARNING_DECISION_TYPE = "experience.learning.assessment"
LEARNING_INSIGHT_COMMIT_OPERATION = "learning.insight.commit"


class LearningEvidenceCandidate(ContextMemoryModel):
    """Runtime-only verified evidence; never projected wholesale to an Agent."""

    candidate_ref: str = Field(pattern=r"^evidence-[0-9a-f]{32}$")
    scope: MemoryScope
    subject_decision_type: str = Field(min_length=1)
    source_run_id: UUID
    source_task_id: UUID
    subject_decision_id: UUID
    subject_effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    feedback_id: UUID
    feedback_effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    feedback_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    attribution_type: DecisionFeedbackAttributionType
    experience: DecisionFeedbackExperienceReference
    evaluation: DecisionFeedbackEvaluationReference
    artifact: DecisionFeedbackArtifactReference
    runtime_outcome: FeedbackRuntimeOutcome
    evaluation_verdict: str = Field(min_length=1)
    runtime_observation: ImmutableJsonObject
    experience_summary: ImmutableJsonObject
    candidate_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class LearningEvidenceAgentView(ContextMemoryModel):
    """Opaque, sanitized evidence projection available to the Learning Agent."""

    candidate_ref: str = Field(pattern=r"^evidence-[0-9a-f]{32}$")
    outcome: FeedbackRuntimeOutcome
    evaluation_verdict: str = Field(min_length=1)
    observed_pattern: str = Field(min_length=1)
    possible_relevance: str = Field(min_length=1)
    limitation: str = Field(min_length=1)
    conclusion_eligible: bool


class LearningAssessmentRequest(ContextMemoryModel):
    """Runtime request retaining authoritative evidence bindings."""

    scope: MemoryScope
    subject_decision_type: str = Field(min_length=1)
    trigger_run_id: UUID
    candidates: tuple[LearningEvidenceCandidate, ...] = Field(min_length=2)
    minimum_independent_runs: int = Field(default=2, ge=2)
    evidence_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_candidates(self) -> LearningAssessmentRequest:
        refs = tuple(item.candidate_ref for item in self.candidates)
        if len(set(refs)) != len(refs):
            raise ValueError("Learning evidence candidate refs must be unique")
        if any(item.scope != self.scope for item in self.candidates):
            raise ValueError("Learning evidence crosses Runtime scope")
        if any(
            item.subject_decision_type != self.subject_decision_type
            for item in self.candidates
        ):
            raise ValueError("Learning evidence mixes Decision domains")
        eligible_runs = {
            item.source_run_id
            for item in self.candidates
            if item.attribution_type is DecisionFeedbackAttributionType.ASSOCIATED
        }
        if len(eligible_runs) < self.minimum_independent_runs:
            raise ValueError("Learning requires multiple independent associated runs")
        return self


class LearningAssessmentAgentRequest(ContextMemoryModel):
    """Isolated Agent input with no Runtime, database, or governance identity."""

    subject_decision_type: str = Field(min_length=1)
    evidence: tuple[LearningEvidenceAgentView, ...] = Field(min_length=2)
    minimum_independent_runs: int = Field(default=2, ge=2)


class LearningInsightDraft(ContextMemoryModel):
    """Authority-free semantic observation proposed by a Learning Agent."""

    observed_pattern: str = Field(min_length=1)
    applicable_conditions: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    supporting_candidate_refs: tuple[str, ...] = Field(min_length=1)
    counterevidence_candidate_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_refs(self) -> LearningInsightDraft:
        supporting = set(self.supporting_candidate_refs)
        counter = set(self.counterevidence_candidate_refs)
        if len(supporting) != len(self.supporting_candidate_refs):
            raise ValueError("supporting evidence refs must be unique")
        if len(counter) != len(self.counterevidence_candidate_refs):
            raise ValueError("counterevidence refs must be unique")
        if supporting & counter:
            raise ValueError("evidence cannot be both supporting and counterevidence")
        return self


class LearningEvidenceBinding(ContextMemoryModel):
    """Runtime-normalized evidence provenance never authored by the Agent."""

    candidate_ref: str = Field(pattern=r"^evidence-[0-9a-f]{32}$")
    candidate_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_run_id: UUID
    feedback_id: UUID
    feedback_effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    experience_id: UUID
    experience_effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_report_id: UUID
    evaluation_report_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_outcome: FeedbackRuntimeOutcome
    attribution_type: DecisionFeedbackAttributionType


class LearningInsightEffect(ContextMemoryModel):
    learning_insight_id: UUID
    scope: MemoryScope
    subject_decision_type: str = Field(min_length=1)
    observed_pattern: str = Field(min_length=1)
    applicable_conditions: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    evidence_snapshot: tuple[LearningEvidenceBinding, ...] = Field(min_length=2)
    supporting_evidence: tuple[LearningEvidenceBinding, ...] = Field(min_length=1)
    counterevidence: tuple[LearningEvidenceBinding, ...] = ()
    source_run_refs: tuple[UUID, ...] = Field(min_length=2)
    source_feedback_refs: tuple[UUID, ...] = Field(min_length=2)
    source_experience_refs: tuple[UUID, ...] = Field(min_length=2)
    evidence_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class LearningInsight(ContextMemoryModel):
    learning_insight_id: UUID
    version: int = Field(ge=1)
    scope: MemoryScope
    subject_decision_type: str = Field(min_length=1)
    observed_pattern: str = Field(min_length=1)
    applicable_conditions: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    supporting_evidence_refs: tuple[str, ...] = Field(min_length=1)
    counterevidence_refs: tuple[str, ...] = ()
    source_run_refs: tuple[UUID, ...] = Field(min_length=2)
    source_feedback_refs: tuple[UUID, ...] = Field(min_length=2)
    source_experience_refs: tuple[UUID, ...] = Field(min_length=2)
    evidence_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: AwareDatetime = Field(default_factory=utc_now)


class LearningInsightCommitReceipt(ContextMemoryModel):
    learning_insight_id: UUID
    version: int = Field(ge=1)
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    insight_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    committed_at: AwareDatetime = Field(default_factory=utc_now)


class LearningInsightStore(Protocol):
    async def resolve_candidates(
        self,
        *,
        scope: MemoryScope,
        subject_decision_type: str,
    ) -> tuple[LearningEvidenceCandidate, ...]: ...

    async def commit(
        self,
        effect: LearningInsightEffect,
        *,
        effect_fingerprint: str,
        permit: RuntimeCommitPermit | None,
        target: GovernanceTarget | None,
        subject_fingerprint: str | None,
    ) -> LearningInsight: ...

    async def load_by_effect(self, effect_fingerprint: str) -> LearningInsight | None: ...

    async def load_receipt(
        self, effect_fingerprint: str
    ) -> LearningInsightCommitReceipt | None: ...

    async def list_for_scope(
        self,
        scope: MemoryScope,
        *,
        subject_decision_type: str | None = None,
    ) -> tuple[LearningInsight, ...]: ...


def stable_learning_request_id(
    trigger_run_id: UUID,
    scope: MemoryScope,
    subject_decision_type: str,
) -> UUID:
    scope_key = f"{scope.tenant_id}:{scope.project_id}:{scope.agent_scope}"
    return uuid5(
        NAMESPACE_URL,
        "adaptive-agent-runtime:learning-assessment:"
        f"{trigger_run_id}:{scope_key}:{subject_decision_type}",
    )


def stable_learning_insight_id(request_id: UUID) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        f"adaptive-agent-runtime:learning-insight:{request_id}",
    )

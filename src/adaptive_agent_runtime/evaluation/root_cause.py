"""Authority-free semantic Root Cause decision contracts.

The deterministic Evaluation kernel remains authoritative for metrics, verdicts,
and findings.  These models describe an additional, evidence-bound semantic
assessment that may inform later Runtime decisions but cannot execute them.
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from adaptive_agent_runtime.evaluation.models import (
    EvaluationComponent,
    EvaluationCorrelation,
    EvaluationModel,
    FindingSeverity,
    TraceCompleteness,
    stable_evaluation_id,
    utc_now,
)


ROOT_CAUSE_DECISION_TYPE = "evaluation.root_cause"
ROOT_CAUSE_RECORD_OPERATION = "evaluation.root_cause.record"


class RootCauseTrigger(StrEnum):
    INLINE_FAILURE = "inline_failure"
    POST_RUN_EVALUATION = "post_run_evaluation"


class RootCauseConclusion(StrEnum):
    SUPPORTED = "supported"
    INCONCLUSIVE = "inconclusive"


class RootCauseEvidenceStrength(StrEnum):
    INSUFFICIENT = "insufficient"
    LIMITED = "limited"
    CORROBORATED = "corroborated"


class RootCauseDeterministicAlignment(StrEnum):
    NOT_AVAILABLE = "not_available"
    ALIGNED = "aligned"
    PARTIAL = "partial"
    UNVERIFIED = "unverified"


class RootCauseExecutionPolicy(EvaluationModel):
    """Runtime-owned bound for one semantic diagnostic call."""

    timeout_seconds: float = Field(default=20.0, gt=0.0)
    max_agent_calls: int = Field(default=1, ge=1, le=1)
    max_evidence_items: int = Field(default=12, ge=1, le=64)


class RootCauseEvidenceBinding(EvaluationModel):
    """Runtime-only provenance for an Agent-visible evidence reference."""

    evidence_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    source_record_id: str = Field(min_length=1)
    correlation: EvaluationCorrelation
    reliability: float = Field(ge=0.0, le=1.0)
    direct_failure: bool = False


class DeterministicFindingSnapshot(EvaluationModel):
    """Immutable projection of a deterministic finding, without evaluator identity."""

    finding_id: UUID
    evaluation_id: UUID
    component: EvaluationComponent
    code: str = Field(min_length=1)
    severity: FindingSeverity
    summary: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_evidence(self) -> DeterministicFindingSnapshot:
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("deterministic finding evidence ids must be unique")
        return self


class RootCauseDecisionPayload(EvaluationModel):
    """Full Runtime snapshot retained outside isolated Agent context."""

    trigger: RootCauseTrigger
    failure_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    failure_summary: str = Field(min_length=1)
    correlation: EvaluationCorrelation
    trace_id: UUID | None = None
    trace_completeness: TraceCompleteness
    evidence_bindings: tuple[RootCauseEvidenceBinding, ...] = Field(min_length=1)
    deterministic_findings: tuple[DeterministicFindingSnapshot, ...] = ()
    execution_policy: RootCauseExecutionPolicy = Field(
        default_factory=RootCauseExecutionPolicy
    )

    @model_validator(mode="after")
    def validate_snapshot(self) -> RootCauseDecisionPayload:
        if self.correlation.task_id is None:
            raise ValueError("Root Cause decision requires a task correlation")
        evidence_ids = tuple(item.evidence_id for item in self.evidence_bindings)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("Root Cause evidence bindings must be unique")
        if len(self.evidence_bindings) > self.execution_policy.max_evidence_items:
            raise ValueError("Root Cause evidence exceeds the Runtime item budget")
        for item in self.evidence_bindings:
            if item.correlation.run_id != self.correlation.run_id:
                raise ValueError("Root Cause evidence belongs to another run")
            if item.correlation.task_id not in {None, self.correlation.task_id}:
                raise ValueError("Root Cause evidence belongs to another task")
            if (
                self.correlation.node_id is not None
                and item.correlation.node_id not in {None, self.correlation.node_id}
            ):
                raise ValueError("Root Cause evidence belongs to another node")
            if (
                self.correlation.action_id is not None
                and item.correlation.action_id not in {None, self.correlation.action_id}
            ):
                raise ValueError("Root Cause evidence belongs to another action")
        known = set(evidence_ids)
        finding_ids = tuple(item.finding_id for item in self.deterministic_findings)
        if len(set(finding_ids)) != len(finding_ids):
            raise ValueError("deterministic finding snapshots must be unique")
        for finding in self.deterministic_findings:
            if not set(finding.evidence_ids).issubset(known):
                raise ValueError(
                    "deterministic finding references evidence outside the snapshot"
                )
        if self.trigger is RootCauseTrigger.INLINE_FAILURE:
            if self.correlation.node_id is None or self.correlation.action_id is None:
                raise ValueError("inline Root Cause requires node and action identity")
            if not any(item.direct_failure for item in self.evidence_bindings):
                raise ValueError("inline Root Cause requires direct failure evidence")
        elif self.trace_id is None or not self.deterministic_findings:
            raise ValueError(
                "post-run Root Cause requires a Trace and deterministic findings"
            )
        return self


class RootCauseAlternative(EvaluationModel):
    code: str = Field(min_length=1)
    description: str = Field(min_length=1)
    supporting_evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_evidence(self) -> RootCauseAlternative:
        if len(set(self.supporting_evidence_ids)) != len(
            self.supporting_evidence_ids
        ):
            raise ValueError("Root Cause alternative evidence ids must be unique")
        return self


class RootCauseAssessment(EvaluationModel):
    """Applied advisory result; it has no Recovery or execution authority."""

    assessment_id: UUID
    request_id: UUID
    proposal_id: UUID
    correlation: EvaluationCorrelation
    trigger: RootCauseTrigger
    failure_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    conclusion: RootCauseConclusion
    primary_code: str | None = Field(default=None, min_length=1)
    primary_description: str | None = Field(default=None, min_length=1)
    alternatives: tuple[RootCauseAlternative, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    counter_evidence_ids: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)
    stated_confidence: float = Field(ge=0.0, le=1.0)
    evidence_strength: RootCauseEvidenceStrength
    deterministic_alignment: RootCauseDeterministicAlignment
    trace_completeness: TraceCompleteness
    deterministic_finding_codes: tuple[str, ...] = ()
    assessed_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_assessment(self) -> RootCauseAssessment:
        if self.conclusion is RootCauseConclusion.SUPPORTED:
            if not self.primary_code or not self.primary_description:
                raise ValueError("supported Root Cause requires a primary hypothesis")
            if not self.supporting_evidence_ids:
                raise ValueError("supported Root Cause requires verified evidence")
            if self.evidence_strength is RootCauseEvidenceStrength.INSUFFICIENT:
                raise ValueError("supported Root Cause cannot have insufficient evidence")
        else:
            if self.primary_code is not None or self.primary_description is not None:
                raise ValueError("inconclusive Root Cause cannot assert a primary cause")
            if self.evidence_strength is not RootCauseEvidenceStrength.INSUFFICIENT:
                raise ValueError("inconclusive Root Cause must remain insufficient")
        if set(self.supporting_evidence_ids).intersection(self.counter_evidence_ids):
            raise ValueError("evidence cannot both support and counter a Root Cause")
        unique_sets = (
            self.supporting_evidence_ids,
            self.counter_evidence_ids,
            self.assumptions,
            self.unresolved_questions,
            self.deterministic_finding_codes,
        )
        if any(len(set(items)) != len(items) for items in unique_sets):
            raise ValueError("Root Cause assessment collections must be unique")
        alternative_codes = tuple(item.code for item in self.alternatives)
        if len(set(alternative_codes)) != len(alternative_codes):
            raise ValueError("Root Cause alternative codes must be unique")
        if self.primary_code in set(alternative_codes):
            raise ValueError("primary Root Cause cannot also be an alternative")
        return self


class RootCauseAssessmentEffect(EvaluationModel):
    """Runtime-normalized append-only diagnostic record."""

    assessment: RootCauseAssessment
    source_draft_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


def root_cause_failure_signature(
    failure_summary: str,
    *,
    component: str = "runtime",
) -> str:
    """Build a stable, non-secret signature for matching equivalent failures."""

    normalized = re.sub(r"\s+", " ", failure_summary.strip().lower())
    return hashlib.sha256(f"{component}|{normalized}".encode("utf-8")).hexdigest()


def stable_root_cause_id(*parts: object) -> UUID:
    return stable_evaluation_id("semantic-root-cause", *parts)

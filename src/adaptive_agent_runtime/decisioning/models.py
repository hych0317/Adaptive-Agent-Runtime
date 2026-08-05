"""Immutable contracts for the Runtime-owned decision lifecycle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum, StrEnum
from math import isfinite
from typing import Any, Generic, Mapping, Self, TypeVar
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, Field, model_validator

from adaptive_agent_runtime.core.models import (
    FrozenModel,
    ImmutableJsonObject,
    ImmutableJsonValue,
    utc_now,
)


RequestPayloadT = TypeVar("RequestPayloadT", bound=BaseModel)
ProposalPayloadT = TypeVar("ProposalPayloadT", bound=BaseModel)
EffectPayloadT = TypeVar("EffectPayloadT", bound=BaseModel)


class DecisionModel(FrozenModel):
    """Frozen model whose copies are revalidated instead of blindly updated."""

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        del deep
        values = self.model_dump(mode="python")
        if update:
            values.update(update)
        return self.__class__.model_validate(values)


def decision_fingerprint(value: object) -> str:
    """Return a stable SHA-256 fingerprint for a decision snapshot."""

    canonical = json.dumps(
        _fingerprint_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fingerprint_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _fingerprint_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("fingerprinted object keys must be strings")
        return {
            key: _fingerprint_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return _fingerprint_value(value.value)
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("fingerprinted numbers must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported decision fingerprint type: {type(value).__name__}")


class DecisionValidationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class DecisionResultStatus(StrEnum):
    APPLIED = "applied"
    REJECTED = "rejected"
    FAILED = "failed"
    EXPIRED = "expired"


class DecisionGovernanceOutcome(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REVIEW_REQUIRED = "review_required"


class DecisionGovernanceScope(StrEnum):
    ACTION = "action"
    STATE = "state"
    EVOLUTION = "evolution"


class DecisionRiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DecisionCheckpointStage(StrEnum):
    REQUESTED = "requested"
    CONTEXT_PROJECTED = "context_projected"
    PROPOSED = "proposed"
    VALIDATED = "validated"
    REVIEW_PENDING = "review_pending"
    AUTHORIZED = "authorized"
    APPLYING = "applying"
    COMPLETED = "completed"


class DecisionReconciliationStatus(StrEnum):
    """Authoritative state observed while recovering an interrupted Apply."""

    NOT_COMMITTED = "not_committed"
    COMMITTED = "committed"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class DecisionTraceKind(StrEnum):
    REQUESTED = "decision.requested"
    CONTEXT_PROJECTED = "decision.context_projected"
    PROPOSED = "decision.proposed"
    VALIDATION_PASSED = "decision.validation_passed"
    VALIDATION_FAILED = "decision.validation_failed"
    GOVERNANCE_REQUESTED = "decision.governance_requested"
    REVIEW_REQUIRED = "decision.review_required"
    AUTHORIZED = "decision.authorized"
    DENIED = "decision.denied"
    APPLY_STARTED = "decision.apply_started"
    APPLIED = "decision.applied"
    FAILED = "decision.failed"
    EXPIRED = "decision.expired"


class DecisionTarget(DecisionModel):
    target_type: str = Field(min_length=1)
    target_id: str = Field(min_length=1)


class DecisionCorrelation(DecisionModel):
    run_id: UUID
    task_id: UUID | None = None
    node_id: UUID | None = None
    action_id: UUID | None = None
    decision_cycle: int = Field(default=0, ge=0)


class DecisionBasis(DecisionModel):
    snapshot_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_revision: int | None = Field(default=None, ge=0)
    graph_version: int | None = Field(default=None, ge=0)
    configuration_revision: int | None = Field(default=None, ge=0)


class DecisionConstraint(DecisionModel):
    constraint_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    value: ImmutableJsonValue = None


class DecisionEvidenceReference(DecisionModel):
    evidence_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    source: str = Field(min_length=1)
    reliability: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1)


class DecisionBudget(DecisionModel):
    """Runtime-enforced limits for one bounded decision sequence."""

    max_decision_cycles: int = Field(default=1, ge=0)
    max_agent_calls: int = Field(default=1, ge=0)
    max_revision_count: int = Field(default=0, ge=0)
    max_elapsed_seconds: float | None = Field(default=None, gt=0.0)
    max_cost_units: float | None = Field(default=None, ge=0.0)


class DecisionBudgetUsage(DecisionModel):
    decision_cycles: int = Field(default=0, ge=0)
    agent_calls: int = Field(default=0, ge=0)
    revision_count: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    consumed_cost_units: float = Field(default=0.0, ge=0.0)


class DecisionRequest(DecisionModel, Generic[RequestPayloadT]):
    """A Runtime-created request for one bounded semantic decision."""

    request_id: UUID = Field(default_factory=uuid4)
    decision_type: str = Field(min_length=1)
    schema_version: str = Field(default="1", min_length=1)
    target: DecisionTarget
    correlation: DecisionCorrelation
    basis: DecisionBasis
    payload: RequestPayloadT
    allowed_actions: tuple[str, ...] = Field(min_length=1)
    constraints: tuple[DecisionConstraint, ...] = ()
    evidence: tuple[DecisionEvidenceReference, ...] = ()
    budget: DecisionBudget = Field(default_factory=DecisionBudget)
    created_at: AwareDatetime = Field(default_factory=utc_now)
    expires_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_request(self) -> DecisionRequest[RequestPayloadT]:
        if len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise ValueError("allowed decision actions must be unique")
        if any(not action for action in self.allowed_actions):
            raise ValueError("allowed decision actions cannot be empty")
        constraint_ids = tuple(item.constraint_id for item in self.constraints)
        if len(set(constraint_ids)) != len(constraint_ids):
            raise ValueError("decision constraint ids must be unique")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("decision evidence ids must be unique")
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("decision request expiry must follow creation")
        return self


class DecisionProducer(DecisionModel):
    producer_id: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    inference_request_id: UUID | None = None
    model_id: str | None = None
    implementation_version: str | None = None


class DecisionProposal(DecisionModel, Generic[ProposalPayloadT]):
    """An Agent proposal with no authorization or execution authority."""

    proposal_id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    proposal_type: str = Field(min_length=1)
    schema_version: str = Field(default="1", min_length=1)
    producer: DecisionProducer
    input_snapshot_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    revision: int = Field(default=0, ge=0)
    selected_action: str = Field(min_length=1)
    payload: ProposalPayloadT
    rationale: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_evidence_refs(self) -> DecisionProposal[ProposalPayloadT]:
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("proposal evidence references must be unique")
        if any(not item for item in self.evidence_refs):
            raise ValueError("proposal evidence references cannot be empty")
        return self


class AgentCallResult(DecisionModel, Generic[ProposalPayloadT]):
    """Runtime-observed accounting around one Agent proposal call."""

    proposal: DecisionProposal[ProposalPayloadT]
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    cost_units: float = Field(default=0.0, ge=0.0)


class DecisionValidationCheck(DecisionModel):
    name: str = Field(min_length=1)
    passed: bool
    detail: str = Field(min_length=1)


class DecisionValidationViolation(DecisionModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class DecisionValidation(DecisionModel):
    validation_id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    proposal_id: UUID
    status: DecisionValidationStatus
    checks: tuple[DecisionValidationCheck, ...] = ()
    violations: tuple[DecisionValidationViolation, ...] = ()
    validated_basis: DecisionBasis
    normalized_effect_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_outcome(self) -> DecisionValidation:
        failed_checks = any(not item.passed for item in self.checks)
        if self.status is DecisionValidationStatus.PASSED:
            if failed_checks or self.violations:
                raise ValueError("passed validation cannot contain failures")
            if self.normalized_effect_fingerprint is None:
                raise ValueError("passed validation requires an effect fingerprint")
        else:
            if not failed_checks and not self.violations:
                raise ValueError("failed validation requires a failed check or violation")
            if self.normalized_effect_fingerprint is not None:
                raise ValueError("failed validation cannot approve an effect")
        return self


class NormalizedDecisionEffect(DecisionModel, Generic[EffectPayloadT]):
    """Runtime-normalized effect that Governance evaluates and Apply commits."""

    payload: EffectPayloadT
    operation: str = Field(min_length=1)
    target: DecisionTarget
    governance_scope: DecisionGovernanceScope
    risk: DecisionRiskLevel
    impact_score: float = Field(ge=0.0, le=1.0)
    reversible: bool
    impact_description: str = Field(min_length=1)
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_fingerprint(self) -> NormalizedDecisionEffect[EffectPayloadT]:
        expected = decision_fingerprint(_normalized_effect_subject(self))
        if self.effect_fingerprint != expected:
            raise ValueError("normalized effect fingerprint does not match payload")
        return self

    @classmethod
    def create(
        cls,
        *,
        payload: EffectPayloadT,
        operation: str,
        target: DecisionTarget,
        governance_scope: DecisionGovernanceScope,
        risk: DecisionRiskLevel,
        impact_score: float,
        reversible: bool,
        impact_description: str,
    ) -> NormalizedDecisionEffect[EffectPayloadT]:
        values: dict[str, object] = {
            "payload": payload,
            "operation": operation,
            "target": target,
            "governance_scope": governance_scope,
            "risk": risk,
            "impact_score": impact_score,
            "reversible": reversible,
            "impact_description": impact_description,
        }
        return cls(
            payload=payload,
            operation=operation,
            target=target,
            governance_scope=governance_scope,
            risk=risk,
            impact_score=impact_score,
            reversible=reversible,
            impact_description=impact_description,
            effect_fingerprint=decision_fingerprint(values),
        )


def _normalized_effect_subject(
    effect: NormalizedDecisionEffect[Any],
) -> dict[str, object]:
    return {
        "payload": effect.payload,
        "operation": effect.operation,
        "target": effect.target,
        "governance_scope": effect.governance_scope,
        "risk": effect.risk,
        "impact_score": effect.impact_score,
        "reversible": effect.reversible,
        "impact_description": effect.impact_description,
    }


class DecisionValidationOutcome(DecisionModel, Generic[EffectPayloadT]):
    validation: DecisionValidation
    normalized_effect: NormalizedDecisionEffect[EffectPayloadT] | None = None

    @model_validator(mode="after")
    def validate_effect(self) -> DecisionValidationOutcome[EffectPayloadT]:
        passed = self.validation.status is DecisionValidationStatus.PASSED
        if passed != (self.normalized_effect is not None):
            raise ValueError("only passed validation can contain a normalized effect")
        if (
            self.normalized_effect is not None
            and self.validation.normalized_effect_fingerprint
            != self.normalized_effect.effect_fingerprint
        ):
            raise ValueError("validation outcome effect fingerprints differ")
        return self


class ValidatedDecision(
    DecisionModel,
    Generic[RequestPayloadT, ProposalPayloadT, EffectPayloadT],
):
    request: DecisionRequest[RequestPayloadT]
    proposal: DecisionProposal[ProposalPayloadT]
    validation: DecisionValidation
    normalized_effect: NormalizedDecisionEffect[EffectPayloadT]

    @model_validator(mode="after")
    def validate_binding(
        self,
    ) -> ValidatedDecision[RequestPayloadT, ProposalPayloadT, EffectPayloadT]:
        if self.validation.status is not DecisionValidationStatus.PASSED:
            raise ValueError("validated decision requires passed validation")
        if self.request.request_id != self.proposal.request_id:
            raise ValueError("proposal belongs to another decision request")
        if self.validation.request_id != self.request.request_id:
            raise ValueError("validation belongs to another decision request")
        if self.validation.proposal_id != self.proposal.proposal_id:
            raise ValueError("validation belongs to another proposal")
        if (
            self.validation.normalized_effect_fingerprint
            != self.normalized_effect.effect_fingerprint
        ):
            raise ValueError("validation and normalized effect fingerprints differ")
        return self


class DecisionGovernanceReceipt(DecisionModel):
    outcome: DecisionGovernanceOutcome
    governance_request_id: UUID
    governance_decision_id: UUID
    authorization_id: UUID | None = None
    review_request_id: UUID | None = None
    reason: str = Field(min_length=1)
    decided_at: AwareDatetime
    approval_snapshot: ImmutableJsonObject | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> DecisionGovernanceReceipt:
        if self.outcome is DecisionGovernanceOutcome.ALLOW:
            if self.authorization_id is None:
                raise ValueError("allowed decision requires authorization")
        elif self.outcome is DecisionGovernanceOutcome.REVIEW_REQUIRED:
            if self.review_request_id is None or self.authorization_id is not None:
                raise ValueError("review decision requires only a review request")
        elif self.authorization_id is not None:
            raise ValueError("denied decision cannot contain authorization")
        return self


class DecisionApplyReceipt(DecisionModel):
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    committed_state_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    result: ImmutableJsonValue = None
    applied_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_committed_state(self) -> DecisionApplyReceipt:
        if self.committed_state_fingerprint != decision_fingerprint(self.result):
            raise ValueError("Apply receipt does not match committed read-back result")
        return self


class DecisionReconciliation(DecisionModel):
    status: DecisionReconciliationStatus
    reason: str = Field(min_length=1)
    apply_receipt: DecisionApplyReceipt | None = None

    @model_validator(mode="after")
    def validate_reconciliation(self) -> DecisionReconciliation:
        if self.status is DecisionReconciliationStatus.COMMITTED:
            if self.apply_receipt is None:
                raise ValueError("committed reconciliation requires an Apply receipt")
        elif self.apply_receipt is not None:
            raise ValueError("only committed reconciliation can contain an Apply receipt")
        return self


class DecisionResult(DecisionModel):
    result_id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    proposal_id: UUID | None = None
    status: DecisionResultStatus
    reason: str = Field(min_length=1)
    apply_receipt: DecisionApplyReceipt | None = None
    completed_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_result(self) -> DecisionResult:
        if self.status is DecisionResultStatus.APPLIED:
            if self.proposal_id is None or self.apply_receipt is None:
                raise ValueError("applied result requires proposal and apply receipt")
        elif self.apply_receipt is not None:
            raise ValueError("only an applied result can contain an apply receipt")
        return self


class AgentContextManifest(DecisionModel):
    context_id: UUID
    request_id: UUID
    agent_scope: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    context_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    included_source_ids: tuple[str, ...] = ()
    omitted_source_ids: tuple[str, ...] = ()
    estimated_tokens: int = Field(default=0, ge=0)


class DecisionCheckpoint(
    DecisionModel,
    Generic[RequestPayloadT, ProposalPayloadT, EffectPayloadT],
):
    request_id: UUID
    run_id: UUID
    revision: int = Field(default=0, ge=0)
    stage: DecisionCheckpointStage
    request: DecisionRequest[RequestPayloadT]
    budget_usage: DecisionBudgetUsage = Field(default_factory=DecisionBudgetUsage)
    context_manifest: AgentContextManifest | None = None
    proposal: DecisionProposal[ProposalPayloadT] | None = None
    validation: DecisionValidation | None = None
    validated_decision: (
        ValidatedDecision[RequestPayloadT, ProposalPayloadT, EffectPayloadT] | None
    ) = None
    governance_receipt: DecisionGovernanceReceipt | None = None
    result: DecisionResult | None = None
    updated_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_checkpoint(
        self,
    ) -> DecisionCheckpoint[RequestPayloadT, ProposalPayloadT, EffectPayloadT]:
        if self.request_id != self.request.request_id:
            raise ValueError("checkpoint and request identities differ")
        if self.run_id != self.request.correlation.run_id:
            raise ValueError("checkpoint and request runs differ")
        if self.context_manifest is not None:
            if self.context_manifest.request_id != self.request_id:
                raise ValueError("context manifest belongs to another request")
        if self.proposal is not None and self.proposal.request_id != self.request_id:
            raise ValueError("checkpoint proposal belongs to another request")
        if self.validation is not None:
            if self.validation.request_id != self.request_id:
                raise ValueError("checkpoint validation belongs to another request")
            if self.proposal is None:
                raise ValueError("checkpoint validation requires a proposal")
        if self.stage is DecisionCheckpointStage.COMPLETED and self.result is None:
            raise ValueError("completed checkpoint requires a decision result")
        if self.stage is not DecisionCheckpointStage.COMPLETED and self.result is not None:
            raise ValueError("only a completed checkpoint can contain a result")
        if self.stage is DecisionCheckpointStage.REVIEW_PENDING:
            receipt = self.governance_receipt
            if (
                receipt is None
                or receipt.outcome is not DecisionGovernanceOutcome.REVIEW_REQUIRED
            ):
                raise ValueError("review checkpoint requires a review receipt")
        return self


class DecisionTraceEvent(DecisionModel):
    kind: DecisionTraceKind
    run_id: UUID
    source: str = Field(min_length=1)
    request_id: UUID
    task_id: UUID | None = None
    node_id: UUID | None = None
    action_id: UUID | None = None
    proposal_id: UUID | None = None
    validation_id: UUID | None = None
    governance_decision_id: UUID | None = None
    review_request_id: UUID | None = None
    authorization_id: UUID | None = None
    occurred_at: AwareDatetime = Field(default_factory=utc_now)
    payload: ImmutableJsonObject = Field(default_factory=dict)

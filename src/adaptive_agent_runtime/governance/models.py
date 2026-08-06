"""Immutable, execution-agnostic models for Runtime Governance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum, StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Annotated, Any, Mapping, Self, TypeAlias
from uuid import UUID, uuid4, uuid5

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    PlainSerializer,
    model_validator,
)


_GOVERNANCE_NAMESPACE = UUID("2cf35e75-7ad8-4f90-bfc8-452855fd195b")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stable_governance_id(*parts: object) -> UUID:
    return uuid5(_GOVERNANCE_NAMESPACE, "|".join(str(part) for part in parts))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("JSON numbers must be finite")
    return value


ImmutableJsonObject: TypeAlias = Annotated[
    Mapping[str, JsonValue],
    BeforeValidator(_thaw_json),
    AfterValidator(_freeze_json),
    PlainSerializer(_thaw_json),
]


class GovernanceModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

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


def governance_fingerprint(value: object) -> str:
    """Return a canonical fingerprint for immutable Governance snapshots."""

    serializable = _fingerprint_value(value)
    canonical = json.dumps(
        serializable,
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
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return _fingerprint_value(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"unsupported Governance fingerprint type: {type(value).__name__}"
    )


class GovernanceScope(StrEnum):
    ACTION = "action"
    STATE = "state"
    EVOLUTION = "evolution"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DecisionOutcome(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REVIEW_REQUIRED = "review_required"


class DecisionLevel(StrEnum):
    RULE = "rule"
    CONFIDENCE = "confidence"
    REVIEW = "review"


class RuleEffect(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ReviewOutcome(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


class AuthorizationUseStatus(StrEnum):
    RESERVED = "reserved"
    APPLIED = "applied"
    FAILED = "failed"


class GovernanceCorrelation(GovernanceModel):
    run_id: UUID | None = None
    task_id: UUID | None = None
    node_id: UUID | None = None
    action_id: UUID | None = None


class GovernanceTarget(GovernanceModel):
    target_type: str = Field(min_length=1)
    target_id: str = Field(min_length=1)


class GovernanceEvidence(GovernanceModel):
    evidence_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    source: str = Field(min_length=1)
    reliability: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1)


class GovernanceHistory(GovernanceModel):
    successful_similar: int = Field(default=0, ge=0)
    failed_similar: int = Field(default=0, ge=0)
    prior_denials: int = Field(default=0, ge=0)


class ImpactAssessment(GovernanceModel):
    score: float = Field(ge=0.0, le=1.0)
    reversible: bool
    description: str = Field(min_length=1)


class ConfidenceSignals(GovernanceModel):
    stated_confidence: float = Field(ge=0.0, le=1.0)
    evidence: tuple[GovernanceEvidence, ...] = ()
    impact: ImpactAssessment
    history: GovernanceHistory = Field(default_factory=GovernanceHistory)

    @model_validator(mode="after")
    def validate_evidence(self) -> ConfidenceSignals:
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("governance evidence ids must be unique")
        return self


class GovernanceRequest(GovernanceModel):
    request_id: UUID = Field(default_factory=uuid4)
    scope: GovernanceScope
    operation: str = Field(min_length=1)
    target: GovernanceTarget
    risk: RiskLevel
    signals: ConfidenceSignals
    correlation: GovernanceCorrelation = Field(
        default_factory=GovernanceCorrelation
    )
    attributes: ImmutableJsonObject = Field(default_factory=dict)
    requested_at: AwareDatetime = Field(default_factory=utc_now)


class GovernanceRule(GovernanceModel):
    rule_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    effect: RuleEffect
    scopes: tuple[GovernanceScope, ...] = ()
    operations: tuple[str, ...] = ()
    risk_levels: tuple[RiskLevel, ...] = ()
    priority: int = 0
    enabled: bool = True

    @model_validator(mode="after")
    def validate_selectors(self) -> GovernanceRule:
        selectors = (self.scopes, self.operations, self.risk_levels)
        if any(len(set(items)) != len(items) for items in selectors):
            raise ValueError("governance rule selectors must be unique")
        if any(not operation for operation in self.operations):
            raise ValueError("governance rule operations cannot be empty")
        return self


class ConfidencePolicy(GovernanceModel):
    allow_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    deny_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    min_evidence_for_allow: int = Field(default=2, ge=0)
    max_impact_for_allow: float = Field(default=0.6, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> ConfidencePolicy:
        if self.allow_threshold <= self.deny_threshold:
            raise ValueError("allow threshold must exceed deny threshold")
        return self


class GovernancePolicy(GovernanceModel):
    policy_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    rules: tuple[GovernanceRule, ...] = ()
    confidence: ConfidencePolicy = Field(default_factory=ConfidencePolicy)

    @model_validator(mode="after")
    def validate_rules(self) -> GovernancePolicy:
        rule_ids = tuple(item.rule_id for item in self.rules)
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("governance policy rule ids must be unique")
        return self


class RuleEvaluation(GovernanceModel):
    matched_rule_ids: tuple[str, ...] = ()
    effect: RuleEffect | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_match(self) -> RuleEvaluation:
        if len(set(self.matched_rule_ids)) != len(self.matched_rule_ids):
            raise ValueError("matched governance rule ids must be unique")
        if bool(self.matched_rule_ids) is not (self.effect is not None):
            raise ValueError("rule effect and matched ids must appear together")
        return self


class ConfidenceAssessment(GovernanceModel):
    score: float = Field(ge=0.0, le=1.0)
    outcome: DecisionOutcome
    evidence_score: float = Field(ge=0.0, le=1.0)
    history_score: float = Field(ge=0.0, le=1.0)
    safety_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


class GovernanceDecision(GovernanceModel):
    decision_id: UUID
    request_id: UUID
    request_fingerprint: str = Field(min_length=64, max_length=64)
    policy_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    outcome: DecisionOutcome
    level: DecisionLevel
    reason: str = Field(min_length=1)
    matched_rule_ids: tuple[str, ...] = ()
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    review_request_id: UUID | None = None
    decided_at: AwareDatetime

    @model_validator(mode="after")
    def validate_decision(self) -> GovernanceDecision:
        requires_review = self.outcome is DecisionOutcome.REVIEW_REQUIRED
        should_reference_review = (
            requires_review or self.level is DecisionLevel.REVIEW
        )
        if should_reference_review != (self.review_request_id is not None):
            raise ValueError(
                "review decisions must reference one review request"
            )
        if self.level is DecisionLevel.CONFIDENCE:
            if self.confidence_score is None:
                raise ValueError("confidence decision requires a score")
        elif self.confidence_score is not None:
            raise ValueError("only confidence decisions can contain a score")
        if len(set(self.matched_rule_ids)) != len(self.matched_rule_ids):
            raise ValueError("decision matched rule ids must be unique")
        return self


class HumanReviewDecision(GovernanceModel):
    outcome: ReviewOutcome
    reviewer_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    decided_at: AwareDatetime = Field(default_factory=utc_now)


class ReviewRequest(GovernanceModel):
    review_request_id: UUID
    governance_request_id: UUID
    request_fingerprint: str = Field(min_length=64, max_length=64)
    policy_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    status: ReviewStatus = ReviewStatus.PENDING
    reason: str = Field(min_length=1)
    revision: int = Field(default=0, ge=0)
    requested_at: AwareDatetime
    updated_at: AwareDatetime
    decision: HumanReviewDecision | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> ReviewRequest:
        if self.updated_at < self.requested_at:
            raise ValueError("review update cannot precede its request")
        if self.status is ReviewStatus.PENDING:
            if self.decision is not None:
                raise ValueError("pending review cannot contain a decision")
        else:
            if self.decision is None:
                raise ValueError("resolved review requires a human decision")
            expected = (
                ReviewStatus.APPROVED
                if self.decision.outcome is ReviewOutcome.APPROVE
                else ReviewStatus.REJECTED
            )
            if self.status is not expected:
                raise ValueError("review status conflicts with human decision")
        return self


class GovernanceAuthorization(GovernanceModel):
    authorization_id: UUID
    request_id: UUID
    request_fingerprint: str = Field(min_length=64, max_length=64)
    decision_id: UUID
    policy_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    target: GovernanceTarget
    issued_at: AwareDatetime
    integrity_seal: str | None = Field(default=None, min_length=64, max_length=64)


class RuntimeCommitPermit(GovernanceModel):
    """Unforgeable, fingerprint-bound capability for one authoritative commit."""

    authorization_id: UUID
    request_id: UUID
    decision_id: UUID
    operation: str = Field(min_length=1)
    target: GovernanceTarget
    subject_fingerprint: str = Field(min_length=64, max_length=64)
    issued_at: AwareDatetime
    integrity_seal: str = Field(min_length=64, max_length=64)


class AuthorizationUse(GovernanceModel):
    """Durable, single-use state of one authorization at its apply point."""

    authorization_id: UUID
    request_id: UUID
    decision_id: UUID
    operation: str = Field(min_length=1)
    target: GovernanceTarget
    subject_fingerprint: str = Field(min_length=64, max_length=64)
    status: AuthorizationUseStatus
    revision: int = Field(default=0, ge=0)
    reserved_at: AwareDatetime
    updated_at: AwareDatetime
    result_fingerprint: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )
    error: str | None = None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> AuthorizationUse:
        if self.updated_at < self.reserved_at:
            raise ValueError("authorization update cannot precede reservation")
        if self.status is AuthorizationUseStatus.RESERVED:
            if self.result_fingerprint is not None or self.error is not None:
                raise ValueError("reserved authorization cannot have a result")
        elif self.status is AuthorizationUseStatus.APPLIED:
            if self.result_fingerprint is None or self.error is not None:
                raise ValueError("applied authorization requires a result")
        elif self.result_fingerprint is not None or not self.error:
            raise ValueError("failed authorization use requires only an error")
        return self

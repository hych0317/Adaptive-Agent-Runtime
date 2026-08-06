"""Small, deterministic contracts for Phase 4-C controlled adaptation."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID, NAMESPACE_URL, uuid5

from pydantic import AwareDatetime, Field, model_validator

from adaptive_agent_runtime.context_memory.json_types import ContextMemoryModel, utc_now
from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.optimization.models import (
    OptimizationRiskClassification,
    OptimizationScope,
    OptimizationTargetKey,
)


class AutoAdaptationStatus(StrEnum):
    SELECTED = "selected"
    APPLIED = "applied"
    SKIPPED = "skipped"
    DENIED = "denied"
    REVIEW_PENDING = "review_pending"
    UNKNOWN = "unknown"
    FAILED = "failed"


class AutoAdaptationSkipReason(StrEnum):
    DISABLED = "disabled"
    RUN_NOT_COMPLETED = "run_not_completed"
    NO_ELIGIBLE_PROPOSAL = "no_eligible_proposal"
    MULTIPLE_ELIGIBLE_PROPOSALS = "multiple_eligible_proposals"
    INCOMPLETE_PHASE3_PROVENANCE = "incomplete_phase3_provenance"
    COUNTEREVIDENCE_PRESENT = "counterevidence_present"
    RISK_NOT_LOW = "risk_not_low"
    CHANGE_NOT_ONE = "change_not_one"
    BASELINE_STALE = "baseline_stale"
    SCOPE_MISMATCH = "scope_mismatch"
    TARGET_NOT_ALLOWED = "target_not_allowed"
    PROPOSAL_INACTIVE = "proposal_inactive"
    PROPOSAL_EXPIRED = "proposal_expired"
    PROPOSAL_ALREADY_APPLIED = "proposal_already_applied"
    GOVERNANCE_DENIED = "governance_denied"
    REVIEW_PENDING = "review_pending"
    RECONCILIATION_UNKNOWN = "reconciliation_unknown"
    APPLY_REJECTED = "apply_rejected"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


class AutoAdaptationPolicy(ContextMemoryModel):
    """Operator-owned fixed policy; no Agent or Proposal can alter it."""

    enabled: bool = False
    policy_id: str = "research.initial_planning.max_nodes.auto_adaptation"
    version: str = "1"
    scope: OptimizationScope
    target_key: OptimizationTargetKey = OptimizationTargetKey.PLANNER_MAX_NODES
    minimum_value: int = 8
    maximum_value: int = 32
    required_delta: int = 1
    required_risk: OptimizationRiskClassification = (
        OptimizationRiskClassification.LOW
    )

    @model_validator(mode="after")
    def keep_phase_4c_policy_fixed(self) -> AutoAdaptationPolicy:
        if (
            self.scope.tenant,
            self.scope.project,
            self.scope.application,
            self.scope.decision_type,
        ) != (
            "default",
            "research",
            "research_agent",
            "planning.task_graph.initialize",
        ):
            raise ValueError("Auto-adaptation scope is fixed for Phase 4-C")
        if self.target_key is not OptimizationTargetKey.PLANNER_MAX_NODES:
            raise ValueError("Auto-adaptation only supports planner.max_nodes")
        if (
            self.minimum_value != 8
            or self.maximum_value != 32
            or self.required_delta != 1
            or self.required_risk is not OptimizationRiskClassification.LOW
        ):
            raise ValueError("Phase 4-C safety bounds cannot be reconfigured")
        return self

    @property
    def fingerprint(self) -> str:
        return decision_fingerprint(self)


class AutoAdaptationTriggerRecord(ContextMemoryModel):
    """Durable claim and terminal outcome for one completed Research Run."""

    attempt_id: UUID
    trigger_run_id: UUID
    policy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_revision: int = Field(ge=0)
    baseline_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: AutoAdaptationStatus
    selected_proposal_id: UUID | None = None
    trigger_identity: UUID | None = None
    apply_request_id: UUID | None = None
    apply_effect_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    active_revision: int | None = Field(default=None, ge=1)
    skip_reason: AutoAdaptationSkipReason | None = None
    candidate_rejection_reasons: tuple[str, ...] = ()
    error_summary: str | None = Field(default=None, max_length=1024)
    outcome_persisted: bool = True
    revision: int = Field(default=1, ge=1)
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> AutoAdaptationTriggerRecord:
        selected = self.status in {
            AutoAdaptationStatus.SELECTED,
            AutoAdaptationStatus.APPLIED,
            AutoAdaptationStatus.DENIED,
            AutoAdaptationStatus.REVIEW_PENDING,
            AutoAdaptationStatus.UNKNOWN,
        }
        if selected and (
            self.selected_proposal_id is None
            or self.trigger_identity is None
            or self.apply_request_id is None
        ):
            raise ValueError("Selected auto-adaptation record lacks trigger identity")
        if self.status is AutoAdaptationStatus.APPLIED and (
            self.apply_effect_fingerprint is None or self.active_revision is None
        ):
            raise ValueError("Applied auto-adaptation record lacks commit provenance")
        if self.status is AutoAdaptationStatus.SKIPPED and self.skip_reason is None:
            raise ValueError("Skipped auto-adaptation record requires a reason")
        if self.status in {
            AutoAdaptationStatus.DENIED,
            AutoAdaptationStatus.REVIEW_PENDING,
            AutoAdaptationStatus.UNKNOWN,
            AutoAdaptationStatus.FAILED,
        } and self.skip_reason is None:
            raise ValueError("Non-applied auto-adaptation outcome requires a reason")
        if self.status is AutoAdaptationStatus.FAILED and (
            self.skip_reason is not AutoAdaptationSkipReason.INFRASTRUCTURE_FAILURE
            or self.error_summary is None
        ):
            raise ValueError(
                "Failed auto-adaptation outcome requires infrastructure diagnostics"
            )
        if not self.outcome_persisted and self.status not in {
            AutoAdaptationStatus.FAILED,
            AutoAdaptationStatus.UNKNOWN,
        }:
            raise ValueError(
                "Only an infrastructure failure or unknown commit may be non-persistent"
            )
        if self.updated_at < self.created_at:
            raise ValueError("Auto-adaptation update precedes creation")
        return self


def stable_auto_adaptation_attempt_id(trigger_run_id: UUID) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        f"adaptive-agent-runtime:auto-adaptation-attempt:{trigger_run_id}",
    )


def stable_auto_adaptation_trigger_identity(
    trigger_run_id: UUID,
    proposal_id: UUID,
    baseline_revision: int,
) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        "adaptive-agent-runtime:auto-adaptation-trigger:"
        f"{trigger_run_id}:{proposal_id}:{baseline_revision}",
    )

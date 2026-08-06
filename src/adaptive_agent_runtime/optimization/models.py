"""Authority-separated contracts for governed Optimization evolution.

Phase 4-A records immutable proposals.  Phase 4-B adds explicit, governed
configuration Apply and Rollback effects without reviving the legacy Evolution
deployment API or exposing an unguarded configuration mutation primitive.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol
from uuid import UUID, NAMESPACE_URL, uuid5

from pydantic import AwareDatetime, Field, model_serializer, model_validator

from adaptive_agent_runtime.context_memory.json_types import (
    ContextMemoryModel,
    ImmutableJsonObject,
    ImmutableJsonValue,
    utc_now,
)
from adaptive_agent_runtime.decisioning import decision_fingerprint
from adaptive_agent_runtime.governance.models import (
    GovernanceTarget,
    RuntimeCommitPermit,
)


OPTIMIZATION_ASSESSMENT_DECISION_TYPE = "optimization.proposal.assessment"
OPTIMIZATION_PROPOSAL_COMMIT_OPERATION = "optimization.proposal.commit"
OPTIMIZATION_APPLY_DECISION_TYPE = "optimization.apply.request"
OPTIMIZATION_APPLY_COMMIT_OPERATION = "optimization.apply.commit"
OPTIMIZATION_ROLLBACK_DECISION_TYPE = "optimization.rollback.request"
OPTIMIZATION_ROLLBACK_COMMIT_OPERATION = "optimization.rollback.commit"


class OptimizationRiskClassification(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OptimizationProposalStatus(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"


class RuntimeConfigurationActivationMode(StrEnum):
    """Authority provenance for one immutable configuration snapshot."""

    DEFAULT_CONFIGURATION = "default_configuration"
    MANUAL_APPLY = "manual_apply"
    POLICY_AUTO = "policy_auto"
    ROLLBACK = "rollback"


class OptimizationTargetType(StrEnum):
    INITIAL_PLANNING_POLICY = "initial_planning_policy"


class OptimizationTargetKey(StrEnum):
    PLANNER_MAX_NODES = "planner.max_nodes"
    PLANNER_MAX_DEPTH = "planner.max_depth"
    PLANNER_MAX_PARALLELISM = "planner.max_parallelism"
    PLANNER_REQUIRE_VALIDATION_NODE = "planner.require_validation_node"
    PLANNER_ALLOWED_STRATEGY_REF = "planner.allowed_strategy_ref"


class OptimizationScope(ContextMemoryModel):
    """Explicit non-global authority scope for one Proposal."""

    tenant: str = Field(min_length=1)
    project: str = Field(min_length=1)
    application: str = Field(min_length=1)
    decision_type: str = Field(min_length=1)

    @model_validator(mode="after")
    def reject_global_scope(self) -> OptimizationScope:
        forbidden = {"*", "all", "global"}
        values = (self.tenant, self.project, self.application, self.decision_type)
        if any(value.strip().lower() in forbidden for value in values):
            raise ValueError("Optimization Proposal scope cannot be global")
        return self


class OptimizationTargetConstraints(ContextMemoryModel):
    value_type: str = Field(min_length=1)
    minimum: int | None = None
    maximum: int | None = None
    allowed_values: tuple[ImmutableJsonValue, ...] = ()

    @model_validator(mode="after")
    def validate_bounds(self) -> OptimizationTargetConstraints:
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("Optimization target bounds are inverted")
        return self


class OptimizationTarget(ContextMemoryModel):
    """Runtime-owned target and observed baseline; never Agent-authored."""

    target_ref: str = Field(pattern=r"^target-[0-9a-f]{32}$")
    target_type: OptimizationTargetType
    target_key: OptimizationTargetKey
    current_value: ImmutableJsonValue
    current_value_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_configuration_revision: int = Field(default=0, ge=0)
    current_configuration_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    configuration_source: str = Field(min_length=1)
    constraints: OptimizationTargetConstraints

    @model_validator(mode="after")
    def validate_baseline_fingerprint(self) -> OptimizationTarget:
        if self.current_value_fingerprint != decision_fingerprint(
            self.current_value
        ):
            raise ValueError("Optimization target baseline fingerprint is invalid")
        return self


class OptimizationTargetAgentView(ContextMemoryModel):
    """Opaque target projection without target key, scope, or current value."""

    target_ref: str = Field(pattern=r"^target-[0-9a-f]{32}$")
    description: str = Field(min_length=1)
    constraints: OptimizationTargetConstraints


class OptimizationEvidenceBinding(ContextMemoryModel):
    """Runtime-only binding to verified Phase 3 authority records."""

    candidate_ref: str = Field(pattern=r"^optimization-evidence-[0-9a-f]{32}$")
    candidate_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope: OptimizationScope
    learning_insight_id: UUID
    learning_insight_version: int = Field(ge=1)
    learning_insight_effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    learning_insight_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    supporting_feedback_refs: tuple[UUID, ...] = Field(min_length=1)
    supporting_experience_refs: tuple[UUID, ...] = Field(min_length=1)
    supporting_evaluation_refs: tuple[UUID, ...] = Field(min_length=1)
    supporting_artifact_effect_fingerprints: tuple[str, ...] = Field(min_length=1)
    counterevidence_feedback_refs: tuple[UUID, ...] = ()
    source_run_refs: tuple[UUID, ...] = Field(min_length=2)
    observed_pattern: str = Field(min_length=1)
    applicable_conditions: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)


class OptimizationEvidenceAgentView(ContextMemoryModel):
    """Sanitized evidence exposed through PolicyAgentContextBuilder only."""

    candidate_ref: str = Field(pattern=r"^optimization-evidence-[0-9a-f]{32}$")
    observed_pattern: str = Field(min_length=1)
    applicable_conditions: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    independent_run_count: int = Field(ge=2)
    has_counterevidence: bool


class OptimizationAssessmentRequest(ContextMemoryModel):
    """Runtime-owned request retaining baseline and evidence authority bindings."""

    scope: OptimizationScope
    trigger_run_id: UUID
    targets: tuple[OptimizationTarget, ...] = Field(min_length=1)
    candidates: tuple[OptimizationEvidenceBinding, ...] = Field(min_length=1)
    evidence_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_request(self) -> OptimizationAssessmentRequest:
        target_refs = tuple(item.target_ref for item in self.targets)
        candidate_refs = tuple(item.candidate_ref for item in self.candidates)
        if len(set(target_refs)) != len(target_refs):
            raise ValueError("Optimization target refs must be unique")
        if len(set(candidate_refs)) != len(candidate_refs):
            raise ValueError("Optimization evidence refs must be unique")
        if any(item.scope != self.scope for item in self.candidates):
            raise ValueError("Optimization evidence crosses authority scope")
        expected = decision_fingerprint(
            tuple(item.candidate_fingerprint for item in self.candidates)
        )
        if expected != self.evidence_set_fingerprint:
            raise ValueError("Optimization evidence-set fingerprint is invalid")
        return self


class OptimizationAssessmentAgentRequest(ContextMemoryModel):
    """Isolated semantic input with no real scope, baseline, or Runtime IDs."""

    targets: tuple[OptimizationTargetAgentView, ...] = Field(min_length=1)
    evidence: tuple[OptimizationEvidenceAgentView, ...] = Field(min_length=1)


class OptimizationProposalDraft(ContextMemoryModel):
    """Authority-free Agent proposal; it cannot request Apply or activation."""

    target_ref: str = Field(pattern=r"^target-[0-9a-f]{32}$")
    proposed_value: ImmutableJsonValue
    expected_impact: str = Field(min_length=1)
    applicable_conditions: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    supporting_candidate_refs: tuple[str, ...] = Field(min_length=1)
    counterevidence_candidate_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_refs(self) -> OptimizationProposalDraft:
        supporting = set(self.supporting_candidate_refs)
        counter = set(self.counterevidence_candidate_refs)
        if len(supporting) != len(self.supporting_candidate_refs):
            raise ValueError("Optimization supporting refs must be unique")
        if len(counter) != len(self.counterevidence_candidate_refs):
            raise ValueError("Optimization counterevidence refs must be unique")
        if supporting & counter:
            raise ValueError(
                "Optimization evidence cannot be both supporting and counterevidence"
            )
        return self


class OptimizationProposalEffect(ContextMemoryModel):
    """Final Runtime-normalized object authorized only for Proposal commit."""

    proposal_id: UUID
    scope: OptimizationScope
    target: OptimizationTarget
    proposed_value: ImmutableJsonValue
    evidence_snapshot: tuple[OptimizationEvidenceBinding, ...] = Field(min_length=1)
    supporting_candidate_refs: tuple[str, ...] = Field(min_length=1)
    counterevidence_candidate_refs: tuple[str, ...] = ()
    expected_impact: str = Field(min_length=1)
    applicable_conditions: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    risk_classification: OptimizationRiskClassification
    rollback_requirements: tuple[str, ...] = Field(min_length=1)
    evidence_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class OptimizationProposal(ContextMemoryModel):
    """Immutable Phase 4-A authority record.  It is not executable."""

    proposal_id: UUID
    source_decision_request_id: UUID
    scope: OptimizationScope
    target_type: OptimizationTargetType
    target_key: OptimizationTargetKey
    current_value: ImmutableJsonValue
    current_value_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_configuration_revision: int = Field(default=0, ge=0)
    current_configuration_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    configuration_source: str = Field(min_length=1)
    proposed_value: ImmutableJsonValue
    supporting_learning_insight_refs: tuple[UUID, ...] = Field(min_length=1)
    supporting_feedback_refs: tuple[UUID, ...] = Field(min_length=1)
    supporting_experience_refs: tuple[UUID, ...] = Field(min_length=1)
    supporting_evaluation_refs: tuple[UUID, ...] = Field(min_length=1)
    counterevidence_refs: tuple[str, ...] = ()
    applicable_conditions: tuple[str, ...] = Field(min_length=1)
    expected_impact: str = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    risk_classification: OptimizationRiskClassification
    rollback_requirements: tuple[str, ...] = Field(min_length=1)
    evidence_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: OptimizationProposalStatus = OptimizationProposalStatus.ACTIVE
    expires_at: AwareDatetime | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_expiry(self) -> OptimizationProposal:
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("Optimization Proposal expiry must follow creation")
        return self


class OptimizationProposalCommitReceipt(ContextMemoryModel):
    proposal_id: UUID
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    committed_at: AwareDatetime = Field(default_factory=utc_now)


class OptimizationApplyRequest(ContextMemoryModel):
    """Manual or policy trigger; it carries no Apply authority."""

    proposal_id: UUID
    requested_by: str = Field(min_length=1)
    proposal_snapshot: OptimizationProposal
    current_configuration: RuntimeConfigurationSnapshot | None = None
    trigger_mode: RuntimeConfigurationActivationMode = (
        RuntimeConfigurationActivationMode.MANUAL_APPLY
    )
    trigger_run_id: UUID | None = None
    auto_adaptation_policy_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    selected_proposal_id: UUID | None = None
    auto_adaptation_trigger_id: UUID | None = None
    candidate_set_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_proposal_binding(self) -> OptimizationApplyRequest:
        if self.proposal_snapshot.proposal_id != self.proposal_id:
            raise ValueError("Apply request proposal identity is inconsistent")
        if self.trigger_mode is RuntimeConfigurationActivationMode.POLICY_AUTO:
            if (
                self.trigger_run_id is None
                or self.auto_adaptation_policy_fingerprint is None
                or self.selected_proposal_id != self.proposal_id
                or self.auto_adaptation_trigger_id is None
                or self.candidate_set_fingerprint is None
                or self.current_configuration is None
            ):
                raise ValueError(
                    "Policy-auto Apply requires run, policy, selected Proposal, "
                    "and baseline provenance"
                )
        elif self.trigger_mode is RuntimeConfigurationActivationMode.MANUAL_APPLY:
            if (
                self.trigger_run_id is not None
                or self.auto_adaptation_policy_fingerprint is not None
                or self.selected_proposal_id is not None
                or self.auto_adaptation_trigger_id is not None
                or self.candidate_set_fingerprint is not None
            ):
                raise ValueError("Manual Apply cannot carry policy-auto provenance")
        else:
            raise ValueError("Apply request trigger mode is not valid for activation")
        return self

    @model_serializer(mode="wrap")
    def serialize_with_legacy_manual_shape(self, handler: Any) -> Any:
        data = handler(self)
        if self.trigger_mode is RuntimeConfigurationActivationMode.MANUAL_APPLY:
            for key in (
                "trigger_mode",
                "trigger_run_id",
                "auto_adaptation_policy_fingerprint",
                "selected_proposal_id",
                "auto_adaptation_trigger_id",
                "candidate_set_fingerprint",
            ):
                data.pop(key, None)
        return data


class OptimizationApplyIntent(ContextMemoryModel):
    """Lifecycle proposal emitted from the explicit request, never by an Agent."""

    proposal_id: UUID
    explicit_request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class OptimizationRollbackRequest(ContextMemoryModel):
    """Explicit request to restore the predecessor of one active revision."""

    source_apply_effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_by: str = Field(min_length=1)
    current_configuration: RuntimeConfigurationSnapshot
    restore_configuration: RuntimeConfigurationSnapshot

    @model_validator(mode="after")
    def validate_snapshots(self) -> OptimizationRollbackRequest:
        if (
            self.current_configuration.scope != self.restore_configuration.scope
            or self.current_configuration.target_key
            != self.restore_configuration.target_key
        ):
            raise ValueError("Rollback snapshots cross configuration authority scope")
        if self.current_configuration.source_effect_fingerprint != (
            self.source_apply_effect_fingerprint
        ):
            raise ValueError("Rollback request does not target the active Apply")
        return self


class OptimizationRollbackIntent(ContextMemoryModel):
    source_apply_effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    explicit_request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeConfigurationSnapshot(ContextMemoryModel):
    """Immutable value for one scoped Runtime configuration revision."""

    snapshot_id: UUID
    scope: OptimizationScope
    target_key: OptimizationTargetKey
    revision: int = Field(ge=0)
    value: ImmutableJsonValue
    value_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_revision: int | None = Field(default=None, ge=0)
    source_revision: int | None = Field(default=None, ge=0)
    source_proposal_id: UUID | None = None
    source_effect_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    configuration_source: str = Field(min_length=1)
    activation_mode: RuntimeConfigurationActivationMode | None = None
    trigger_run_id: UUID | None = None
    auto_adaptation_policy_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    snapshot_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_snapshot(self) -> RuntimeConfigurationSnapshot:
        if self.value_fingerprint != decision_fingerprint(self.value):
            raise ValueError("Runtime configuration value fingerprint is invalid")
        expected = decision_fingerprint(runtime_configuration_snapshot_subject(self))
        if self.snapshot_fingerprint != expected:
            raise ValueError("Runtime configuration snapshot fingerprint is invalid")
        if self.revision == 0 and self.previous_revision is not None:
            raise ValueError("baseline configuration cannot have a predecessor")
        if self.revision > 0 and self.previous_revision is None:
            raise ValueError("activated configuration must reference its predecessor")
        if self.activation_mode is RuntimeConfigurationActivationMode.POLICY_AUTO:
            if (
                self.trigger_run_id is None
                or self.auto_adaptation_policy_fingerprint is None
                or self.source_proposal_id is None
            ):
                raise ValueError("Policy-auto configuration provenance is incomplete")
        elif (
            self.trigger_run_id is not None
            or self.auto_adaptation_policy_fingerprint is not None
        ):
            raise ValueError(
                "Only policy-auto configuration may bind policy trigger provenance"
            )
        return self

    @property
    def effective_activation_mode(self) -> RuntimeConfigurationActivationMode:
        if self.activation_mode is not None:
            return self.activation_mode
        if self.revision == 0:
            return RuntimeConfigurationActivationMode.DEFAULT_CONFIGURATION
        if "rollback" in self.configuration_source:
            return RuntimeConfigurationActivationMode.ROLLBACK
        return RuntimeConfigurationActivationMode.MANUAL_APPLY

    @model_serializer(mode="wrap")
    def serialize_with_legacy_provenance_shape(self, handler: Any) -> Any:
        data = handler(self)
        if self.activation_mode is None:
            data.pop("activation_mode", None)
            data.pop("trigger_run_id", None)
            data.pop("auto_adaptation_policy_fingerprint", None)
        return data


class OptimizationApplyEffect(ContextMemoryModel):
    proposal_id: UUID
    proposal_effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope: OptimizationScope
    target_key: OptimizationTargetKey
    expected_current_value: ImmutableJsonValue
    expected_current_value_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_current_revision: int = Field(ge=0)
    expected_current_configuration_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    proposed_value: ImmutableJsonValue
    rollback_revision: int = Field(ge=0)
    evidence_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    trigger_mode: RuntimeConfigurationActivationMode = (
        RuntimeConfigurationActivationMode.MANUAL_APPLY
    )
    trigger_run_id: UUID | None = None
    auto_adaptation_policy_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    selected_proposal_id: UUID | None = None
    auto_adaptation_trigger_id: UUID | None = None
    candidate_set_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_trigger_provenance(self) -> OptimizationApplyEffect:
        if self.trigger_mode is RuntimeConfigurationActivationMode.POLICY_AUTO:
            if (
                self.trigger_run_id is None
                or self.auto_adaptation_policy_fingerprint is None
                or self.selected_proposal_id != self.proposal_id
                or self.auto_adaptation_trigger_id is None
                or self.candidate_set_fingerprint is None
            ):
                raise ValueError("Policy-auto Effect provenance is incomplete")
        elif self.trigger_mode is RuntimeConfigurationActivationMode.MANUAL_APPLY:
            if (
                self.trigger_run_id is not None
                or self.auto_adaptation_policy_fingerprint is not None
                or self.selected_proposal_id is not None
                or self.auto_adaptation_trigger_id is not None
                or self.candidate_set_fingerprint is not None
            ):
                raise ValueError("Manual Effect cannot carry policy-auto provenance")
        else:
            raise ValueError("Apply Effect trigger mode is invalid")
        return self

    @model_serializer(mode="wrap")
    def serialize_with_legacy_manual_shape(self, handler: Any) -> Any:
        data = handler(self)
        if self.trigger_mode is RuntimeConfigurationActivationMode.MANUAL_APPLY:
            for key in (
                "trigger_mode",
                "trigger_run_id",
                "auto_adaptation_policy_fingerprint",
                "selected_proposal_id",
                "auto_adaptation_trigger_id",
                "candidate_set_fingerprint",
            ):
                data.pop(key, None)
        return data


class OptimizationRollbackEffect(ContextMemoryModel):
    source_apply_effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope: OptimizationScope
    target_key: OptimizationTargetKey
    expected_current_value: ImmutableJsonValue
    expected_current_revision: int = Field(ge=1)
    expected_current_configuration_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    restore_source_revision: int = Field(ge=0)
    restore_value: ImmutableJsonValue
    restore_source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class OptimizationApplyCommitReceipt(ContextMemoryModel):
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    operation: str = Field(min_length=1)
    scope: OptimizationScope
    target_key: OptimizationTargetKey
    active_revision: int = Field(ge=1)
    active_snapshot_id: UUID
    active_snapshot_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_revision: int = Field(ge=0)
    payload_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    activation_mode: RuntimeConfigurationActivationMode | None = None
    trigger_run_id: UUID | None = None
    source_proposal_id: UUID | None = None
    auto_adaptation_policy_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    committed_at: AwareDatetime = Field(default_factory=utc_now)


def runtime_configuration_snapshot_subject(
    snapshot: RuntimeConfigurationSnapshot,
) -> ImmutableJsonObject:
    subject: dict[str, ImmutableJsonValue] = {
        "snapshot_id": str(snapshot.snapshot_id),
        "scope": snapshot.scope.model_dump(mode="json"),
        "target_key": snapshot.target_key.value,
        "revision": snapshot.revision,
        "value": snapshot.value,
        "value_fingerprint": snapshot.value_fingerprint,
        "previous_revision": snapshot.previous_revision,
        "source_revision": snapshot.source_revision,
        "source_proposal_id": (
            str(snapshot.source_proposal_id)
            if snapshot.source_proposal_id is not None
            else None
        ),
        "source_effect_fingerprint": snapshot.source_effect_fingerprint,
        "configuration_source": snapshot.configuration_source,
    }
    # Optional provenance preserves read compatibility with Phase 4-B snapshots
    # while binding every newly activated Phase 4-C snapshot to its trigger.
    if snapshot.activation_mode is not None:
        subject["activation_mode"] = snapshot.activation_mode.value
    if snapshot.trigger_run_id is not None:
        subject["trigger_run_id"] = str(snapshot.trigger_run_id)
    if snapshot.auto_adaptation_policy_fingerprint is not None:
        subject["auto_adaptation_policy_fingerprint"] = (
            snapshot.auto_adaptation_policy_fingerprint
        )
    return subject


def create_runtime_configuration_snapshot(
    *,
    scope: OptimizationScope,
    target_key: OptimizationTargetKey,
    revision: int,
    value: ImmutableJsonValue,
    previous_revision: int | None,
    source_revision: int | None,
    source_proposal_id: UUID | None,
    source_effect_fingerprint: str | None,
    configuration_source: str,
    activation_mode: RuntimeConfigurationActivationMode | None = None,
    trigger_run_id: UUID | None = None,
    auto_adaptation_policy_fingerprint: str | None = None,
    created_at: AwareDatetime | None = None,
) -> RuntimeConfigurationSnapshot:
    snapshot_id = stable_runtime_configuration_snapshot_id(
        scope, target_key, revision
    )
    value_fingerprint = decision_fingerprint(value)
    snapshot_created_at = created_at or utc_now()
    provisional = RuntimeConfigurationSnapshot.model_construct(
        snapshot_id=snapshot_id,
        scope=scope,
        target_key=target_key,
        revision=revision,
        value=value,
        value_fingerprint=value_fingerprint,
        previous_revision=previous_revision,
        source_revision=source_revision,
        source_proposal_id=source_proposal_id,
        source_effect_fingerprint=source_effect_fingerprint,
        configuration_source=configuration_source,
        activation_mode=activation_mode,
        trigger_run_id=trigger_run_id,
        auto_adaptation_policy_fingerprint=auto_adaptation_policy_fingerprint,
        created_at=snapshot_created_at,
        snapshot_fingerprint="0" * 64,
    )
    return RuntimeConfigurationSnapshot(
        snapshot_id=snapshot_id,
        scope=scope,
        target_key=target_key,
        revision=revision,
        value=value,
        value_fingerprint=value_fingerprint,
        previous_revision=previous_revision,
        source_revision=source_revision,
        source_proposal_id=source_proposal_id,
        source_effect_fingerprint=source_effect_fingerprint,
        configuration_source=configuration_source,
        activation_mode=activation_mode,
        trigger_run_id=trigger_run_id,
        auto_adaptation_policy_fingerprint=auto_adaptation_policy_fingerprint,
        created_at=snapshot_created_at,
        snapshot_fingerprint=decision_fingerprint(
            runtime_configuration_snapshot_subject(provisional)
        ),
    )


class OptimizationProposalStore(Protocol):
    async def commit(
        self,
        effect: OptimizationProposalEffect,
        *,
        source_decision_request_id: UUID,
        effect_fingerprint: str,
        permit: RuntimeCommitPermit | None,
        target: GovernanceTarget | None,
        subject_fingerprint: str | None,
    ) -> OptimizationProposal: ...

    async def load_by_effect(
        self, effect_fingerprint: str
    ) -> OptimizationProposal | None: ...

    async def load_by_id(self, proposal_id: UUID) -> OptimizationProposal | None: ...

    async def load_receipt(
        self, effect_fingerprint: str
    ) -> OptimizationProposalCommitReceipt | None: ...

    async def list_for_scope(
        self, scope: OptimizationScope
    ) -> tuple[OptimizationProposal, ...]: ...

    async def verify_phase3_provenance(self, proposal_id: UUID) -> bool: ...


def stable_optimization_request_id(
    trigger_run_id: UUID,
    scope: OptimizationScope,
) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        "adaptive-agent-runtime:optimization-assessment:"
        f"{trigger_run_id}:{scope.tenant}:{scope.project}:"
        f"{scope.application}:{scope.decision_type}",
    )


def stable_optimization_proposal_id(request_id: UUID) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        f"adaptive-agent-runtime:optimization-proposal:{request_id}",
    )


def stable_optimization_target_ref(
    scope: OptimizationScope,
    target_key: OptimizationTargetKey,
) -> str:
    value = uuid5(
        NAMESPACE_URL,
        "adaptive-agent-runtime:optimization-target:"
        f"{scope.model_dump_json()}:{target_key.value}",
    )
    return f"target-{value.hex}"


def stable_optimization_evidence_ref(
    learning_insight_id: UUID,
    version: int,
) -> str:
    value = uuid5(
        NAMESPACE_URL,
        "adaptive-agent-runtime:optimization-evidence:"
        f"{learning_insight_id}:{version}",
    )
    return f"optimization-evidence-{value.hex}"


def stable_optimization_apply_request_id(
    proposal_id: UUID,
    *,
    trigger_run_id: UUID | None = None,
    baseline_revision: int | None = None,
) -> UUID:
    if trigger_run_id is not None:
        if baseline_revision is None or baseline_revision < 0:
            raise ValueError("Policy-auto Apply identity requires a baseline revision")
        return uuid5(
            NAMESPACE_URL,
            "adaptive-agent-runtime:optimization-apply:policy-auto:"
            f"{trigger_run_id}:{proposal_id}:{baseline_revision}",
        )
    if baseline_revision is not None:
        raise ValueError("Manual Apply identity cannot include a baseline revision")
    return uuid5(
        NAMESPACE_URL,
        f"adaptive-agent-runtime:optimization-apply:{proposal_id}",
    )


def stable_optimization_rollback_request_id(
    source_apply_effect_fingerprint: str,
    active_revision: int,
) -> UUID:
    del active_revision
    return uuid5(
        NAMESPACE_URL,
        "adaptive-agent-runtime:optimization-rollback:"
        f"{source_apply_effect_fingerprint}",
    )


def stable_runtime_configuration_snapshot_id(
    scope: OptimizationScope,
    target_key: OptimizationTargetKey,
    revision: int,
) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        "adaptive-agent-runtime:runtime-configuration:"
        f"{scope.model_dump_json()}:{target_key.value}:{revision}",
    )


def runtime_configuration_target_id(
    scope: OptimizationScope,
    target_key: OptimizationTargetKey,
) -> str:
    value = uuid5(
        NAMESPACE_URL,
        "adaptive-agent-runtime:runtime-configuration-target:"
        f"{scope.model_dump_json()}:{target_key.value}",
    )
    return f"runtime-configuration-{value.hex}"

"""Explicit Phase 4-B Optimization Apply and Rollback Decision gateways."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, JsonValue

from adaptive_agent_runtime import TraceSink
from adaptive_agent_runtime.decisioning import (
    AgentCallResult,
    AgentContext,
    ContextProjectionPolicy,
    ContextSensitivity,
    DecisionApplyReceipt,
    DecisionBasis,
    DecisionBudget,
    DecisionCheckpoint,
    DecisionCheckpointStage,
    DecisionCheckpointStore,
    DecisionConstraint,
    DecisionCorrelation,
    DecisionEvidenceReference,
    DecisionFaultPoint,
    DecisionGovernanceScope,
    DecisionLifecycleCoordinator,
    DecisionProducer,
    DecisionProposal,
    DecisionReconciliation,
    DecisionReconciliationStatus,
    DecisionRequest,
    DecisionResultStatus,
    DecisionRiskLevel,
    DecisionTarget,
    InMemoryDecisionCheckpointStore,
    NormalizedDecisionEffect,
    PolicyAgentContextBuilder,
    ProjectionSource,
    ProjectionSources,
    RuntimeDecisionTraceWriter,
    RuntimeDecisionValidator,
    decision_fingerprint,
)
from adaptive_agent_runtime.governance import (
    DecisionGovernanceBinding,
    GovernanceAuthorizationIssuer,
    GovernanceEvaluator,
    GovernanceTarget,
    GovernedDecisionApplier,
    GovernedOperationExecutor,
    HumanReviewService,
    RuntimeCommitPermit,
    RuntimeDecisionGovernanceAdapter,
    governance_fingerprint,
)
from adaptive_agent_runtime.optimization import (
    OPTIMIZATION_APPLY_COMMIT_OPERATION,
    OPTIMIZATION_APPLY_DECISION_TYPE,
    OPTIMIZATION_ROLLBACK_COMMIT_OPERATION,
    OPTIMIZATION_ROLLBACK_DECISION_TYPE,
    GovernedRuntimeConfigurationCommitPort,
    OptimizationApplyCommitReceipt,
    OptimizationApplyEffect,
    OptimizationApplyIntent,
    OptimizationApplyRequest,
    OptimizationProposal,
    OptimizationProposalQuery,
    OptimizationProposalStatus,
    OptimizationRollbackEffect,
    OptimizationRollbackIntent,
    OptimizationRollbackRequest,
    OptimizationScope,
    OptimizationTargetKey,
    RuntimeConfigurationActivationMode,
    RuntimeConfigurationSnapshot,
    runtime_configuration_target_id,
    stable_optimization_apply_request_id,
    stable_optimization_rollback_request_id,
)
from adaptive_agent_runtime.persistence import default_runtime_configuration_snapshot

from applications.research_agent.decision_support import (
    RecordingAuthorizationIssuer,
    RecordingGovernanceEvaluator,
)
from applications.research_agent.report import GovernanceRecord


_PHASE_4B_SCOPE = OptimizationScope(
    tenant="default",
    project="research",
    application="research_agent",
    decision_type="planning.task_graph.initialize",
)
_INPUT_SOURCE_TYPE = "optimization_configuration_request"
RequestT = TypeVar("RequestT", bound=BaseModel)
IntentT = TypeVar("IntentT", bound=BaseModel)
EffectT = TypeVar("EffectT", bound=BaseModel)


@dataclass(frozen=True)
class ResearchOptimizationConfigurationResult:
    request_id: UUID
    receipt: OptimizationApplyCommitReceipt | None
    active_configuration: RuntimeConfigurationSnapshot | None
    governance_record: GovernanceRecord | None
    review_pending: bool = False
    decision_status: DecisionResultStatus | None = None
    reconciliation_status: DecisionReconciliationStatus | None = None
    reason: str | None = None


class _ExplicitIntentProposalProducer(Generic[IntentT]):
    """Bridge an explicit caller intent into the existing typed lifecycle.

    This is not an Agent invocation and performs no semantic proposal generation.
    """

    module_id = "research_agent.optimization_configuration.explicit_intent"

    def __init__(
        self,
        intent: IntentT,
        operation: str,
        *,
        policy_auto: bool = False,
    ) -> None:
        self._intent = intent
        self._operation = operation
        self._policy_auto = policy_auto

    async def propose(self, context: AgentContext) -> AgentCallResult[IntentT]:
        return AgentCallResult(
            proposal=DecisionProposal(
                request_id=context.request_id,
                proposal_type=context.decision_type,
                producer=DecisionProducer(
                    producer_id=(
                        "runtime.policy.auto_adaptation"
                        if self._policy_auto
                        else "runtime.explicit_request"
                    ),
                    capability=(
                        "deterministic_runtime_policy"
                        if self._policy_auto
                        else "operator_decision_gateway"
                    ),
                    implementation_version=(
                        "phase-4c" if self._policy_auto else "phase-4b"
                    ),
                ),
                input_snapshot_fingerprint=context.basis_fingerprint,
                context_fingerprint=context.context_fingerprint,
                selected_action=self._operation,
                payload=self._intent,
                rationale="Apply only the explicitly requested governed operation.",
                evidence_refs=tuple(item.evidence_id for item in context.evidence),
                confidence=1.0,
            )
        )


class _ApplyEffectNormalizer:
    module_id = "research_agent.optimization_apply.normalizer"

    def normalize(
        self,
        request: DecisionRequest[OptimizationApplyRequest],
        proposal: DecisionProposal[OptimizationApplyIntent],
    ) -> NormalizedDecisionEffect[OptimizationApplyEffect]:
        source = request.payload
        if proposal.payload.proposal_id != source.proposal_id:
            raise ValueError("Apply intent changed the explicit Proposal")
        if proposal.payload.explicit_request_fingerprint != decision_fingerprint(source):
            raise ValueError("Apply intent is not bound to the explicit request")
        stored = source.proposal_snapshot
        current = source.current_configuration
        if current is None:
            current = default_runtime_configuration_snapshot(stored.scope)
        _validate_proposal_for_apply(
            stored,
            current,
            trigger_mode=source.trigger_mode,
        )
        effect = OptimizationApplyEffect(
            proposal_id=stored.proposal_id,
            proposal_effect_fingerprint=stored.effect_fingerprint,
            proposal_record_fingerprint=decision_fingerprint(stored),
            scope=stored.scope,
            target_key=stored.target_key,
            expected_current_value=current.value,
            expected_current_value_fingerprint=current.value_fingerprint,
            expected_current_revision=current.revision,
            expected_current_configuration_fingerprint=(
                current.snapshot_fingerprint
            ),
            proposed_value=stored.proposed_value,
            rollback_revision=current.revision,
            evidence_set_fingerprint=stored.evidence_set_fingerprint,
            trigger_mode=source.trigger_mode,
            trigger_run_id=source.trigger_run_id,
            auto_adaptation_policy_fingerprint=(
                source.auto_adaptation_policy_fingerprint
            ),
            selected_proposal_id=source.selected_proposal_id,
            auto_adaptation_trigger_id=source.auto_adaptation_trigger_id,
            candidate_set_fingerprint=source.candidate_set_fingerprint,
        )
        return NormalizedDecisionEffect[OptimizationApplyEffect].create(
            payload=effect,
            operation=OPTIMIZATION_APPLY_COMMIT_OPERATION,
            target=DecisionTarget(
                target_type="runtime_configuration",
                target_id=runtime_configuration_target_id(
                    effect.scope, effect.target_key
                ),
            ),
            governance_scope=DecisionGovernanceScope.EVOLUTION,
            risk=DecisionRiskLevel.MEDIUM,
            impact_score=0.45,
            reversible=True,
            impact_description=(
                "Activate one bounded planner.max_nodes value for future Runs only."
            ),
        )


class _RollbackEffectNormalizer:
    module_id = "research_agent.optimization_rollback.normalizer"

    def normalize(
        self,
        request: DecisionRequest[OptimizationRollbackRequest],
        proposal: DecisionProposal[OptimizationRollbackIntent],
    ) -> NormalizedDecisionEffect[OptimizationRollbackEffect]:
        source = request.payload
        if proposal.payload.source_apply_effect_fingerprint != (
            source.source_apply_effect_fingerprint
        ):
            raise ValueError("Rollback intent changed the explicit source Apply")
        if proposal.payload.explicit_request_fingerprint != decision_fingerprint(source):
            raise ValueError("Rollback intent is not bound to the explicit request")
        current = source.current_configuration
        restore = source.restore_configuration
        if current.scope != _PHASE_4B_SCOPE:
            raise ValueError("Rollback scope is not enabled in Phase 4-B")
        if current.target_key is not OptimizationTargetKey.PLANNER_MAX_NODES:
            raise ValueError("Rollback target is not enabled in Phase 4-B")
        effect = OptimizationRollbackEffect(
            source_apply_effect_fingerprint=(
                source.source_apply_effect_fingerprint
            ),
            scope=current.scope,
            target_key=current.target_key,
            expected_current_value=current.value,
            expected_current_revision=current.revision,
            expected_current_configuration_fingerprint=(
                current.snapshot_fingerprint
            ),
            restore_source_revision=restore.revision,
            restore_value=restore.value,
            restore_source_fingerprint=restore.snapshot_fingerprint,
        )
        return NormalizedDecisionEffect[OptimizationRollbackEffect].create(
            payload=effect,
            operation=OPTIMIZATION_ROLLBACK_COMMIT_OPERATION,
            target=DecisionTarget(
                target_type="runtime_configuration",
                target_id=runtime_configuration_target_id(
                    effect.scope, effect.target_key
                ),
            ),
            governance_scope=DecisionGovernanceScope.EVOLUTION,
            risk=DecisionRiskLevel.LOW,
            impact_score=0.25,
            reversible=True,
            impact_description=(
                "Restore a prior immutable planner.max_nodes value as a new revision."
            ),
        )


class _ApplyBasisProvider:
    module_id = "research_agent.optimization_apply.basis"

    def __init__(
        self,
        proposals: OptimizationProposalQuery,
        configurations: GovernedRuntimeConfigurationCommitPort,
    ) -> None:
        self._proposals = proposals
        self._configurations = configurations

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        payload = OptimizationApplyRequest.model_validate(request.payload)
        proposal = await self._proposals.load_by_id(payload.proposal_id)
        if proposal is None:
            return DecisionBasis(
                snapshot_fingerprint="0" * 64,
                configuration_revision=request.basis.configuration_revision,
            )
        current = await self._configurations.load_active(
            proposal.scope, proposal.target_key
        )
        if (
            current is None
            and payload.trigger_mode
            is RuntimeConfigurationActivationMode.POLICY_AUTO
        ):
            current = default_runtime_configuration_snapshot(proposal.scope)
        rebound = payload.model_copy(
            update={
                "proposal_snapshot": proposal,
                "current_configuration": current,
            }
        )
        return DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(rebound),
            configuration_revision=(
                current.revision if current is not None else 0
            ),
        )


class _RollbackBasisProvider:
    module_id = "research_agent.optimization_rollback.basis"

    def __init__(
        self,
        configurations: GovernedRuntimeConfigurationCommitPort,
    ) -> None:
        self._configurations = configurations

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        payload = OptimizationRollbackRequest.model_validate(request.payload)
        current = await self._configurations.load_active(
            payload.current_configuration.scope,
            payload.current_configuration.target_key,
        )
        restore = await self._configurations.load_snapshot(
            payload.restore_configuration.scope,
            payload.restore_configuration.target_key,
            payload.restore_configuration.revision,
        )
        if current is None or restore is None:
            return DecisionBasis(
                snapshot_fingerprint="0" * 64,
                configuration_revision=request.basis.configuration_revision,
            )
        rebound = payload.model_copy(
            update={
                "current_configuration": current,
                "restore_configuration": restore,
            }
        )
        return DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(rebound),
            configuration_revision=current.revision,
        )


class ResearchOptimizationConfigurationGateway:
    """Public explicit Decision gateway; it exposes no raw activation method."""

    module_id = "research_agent.optimization_configuration.gateway"

    def __init__(
        self,
        *,
        proposals: OptimizationProposalQuery,
        configurations: GovernedRuntimeConfigurationCommitPort,
        governance: GovernanceEvaluator,
        reviews: HumanReviewService,
        issuer: GovernanceAuthorizationIssuer,
        operation_executor: GovernedOperationExecutor,
        trace_sink: TraceSink,
        apply_checkpoints: DecisionCheckpointStore[
            OptimizationApplyRequest,
            OptimizationApplyIntent,
            OptimizationApplyEffect,
        ]
        | None = None,
        rollback_checkpoints: DecisionCheckpointStore[
            OptimizationRollbackRequest,
            OptimizationRollbackIntent,
            OptimizationRollbackEffect,
        ]
        | None = None,
        fault_injector: Callable[[DecisionFaultPoint, Any], None] | None = None,
    ) -> None:
        self._proposals = proposals
        self._configurations = configurations
        self._governance = governance
        self._reviews = reviews
        self._issuer = issuer
        self._operation_executor = operation_executor
        self._trace_sink = trace_sink
        self._apply_checkpoints = apply_checkpoints or InMemoryDecisionCheckpointStore()
        self._rollback_checkpoints = (
            rollback_checkpoints or InMemoryDecisionCheckpointStore()
        )
        self._fault_injector = fault_injector

    async def request_apply(
        self,
        proposal_id: UUID,
        *,
        requested_by: str,
    ) -> ResearchOptimizationConfigurationResult:
        return await self._request_apply(
            proposal_id,
            requested_by=requested_by,
            trigger_mode=RuntimeConfigurationActivationMode.MANUAL_APPLY,
            strict=True,
        )

    async def request_policy_auto_apply(
        self,
        proposal_id: UUID,
        *,
        trigger_run_id: UUID,
        auto_adaptation_policy_fingerprint: str,
        auto_adaptation_trigger_id: UUID,
        candidate_set_fingerprint: str,
        baseline_revision: int,
    ) -> ResearchOptimizationConfigurationResult:
        """Enter the existing Apply lifecycle from a durable Runtime policy claim."""

        return await self._request_apply(
            proposal_id,
            requested_by="runtime.policy.auto_adaptation",
            trigger_mode=RuntimeConfigurationActivationMode.POLICY_AUTO,
            trigger_run_id=trigger_run_id,
            auto_adaptation_policy_fingerprint=(
                auto_adaptation_policy_fingerprint
            ),
            auto_adaptation_trigger_id=auto_adaptation_trigger_id,
            candidate_set_fingerprint=candidate_set_fingerprint,
            baseline_revision=baseline_revision,
            strict=False,
        )

    async def _request_apply(
        self,
        proposal_id: UUID,
        *,
        requested_by: str,
        trigger_mode: RuntimeConfigurationActivationMode,
        strict: bool,
        trigger_run_id: UUID | None = None,
        auto_adaptation_policy_fingerprint: str | None = None,
        auto_adaptation_trigger_id: UUID | None = None,
        candidate_set_fingerprint: str | None = None,
        baseline_revision: int | None = None,
    ) -> ResearchOptimizationConfigurationResult:
        policy_auto = trigger_mode is RuntimeConfigurationActivationMode.POLICY_AUTO
        request_id = stable_optimization_apply_request_id(
            proposal_id,
            trigger_run_id=trigger_run_id if policy_auto else None,
            baseline_revision=baseline_revision if policy_auto else None,
        )
        existing = await self._apply_checkpoints.load(request_id)
        if existing is not None:
            coordinator, recorder, issuer = self._apply_coordinator(existing.request)
            checkpoint = await coordinator.resume(request_id)
            return await self._result(
                checkpoint,
                recorder,
                issuer,
                strict=strict,
            )
        proposal = await self._proposals.load_by_id(proposal_id)
        if proposal is None:
            raise ValueError("Optimization Proposal was not found")
        active = await self._configurations.load_active(
            proposal.scope, proposal.target_key
        )
        request_current = (
            active or default_runtime_configuration_snapshot(proposal.scope)
            if policy_auto
            else active
        )
        if (
            policy_auto
            and (
                baseline_revision is None
                or request_current is None
                or request_current.revision != baseline_revision
            )
        ):
            raise ValueError("Policy-auto Apply baseline is stale")
        payload = OptimizationApplyRequest(
            proposal_id=proposal_id,
            requested_by=requested_by,
            proposal_snapshot=proposal,
            current_configuration=request_current,
            trigger_mode=trigger_mode,
            trigger_run_id=trigger_run_id,
            auto_adaptation_policy_fingerprint=(
                auto_adaptation_policy_fingerprint
            ),
            selected_proposal_id=proposal_id if policy_auto else None,
            auto_adaptation_trigger_id=auto_adaptation_trigger_id,
            candidate_set_fingerprint=candidate_set_fingerprint,
        )
        basis = DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(payload),
            configuration_revision=(
                request_current.revision if request_current is not None else 0
            ),
        )
        trigger_evidence = (
            DecisionEvidenceReference(
                evidence_id=f"policy-auto-trigger:{auto_adaptation_trigger_id}",
                kind="runtime.policy_auto_trigger",
                source="auto_adaptation_trigger_store",
                reliability=1.0,
                summary=(
                    "A deterministic Runtime policy selected the sole eligible "
                    "bounded Proposal after its trigger Run completed."
                ),
            )
            if policy_auto
            else DecisionEvidenceReference(
                evidence_id=f"explicit-request:{request_id}",
                kind="operator.explicit_request",
                source="optimization_configuration_gateway",
                reliability=1.0,
                summary="An explicit caller requested this bounded Apply.",
            )
        )
        request = DecisionRequest(
            request_id=request_id,
            decision_type=OPTIMIZATION_APPLY_DECISION_TYPE,
            target=DecisionTarget(
                target_type="runtime_configuration",
                target_id=runtime_configuration_target_id(
                    proposal.scope, proposal.target_key
                ),
            ),
            correlation=DecisionCorrelation(
                run_id=(trigger_run_id if trigger_run_id is not None else request_id)
            ),
            basis=basis,
            payload=payload,
            allowed_actions=(OPTIMIZATION_APPLY_COMMIT_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="phase-4b-single-target",
                    description="Only planner.max_nodes in the Research Planning scope.",
                ),
                DecisionConstraint(
                    constraint_id="future-runs-only",
                    description="Activation affects only Research Runs created later.",
                ),
                DecisionConstraint(
                    constraint_id=(
                        "phase-4c-policy-auto"
                        if policy_auto
                        else "phase-4b-manual-apply"
                    ),
                    description=(
                        "Runtime policy trigger is fingerprint-bound and delta is one."
                        if policy_auto
                        else "Apply originates from an explicit operator request."
                    ),
                ),
            ),
            evidence=(
                trigger_evidence,
                DecisionEvidenceReference(
                    evidence_id=f"proposal:{proposal.proposal_id}",
                    kind="optimization.proposal.committed",
                    source="optimization_proposal_store",
                    reliability=1.0,
                    summary="Committed immutable Optimization Proposal.",
                ),
            ),
            budget=DecisionBudget(max_agent_calls=1, max_revision_count=0),
        )
        coordinator, recorder, issuer = self._apply_coordinator(request)
        checkpoint = await coordinator.run(
            request,
            sources=_explicit_sources(request),
            policy=_explicit_policy(OPTIMIZATION_APPLY_DECISION_TYPE),
        )
        return await self._result(
            checkpoint,
            recorder,
            issuer,
            strict=strict,
        )

    async def request_rollback(
        self,
        source_apply_effect_fingerprint: str,
        *,
        requested_by: str,
    ) -> ResearchOptimizationConfigurationResult:
        active = await self._configurations.load_active(
            _PHASE_4B_SCOPE, OptimizationTargetKey.PLANNER_MAX_NODES
        )
        if active is None or active.previous_revision is None:
            raise ValueError("No governed active configuration can be rolled back")
        request_id = stable_optimization_rollback_request_id(
            source_apply_effect_fingerprint, active.revision
        )
        existing = await self._rollback_checkpoints.load(request_id)
        if existing is not None:
            coordinator, recorder, issuer = self._rollback_coordinator(existing.request)
            checkpoint = await coordinator.resume(request_id)
            return await self._result(
                checkpoint,
                recorder,
                issuer,
                strict=True,
            )
        restore = await self._configurations.load_snapshot(
            active.scope, active.target_key, active.previous_revision
        )
        if restore is None:
            raise RuntimeError("Rollback predecessor snapshot is unavailable")
        payload = OptimizationRollbackRequest(
            source_apply_effect_fingerprint=source_apply_effect_fingerprint,
            requested_by=requested_by,
            current_configuration=active,
            restore_configuration=restore,
        )
        basis = DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(payload),
            configuration_revision=active.revision,
        )
        request = DecisionRequest(
            request_id=request_id,
            decision_type=OPTIMIZATION_ROLLBACK_DECISION_TYPE,
            target=DecisionTarget(
                target_type="runtime_configuration",
                target_id=runtime_configuration_target_id(
                    active.scope, active.target_key
                ),
            ),
            correlation=DecisionCorrelation(run_id=request_id),
            basis=basis,
            payload=payload,
            allowed_actions=(OPTIMIZATION_ROLLBACK_COMMIT_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="explicit-rollback-only",
                    description="Rollback is manually requested and never automatic.",
                ),
            ),
            evidence=(
                DecisionEvidenceReference(
                    evidence_id=f"explicit-request:{request_id}",
                    kind="operator.explicit_request",
                    source="optimization_configuration_gateway",
                    reliability=1.0,
                    summary="An explicit caller requested this Rollback.",
                ),
                DecisionEvidenceReference(
                    evidence_id=(
                        f"apply-effect:{source_apply_effect_fingerprint}"
                    ),
                    kind="optimization.apply.committed",
                    source="runtime_configuration_store",
                    reliability=1.0,
                    summary="Committed active configuration Apply.",
                ),
            ),
            budget=DecisionBudget(max_agent_calls=1, max_revision_count=0),
        )
        coordinator, recorder, issuer = self._rollback_coordinator(request)
        checkpoint = await coordinator.run(
            request,
            sources=_explicit_sources(request),
            policy=_explicit_policy(OPTIMIZATION_ROLLBACK_DECISION_TYPE),
        )
        return await self._result(
            checkpoint,
            recorder,
            issuer,
            strict=True,
        )

    def _apply_coordinator(
        self,
        request: DecisionRequest[OptimizationApplyRequest],
    ) -> tuple[
        DecisionLifecycleCoordinator[
            OptimizationApplyRequest,
            OptimizationApplyIntent,
            OptimizationApplyEffect,
            DecisionGovernanceBinding,
        ],
        RecordingGovernanceEvaluator,
        RecordingAuthorizationIssuer,
    ]:
        intent = OptimizationApplyIntent(
            proposal_id=request.payload.proposal_id,
            explicit_request_fingerprint=decision_fingerprint(request.payload),
        )
        return self._coordinator(
            request=request,
            producer=_ExplicitIntentProposalProducer(
                intent,
                OPTIMIZATION_APPLY_COMMIT_OPERATION,
                policy_auto=(
                    request.payload.trigger_mode
                    is RuntimeConfigurationActivationMode.POLICY_AUTO
                ),
            ),
            basis_provider=_ApplyBasisProvider(
                self._proposals, self._configurations
            ),
            normalizer=_ApplyEffectNormalizer(),
            checkpoints=self._apply_checkpoints,
            checkpoint_type=DecisionCheckpoint[
                OptimizationApplyRequest,
                OptimizationApplyIntent,
                OptimizationApplyEffect,
            ],
            commit=self._commit_apply,
        )

    def _rollback_coordinator(
        self,
        request: DecisionRequest[OptimizationRollbackRequest],
    ) -> tuple[
        DecisionLifecycleCoordinator[
            OptimizationRollbackRequest,
            OptimizationRollbackIntent,
            OptimizationRollbackEffect,
            DecisionGovernanceBinding,
        ],
        RecordingGovernanceEvaluator,
        RecordingAuthorizationIssuer,
    ]:
        intent = OptimizationRollbackIntent(
            source_apply_effect_fingerprint=(
                request.payload.source_apply_effect_fingerprint
            ),
            explicit_request_fingerprint=decision_fingerprint(request.payload),
        )
        return self._coordinator(
            request=request,
            producer=_ExplicitIntentProposalProducer(
                intent, OPTIMIZATION_ROLLBACK_COMMIT_OPERATION
            ),
            basis_provider=_RollbackBasisProvider(self._configurations),
            normalizer=_RollbackEffectNormalizer(),
            checkpoints=self._rollback_checkpoints,
            checkpoint_type=DecisionCheckpoint[
                OptimizationRollbackRequest,
                OptimizationRollbackIntent,
                OptimizationRollbackEffect,
            ],
            commit=self._commit_rollback,
        )

    def _coordinator(
        self,
        *,
        request: DecisionRequest[Any],
        producer: Any,
        basis_provider: Any,
        normalizer: Any,
        checkpoints: Any,
        checkpoint_type: Any,
        commit: Any,
    ) -> tuple[Any, RecordingGovernanceEvaluator, RecordingAuthorizationIssuer]:
        recorder = RecordingGovernanceEvaluator(self._governance)
        issuer = RecordingAuthorizationIssuer(self._issuer)

        async def forbidden_apply(effect: BaseModel) -> JsonValue:
            del effect
            raise RuntimeError("Configuration commit requires a Runtime Permit")

        async def apply_authorized(
            normalized: NormalizedDecisionEffect[Any],
            permit: RuntimeCommitPermit,
        ) -> JsonValue:
            receipt = await commit(normalized, permit)
            active = await self._configurations.load_active(
                normalized.payload.scope,
                normalized.payload.target_key,
            )
            if (
                active is None
                or active.revision != receipt.active_revision
                or active.snapshot_id != receipt.active_snapshot_id
                or active.snapshot_fingerprint
                != receipt.active_snapshot_fingerprint
            ):
                raise RuntimeError("Configuration commit read-back failed")
            return _receipt_result(receipt)

        async def reconcile(
            normalized: NormalizedDecisionEffect[Any],
        ) -> DecisionReconciliation:
            try:
                receipt = await self._configurations.load_receipt(
                    normalized.effect_fingerprint
                )
                if receipt is None:
                    return DecisionReconciliation(
                        status=DecisionReconciliationStatus.NOT_COMMITTED,
                        reason="Configuration Effect has not been committed",
                    )
                if receipt.payload_fingerprint != decision_fingerprint(
                    normalized.payload
                ):
                    return DecisionReconciliation(
                        status=DecisionReconciliationStatus.UNKNOWN,
                        reason="Configuration Receipt conflicts with authorized Effect",
                    )
                committed = await self._configurations.load_snapshot(
                    receipt.scope,
                    receipt.target_key,
                    receipt.active_revision,
                )
                active = await self._configurations.load_active(
                    receipt.scope, receipt.target_key
                )
                if (
                    committed is None
                    or committed.snapshot_id != receipt.active_snapshot_id
                    or committed.snapshot_fingerprint
                    != receipt.active_snapshot_fingerprint
                    or active is None
                    or active.revision < receipt.active_revision
                ):
                    return DecisionReconciliation(
                        status=DecisionReconciliationStatus.UNKNOWN,
                        reason="Configuration commit history is indeterminate",
                    )
            except Exception as exc:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.UNKNOWN,
                    reason=f"Configuration reconciliation failed: {type(exc).__name__}: {exc}",
                )
            result = _receipt_result(receipt)
            return DecisionReconciliation(
                status=DecisionReconciliationStatus.COMMITTED,
                reason="Configuration Effect was committed and read back",
                apply_receipt=DecisionApplyReceipt(
                    effect_fingerprint=normalized.effect_fingerprint,
                    committed_state_fingerprint=decision_fingerprint(result),
                    result=result,
                ),
            )

        coordinator = DecisionLifecycleCoordinator(
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=producer,
            basis_provider=basis_provider,
            validator=RuntimeDecisionValidator(normalizer=normalizer),
            governance=RuntimeDecisionGovernanceAdapter(
                evaluator=recorder,
                authorization_issuer=issuer,
                review_service=self._reviews,
            ),
            applier=GovernedDecisionApplier(
                executor=self._operation_executor,
                apply_effect=forbidden_apply,
                apply_authorized_effect=apply_authorized,
                reconcile_effect=reconcile,
            ),
            checkpoint_store=checkpoints,
            checkpoint_type=checkpoint_type,
            trace_writer=RuntimeDecisionTraceWriter(self._trace_sink),
            fault_injector=self._fault_injector,
        )
        return coordinator, recorder, issuer

    async def _commit_apply(
        self,
        normalized: NormalizedDecisionEffect[OptimizationApplyEffect],
        permit: RuntimeCommitPermit,
    ) -> OptimizationApplyCommitReceipt:
        return await self._configurations.commit_apply(
            normalized.payload,
            effect_fingerprint=normalized.effect_fingerprint,
            permit=permit,
            target=GovernanceTarget(
                target_type=normalized.target.target_type,
                target_id=normalized.target.target_id,
            ),
            subject_fingerprint=governance_fingerprint(normalized),
        )

    async def _commit_rollback(
        self,
        normalized: NormalizedDecisionEffect[OptimizationRollbackEffect],
        permit: RuntimeCommitPermit,
    ) -> OptimizationApplyCommitReceipt:
        return await self._configurations.commit_rollback(
            normalized.payload,
            effect_fingerprint=normalized.effect_fingerprint,
            permit=permit,
            target=GovernanceTarget(
                target_type=normalized.target.target_type,
                target_id=normalized.target.target_id,
            ),
            subject_fingerprint=governance_fingerprint(normalized),
        )

    async def _result(
        self,
        checkpoint: Any,
        recorder: RecordingGovernanceEvaluator,
        issuer: RecordingAuthorizationIssuer,
        *,
        strict: bool,
    ) -> ResearchOptimizationConfigurationResult:
        if checkpoint.stage is DecisionCheckpointStage.REVIEW_PENDING:
            return ResearchOptimizationConfigurationResult(
                request_id=checkpoint.request_id,
                receipt=None,
                active_configuration=None,
                governance_record=_governance_record(recorder, issuer),
                review_pending=True,
                reason="Governance requires Human Review",
            )
        if (
            checkpoint.result is None
            or checkpoint.result.status is not DecisionResultStatus.APPLIED
            or checkpoint.validated_decision is None
        ):
            reason = checkpoint.result.reason if checkpoint.result else str(checkpoint.stage)
            if strict:
                raise RuntimeError(
                    f"Optimization configuration Decision failed: {reason}"
                )
            return ResearchOptimizationConfigurationResult(
                request_id=checkpoint.request_id,
                receipt=None,
                active_configuration=None,
                governance_record=_governance_record(recorder, issuer),
                decision_status=(
                    checkpoint.result.status if checkpoint.result is not None else None
                ),
                reconciliation_status=(
                    checkpoint.result.reconciliation_status
                    if checkpoint.result is not None
                    else None
                ),
                reason=reason,
            )
        normalized = checkpoint.validated_decision.normalized_effect
        receipt = await self._configurations.load_receipt(
            normalized.effect_fingerprint
        )
        if receipt is None:
            raise RuntimeError("APPLIED configuration Decision has no Receipt")
        committed = await self._configurations.load_snapshot(
            receipt.scope,
            receipt.target_key,
            receipt.active_revision,
        )
        if (
            committed is None
            or committed.snapshot_id != receipt.active_snapshot_id
            or committed.snapshot_fingerprint
            != receipt.active_snapshot_fingerprint
        ):
            raise RuntimeError("APPLIED configuration Decision has no committed snapshot")
        return ResearchOptimizationConfigurationResult(
            request_id=checkpoint.request_id,
            receipt=receipt,
            active_configuration=committed,
            governance_record=_governance_record(recorder, issuer),
            decision_status=checkpoint.result.status,
            reconciliation_status=checkpoint.result.reconciliation_status,
            reason=checkpoint.result.reason,
        )


def _validate_proposal_for_apply(
    proposal: OptimizationProposal,
    current: RuntimeConfigurationSnapshot,
    *,
    trigger_mode: RuntimeConfigurationActivationMode,
) -> None:
    if proposal.scope != _PHASE_4B_SCOPE:
        raise ValueError("Optimization Proposal scope is not enabled in Phase 4-B")
    if proposal.target_key is not OptimizationTargetKey.PLANNER_MAX_NODES:
        raise ValueError("Only planner.max_nodes may be applied in Phase 4-B")
    if proposal.status is not OptimizationProposalStatus.ACTIVE:
        raise ValueError("Optimization Proposal was revoked")
    if proposal.current_configuration_fingerprint is None:
        raise ValueError("Optimization Proposal has no configuration baseline")
    if (
        proposal.current_configuration_revision != current.revision
        or proposal.current_configuration_fingerprint != current.snapshot_fingerprint
        or proposal.current_value != current.value
        or proposal.current_value_fingerprint != current.value_fingerprint
    ):
        raise ValueError("Optimization Proposal baseline is stale")
    value = proposal.proposed_value
    if isinstance(value, bool) or not isinstance(value, int) or not 8 <= value <= 32:
        raise ValueError("planner.max_nodes must be an integer in the safe range 8..32")
    if trigger_mode is RuntimeConfigurationActivationMode.POLICY_AUTO:
        if proposal.risk_classification.value != "low":
            raise ValueError("Policy-auto Apply requires a low-risk Proposal")
        if proposal.counterevidence_refs:
            raise ValueError("Policy-auto Apply rejects unresolved counterevidence")
        current_value = current.value
        if (
            isinstance(current_value, bool)
            or not isinstance(current_value, int)
            or abs(value - current_value) != 1
        ):
            raise ValueError("Policy-auto Apply requires an exact one-node change")


def _explicit_sources(request: DecisionRequest[Any]) -> ProjectionSources:
    payload = request.payload
    trigger_mode = getattr(payload, "trigger_mode", None)
    return ProjectionSources(
        items=tuple(
            ProjectionSource(
                source_id=f"optimization-configuration-input-{index}",
                source_type=_INPUT_SOURCE_TYPE,
                agent_scope="optimization_configuration_gateway",
                content={
                    "request_id": str(request.request_id),
                    "operation": request.allowed_actions[0],
                    "trigger_mode": (
                        trigger_mode.value
                        if isinstance(
                            trigger_mode, RuntimeConfigurationActivationMode
                        )
                        else RuntimeConfigurationActivationMode.MANUAL_APPLY.value
                    ),
                    "evidence_kind": evidence.kind,
                },
                sensitivity=ContextSensitivity.INTERNAL,
                evidence_id=evidence.evidence_id,
                priority=100 - index,
                estimated_tokens=32,
            )
            for index, evidence in enumerate(request.evidence)
        )
    )


def _explicit_policy(decision_type: str) -> ContextProjectionPolicy:
    return ContextProjectionPolicy(
        policy_id="research.optimization_configuration.request",
        version="1",
        agent_scope="optimization_configuration_gateway",
        allowed_decision_types=frozenset({decision_type}),
        allowed_source_types=frozenset({_INPUT_SOURCE_TYPE}),
        allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
        redact_keys=frozenset({"authorization", "credential", "secret", "token"}),
        max_items=2,
        max_context_tokens=128,
    )


def _receipt_result(receipt: OptimizationApplyCommitReceipt) -> JsonValue:
    return {
        "effect_fingerprint": receipt.effect_fingerprint,
        "operation": receipt.operation,
        "active_revision": receipt.active_revision,
        "active_snapshot_id": str(receipt.active_snapshot_id),
        "active_snapshot_fingerprint": receipt.active_snapshot_fingerprint,
        "activation_mode": (
            receipt.activation_mode.value
            if receipt.activation_mode is not None
            else None
        ),
        "trigger_run_id": (
            str(receipt.trigger_run_id)
            if receipt.trigger_run_id is not None
            else None
        ),
        "source_proposal_id": (
            str(receipt.source_proposal_id)
            if receipt.source_proposal_id is not None
            else None
        ),
        "auto_adaptation_policy_fingerprint": (
            receipt.auto_adaptation_policy_fingerprint
        ),
    }


def _governance_record(
    recorder: RecordingGovernanceEvaluator,
    issuer: RecordingAuthorizationIssuer,
) -> GovernanceRecord | None:
    if (
        recorder.request is None
        or recorder.preliminary is None
        or recorder.final is None
    ):
        return None
    return GovernanceRecord(
        scenario="optimization_configuration",
        request=recorder.request,
        preliminary=recorder.preliminary,
        final=recorder.final,
        authorization=issuer.authorization,
        review=recorder.review,
    )


def research_phase_4b_optimization_scope() -> OptimizationScope:
    return _PHASE_4B_SCOPE

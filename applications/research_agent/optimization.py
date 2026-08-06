"""Governed Phase 4-A Optimization Proposal composition.

The handler consumes only persisted Phase 3 evidence, asks an isolated Agent
for an authority-free Draft, and commits an immutable Proposal.  It has no
configuration, deployment, activation, or rollback capability.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol
from uuid import UUID

from pydantic import JsonValue

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
from adaptive_agent_runtime.llm import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CapabilityTurnResult,
    InferenceCorrelation,
)
from adaptive_agent_runtime.optimization import (
    OPTIMIZATION_ASSESSMENT_DECISION_TYPE,
    OPTIMIZATION_PROPOSAL_COMMIT_OPERATION,
    OptimizationAssessmentAgentRequest,
    OptimizationAssessmentRequest,
    OptimizationBaselineProvider,
    OptimizationEvidenceAgentView,
    OptimizationEvidenceResolver,
    OptimizationProposal,
    OptimizationProposalDraft,
    OptimizationProposalEffect,
    OptimizationProposalStore,
    OptimizationRiskClassification,
    OptimizationScope,
    OptimizationTarget,
    OptimizationTargetAgentView,
    OptimizationTargetConstraints,
    OptimizationTargetKey,
    OptimizationTargetType,
    RuntimeConfigurationQuery,
    stable_optimization_proposal_id,
    stable_optimization_request_id,
    stable_optimization_target_ref,
    validate_optimization_value,
)
from adaptive_agent_runtime.persistence import default_runtime_configuration_snapshot

from applications.research_agent.decision_support import (
    RecordingAuthorizationIssuer,
    RecordingGovernanceEvaluator,
)
from applications.research_agent.planning import RESEARCH_PLANNING_GRAPH_LIMITS
from applications.research_agent.report import GovernanceRecord


OPTIMIZATION_ASSESSMENT_INPUT_SOURCE_TYPE = "optimization_assessment_input"


class OptimizationAssessmentCapability(Protocol):
    module_id: str
    capability_id: str

    async def assess_optimization(
        self,
        request: OptimizationAssessmentAgentRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[OptimizationProposalDraft]: ...


class DeterministicOptimizationAssessmentCapability:
    """Offline Agent substitute that still has proposal-only authority."""

    module_id = "research_agent.optimization_assessment.deterministic"
    capability_id = "optimization_assessment.deterministic"

    async def assess_optimization(
        self,
        request: OptimizationAssessmentAgentRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[OptimizationProposalDraft]:
        del invocation
        target = request.targets[0]
        allowed = target.constraints.allowed_values
        if not allowed:
            raise RuntimeError("Optimization target has no bounded proposal values")
        evidence = request.evidence
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=OptimizationProposalDraft(
                target_ref=target.target_ref,
                proposed_value=allowed[0],
                expected_impact=(
                    "Bound Initial Planning graph complexity under the observed "
                    "cross-run conditions; this remains unvalidated until Phase 4-B."
                ),
                applicable_conditions=tuple(
                    dict.fromkeys(
                        condition
                        for item in evidence
                        for condition in item.applicable_conditions
                    )
                ),
                limitations=tuple(
                    dict.fromkeys(
                        (
                            *(
                                limitation
                                for item in evidence
                                for limitation in item.limitations
                            ),
                            "The Proposal is not an instruction to change Runtime behavior.",
                        )
                    )
                ),
                supporting_candidate_refs=tuple(
                    item.candidate_ref for item in evidence
                ),
                counterevidence_candidate_refs=(),
            ),
        )


class ResearchPlanningOptimizationBaselineProvider(OptimizationBaselineProvider):
    """Read the deterministic policy actually used by Initial Planning."""

    module_id = "research_agent.optimization_baseline.planning"

    def __init__(
        self,
        configurations: RuntimeConfigurationQuery | None = None,
    ) -> None:
        self._configurations = configurations

    async def targets(
        self,
        scope: OptimizationScope,
    ) -> tuple[OptimizationTarget, ...]:
        if scope != research_initial_planning_optimization_scope():
            return ()
        configurations = getattr(self, "_configurations", None)
        active = (
            await configurations.load_active(
                scope, OptimizationTargetKey.PLANNER_MAX_NODES
            )
            if configurations is not None
            else None
        )
        max_nodes_snapshot = active or default_runtime_configuration_snapshot(scope)
        max_nodes_allowed_values = tuple(
            value
            for value in (9, 10, 12, 16)
            if value != max_nodes_snapshot.value
        )
        values = (
            (
                OptimizationTargetKey.PLANNER_MAX_NODES,
                max_nodes_snapshot.value,
                max_nodes_allowed_values,
                "Maximum node count accepted by the Initial Planning validator.",
                max_nodes_snapshot.revision,
                max_nodes_snapshot.snapshot_fingerprint,
                max_nodes_snapshot.configuration_source,
            ),
            (
                OptimizationTargetKey.PLANNER_MAX_DEPTH,
                RESEARCH_PLANNING_GRAPH_LIMITS.max_depth,
                (4, 6, 7),
                "Maximum dependency depth accepted by the Initial Planning validator.",
                0,
                None,
                (
                    "applications.research_agent.planning."
                    "RESEARCH_PLANNING_GRAPH_LIMITS"
                ),
            ),
        )
        return tuple(
            OptimizationTarget(
                target_ref=stable_optimization_target_ref(scope, key),
                target_type=OptimizationTargetType.INITIAL_PLANNING_POLICY,
                target_key=key,
                current_value=current,
                current_value_fingerprint=decision_fingerprint(current),
                current_configuration_revision=configuration_revision,
                current_configuration_fingerprint=configuration_fingerprint,
                configuration_source=configuration_source,
                constraints=OptimizationTargetConstraints(
                    value_type="integer",
                    minimum=(
                        8
                        if key is OptimizationTargetKey.PLANNER_MAX_NODES
                        else 1
                    ),
                    maximum=32,
                    allowed_values=allowed,
                ),
            )
            for (
                key,
                current,
                allowed,
                _description,
                configuration_revision,
                configuration_fingerprint,
                configuration_source,
            ) in values
        )


@dataclass(frozen=True)
class ResearchOptimizationAssessmentResult:
    proposal: OptimizationProposal | None
    governance_record: GovernanceRecord | None = None
    request_id: UUID | None = None
    draft: OptimizationProposalDraft | None = None
    eligible_candidate_count: int = 0


class _OptimizationProposalProducer:
    module_id = "research_agent.optimization.proposal_producer"

    def __init__(
        self,
        capability: OptimizationAssessmentCapability,
        *,
        timeout_seconds: float,
        correlation: InferenceCorrelation,
    ) -> None:
        self._capability = capability
        self._timeout_seconds = timeout_seconds
        self._correlation = correlation

    async def propose(
        self,
        context: AgentContext,
    ) -> AgentCallResult[OptimizationProposalDraft]:
        blocks = tuple(
            item
            for item in context.blocks
            if item.source_type == OPTIMIZATION_ASSESSMENT_INPUT_SOURCE_TYPE
        )
        if len(blocks) != 1 or not isinstance(blocks[0].content, Mapping):
            raise ValueError("Optimization Agent requires one isolated evidence block")
        request = OptimizationAssessmentAgentRequest.model_validate(
            dict(blocks[0].content)
        )
        started = monotonic()
        turn = await asyncio.wait_for(
            self._capability.assess_optimization(
                request,
                invocation=CapabilityInvocationMetadata(
                    correlation=self._correlation,
                    trace_attributes={"operation": "optimization.proposal.assess"},
                ),
            ),
            timeout=self._timeout_seconds,
        )
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            raise RuntimeError("Optimization Assessment cannot return ToolIntents")
        if turn.result is None:
            raise RuntimeError("Optimization Assessment returned no Draft")
        draft = turn.result
        return AgentCallResult(
            proposal=DecisionProposal(
                request_id=context.request_id,
                proposal_type=context.decision_type,
                producer=DecisionProducer(
                    producer_id=self._capability.capability_id,
                    capability="optimization_assessment",
                    implementation_version="phase-4a",
                ),
                input_snapshot_fingerprint=context.basis_fingerprint,
                context_fingerprint=context.context_fingerprint,
                selected_action=OPTIMIZATION_PROPOSAL_COMMIT_OPERATION,
                payload=draft,
                rationale="Semantic proposal over Runtime-verified Phase 3 evidence.",
                # Candidate refs are validated against the typed isolated block.
                # Decision evidence ids remain empty because one projected block
                # intentionally contains the bounded candidate set as a whole.
                evidence_refs=(),
                confidence=0.5,
            ),
            elapsed_seconds=monotonic() - started,
        )


class _OptimizationEffectNormalizer:
    module_id = "research_agent.optimization.effect_normalizer"

    def normalize(
        self,
        request: DecisionRequest[OptimizationAssessmentRequest],
        proposal: DecisionProposal[OptimizationProposalDraft],
    ) -> NormalizedDecisionEffect[OptimizationProposalEffect]:
        source = request.payload
        draft = proposal.payload
        targets = {item.target_ref: item for item in source.targets}
        target = targets.get(draft.target_ref)
        if target is None:
            raise ValueError("Optimization Agent selected a target outside Runtime set")
        validate_optimization_value(target, draft.proposed_value)
        candidates = {item.candidate_ref: item for item in source.candidates}
        selected = (
            *draft.supporting_candidate_refs,
            *draft.counterevidence_candidate_refs,
        )
        if any(item not in candidates for item in selected):
            raise ValueError(
                "Optimization Agent selected evidence outside Runtime candidates"
            )
        if not draft.supporting_candidate_refs:
            raise ValueError("Optimization Proposal requires Learning Insight evidence")
        effect = OptimizationProposalEffect(
            proposal_id=stable_optimization_proposal_id(request.request_id),
            scope=source.scope,
            target=target,
            proposed_value=draft.proposed_value,
            evidence_snapshot=source.candidates,
            supporting_candidate_refs=draft.supporting_candidate_refs,
            counterevidence_candidate_refs=draft.counterevidence_candidate_refs,
            expected_impact=draft.expected_impact,
            applicable_conditions=draft.applicable_conditions,
            limitations=draft.limitations,
            risk_classification=OptimizationRiskClassification.LOW,
            rollback_requirements=(
                "Future Apply must preserve and verify this exact baseline.",
                "Future Apply must provide replay validation and atomic rollback.",
            ),
            evidence_set_fingerprint=source.evidence_set_fingerprint,
            proposal_fingerprint=decision_fingerprint(draft),
            basis_fingerprint=request.basis.snapshot_fingerprint,
        )
        return NormalizedDecisionEffect[OptimizationProposalEffect].create(
            payload=effect,
            operation=OPTIMIZATION_PROPOSAL_COMMIT_OPERATION,
            target=request.target,
            governance_scope=DecisionGovernanceScope.EVOLUTION,
            risk=DecisionRiskLevel.LOW,
            impact_score=0.05,
            reversible=False,
            impact_description=(
                "Append an immutable Optimization Proposal without changing Runtime."
            ),
        )


class _OptimizationBasisProvider:
    module_id = "research_agent.optimization.basis"

    def __init__(self, baseline_provider: OptimizationBaselineProvider) -> None:
        self._baseline_provider = baseline_provider

    async def current_basis(
        self,
        request: DecisionRequest[Any],
    ) -> DecisionBasis:
        payload = OptimizationAssessmentRequest.model_validate(request.payload)
        current_targets = await self._baseline_provider.targets(payload.scope)
        current_by_key = {item.target_key: item for item in current_targets}
        rebound = tuple(
            current_by_key.get(item.target_key, item) for item in payload.targets
        )
        current_payload = payload.model_copy(update={"targets": rebound})
        return DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(current_payload),
            configuration_revision=request.basis.configuration_revision,
        )


class ResearchOptimizationProposalHandler:
    module_id = "research_agent.optimization_proposal"

    def __init__(
        self,
        *,
        evidence_resolver: OptimizationEvidenceResolver,
        baseline_provider: OptimizationBaselineProvider,
        store: OptimizationProposalStore,
        capability: OptimizationAssessmentCapability,
        governance: GovernanceEvaluator,
        reviews: HumanReviewService,
        issuer: GovernanceAuthorizationIssuer,
        operation_executor: GovernedOperationExecutor,
        trace_sink: TraceSink,
        checkpoint_store: DecisionCheckpointStore[
            OptimizationAssessmentRequest,
            OptimizationProposalDraft,
            OptimizationProposalEffect,
        ]
        | None = None,
        fault_injector: Callable[[DecisionFaultPoint, Any], None] | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._evidence_resolver = evidence_resolver
        self._baseline_provider = baseline_provider
        self._store = store
        self._capability = capability
        self._governance = governance
        self._reviews = reviews
        self._issuer = issuer
        self._operation_executor = operation_executor
        self._trace_sink = trace_sink
        self._checkpoints = checkpoint_store or InMemoryDecisionCheckpointStore()
        self._fault_injector = fault_injector
        self._timeout_seconds = timeout_seconds

    async def assess(
        self,
        *,
        trigger_run_id: UUID,
        task_id: UUID,
        scope: OptimizationScope,
    ) -> ResearchOptimizationAssessmentResult:
        request_id = stable_optimization_request_id(trigger_run_id, scope)
        existing = await self._checkpoints.load(request_id)
        if existing is not None:
            coordinator, recorder, issuer = self._coordinator(existing.request)
            checkpoint = await coordinator.resume(request_id)
            return await self._result(checkpoint, recorder, issuer)

        candidates = await self._evidence_resolver.resolve(scope)
        if not candidates:
            return ResearchOptimizationAssessmentResult(
                proposal=None,
                eligible_candidate_count=0,
            )
        targets = await self._baseline_provider.targets(scope)
        if not targets:
            return ResearchOptimizationAssessmentResult(
                proposal=None,
                eligible_candidate_count=len(candidates),
            )
        payload = OptimizationAssessmentRequest(
            scope=scope,
            trigger_run_id=trigger_run_id,
            targets=targets,
            candidates=candidates,
            evidence_set_fingerprint=decision_fingerprint(
                tuple(item.candidate_fingerprint for item in candidates)
            ),
        )
        basis = DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(payload),
            configuration_revision=1,
        )
        proposal_id = stable_optimization_proposal_id(request_id)
        request = DecisionRequest(
            request_id=request_id,
            decision_type=OPTIMIZATION_ASSESSMENT_DECISION_TYPE,
            target=DecisionTarget(
                target_type="optimization_proposal",
                target_id=str(proposal_id),
            ),
            correlation=DecisionCorrelation(run_id=trigger_run_id, task_id=task_id),
            basis=basis,
            payload=payload,
            allowed_actions=(OPTIMIZATION_PROPOSAL_COMMIT_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="optimization-proposal-only",
                    description=(
                        "No Apply, activation, rollback, deployment, or Runtime mutation."
                    ),
                ),
                DecisionConstraint(
                    constraint_id="optimization-bounded-targets",
                    description="Select one Runtime-provided bounded target only.",
                ),
            ),
            evidence=tuple(
                DecisionEvidenceReference(
                    evidence_id=item.candidate_ref,
                    kind="runtime.verified.learning_insight",
                    source="optimization_evidence_resolver",
                    reliability=1.0,
                    summary="Verified Phase 3 Learning evidence with provenance.",
                )
                for item in candidates
            ),
            budget=DecisionBudget(
                max_decision_cycles=1,
                max_agent_calls=1,
                max_revision_count=0,
                max_elapsed_seconds=self._timeout_seconds,
            ),
        )
        agent_request = OptimizationAssessmentAgentRequest(
            targets=tuple(
                OptimizationTargetAgentView(
                    target_ref=item.target_ref,
                    description=(
                        "A bounded Initial Planning structural constraint candidate."
                    ),
                    constraints=item.constraints,
                )
                for item in targets
            ),
            evidence=tuple(
                OptimizationEvidenceAgentView(
                    candidate_ref=item.candidate_ref,
                    observed_pattern=item.observed_pattern,
                    applicable_conditions=item.applicable_conditions,
                    limitations=item.limitations,
                    independent_run_count=len(set(item.source_run_refs)),
                    has_counterevidence=bool(item.counterevidence_feedback_refs),
                )
                for item in candidates
            ),
        )
        sources = ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="optimization-evidence-set",
                    source_type=OPTIMIZATION_ASSESSMENT_INPUT_SOURCE_TYPE,
                    agent_scope="optimization_assessment",
                    content=agent_request.model_dump(mode="json"),
                    sensitivity=ContextSensitivity.INTERNAL,
                    priority=100,
                    estimated_tokens=1536,
                ),
            )
        )
        policy = ContextProjectionPolicy(
            policy_id="research.optimization_assessment.context",
            version="1",
            agent_scope="optimization_assessment",
            allowed_decision_types=frozenset(
                {OPTIMIZATION_ASSESSMENT_DECISION_TYPE}
            ),
            allowed_source_types=frozenset(
                {OPTIMIZATION_ASSESSMENT_INPUT_SOURCE_TYPE}
            ),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            redact_keys=frozenset(
                {
                    "authorization",
                    "configuration_source",
                    "credential",
                    "current_value",
                    "database",
                    "effect_fingerprint",
                    "password",
                    "scope",
                    "secret",
                    "token",
                    "trace",
                }
            ),
            max_items=1,
            max_context_tokens=2048,
        )
        coordinator, recorder, issuer = self._coordinator(request)
        checkpoint = await coordinator.run(request, sources=sources, policy=policy)
        return await self._result(checkpoint, recorder, issuer)

    def _coordinator(
        self,
        request: DecisionRequest[OptimizationAssessmentRequest],
    ) -> tuple[
        DecisionLifecycleCoordinator[
            OptimizationAssessmentRequest,
            OptimizationProposalDraft,
            OptimizationProposalEffect,
            DecisionGovernanceBinding,
        ],
        RecordingGovernanceEvaluator,
        RecordingAuthorizationIssuer,
    ]:
        recorder = RecordingGovernanceEvaluator(self._governance)
        issuer = RecordingAuthorizationIssuer(self._issuer)

        async def forbidden_apply(effect: OptimizationProposalEffect) -> JsonValue:
            del effect
            raise RuntimeError("Optimization Proposal commit requires a Runtime Permit")

        async def apply_authorized(
            normalized: NormalizedDecisionEffect[OptimizationProposalEffect],
            permit: RuntimeCommitPermit,
        ) -> JsonValue:
            current_targets = await self._baseline_provider.targets(
                normalized.payload.scope
            )
            current = {
                item.target_key: item for item in current_targets
            }.get(normalized.payload.target.target_key)
            if current != normalized.payload.target:
                raise RuntimeError("Optimization Proposal baseline became stale")
            proposal = await self._store.commit(
                normalized.payload,
                source_decision_request_id=request.request_id,
                effect_fingerprint=normalized.effect_fingerprint,
                permit=permit,
                target=GovernanceTarget(
                    target_type=normalized.target.target_type,
                    target_id=normalized.target.target_id,
                ),
                subject_fingerprint=governance_fingerprint(normalized),
            )
            readback = await self._store.load_by_effect(
                normalized.effect_fingerprint
            )
            if readback is None or readback != proposal:
                raise RuntimeError("Optimization Proposal commit read-back failed")
            return self._proposal_result(readback)

        async def reconcile(
            normalized: NormalizedDecisionEffect[OptimizationProposalEffect],
        ) -> DecisionReconciliation:
            try:
                proposal = await self._store.load_by_effect(
                    normalized.effect_fingerprint
                )
            except Exception as exc:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.UNKNOWN,
                    reason=(
                        "Optimization Proposal read-back is indeterminate: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            if proposal is None:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.NOT_COMMITTED,
                    reason="Optimization Proposal has not been committed",
                )
            effect = normalized.payload
            if (
                proposal.proposal_id != effect.proposal_id
                or proposal.current_value_fingerprint
                != effect.target.current_value_fingerprint
                or proposal.evidence_set_fingerprint
                != effect.evidence_set_fingerprint
                or proposal.effect_fingerprint != normalized.effect_fingerprint
            ):
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.UNKNOWN,
                    reason="Stored Optimization Proposal conflicts with authorized Effect",
                )
            result = self._proposal_result(proposal)
            return DecisionReconciliation(
                status=DecisionReconciliationStatus.COMMITTED,
                reason="Optimization Proposal was committed and read back",
                apply_receipt=DecisionApplyReceipt(
                    effect_fingerprint=normalized.effect_fingerprint,
                    committed_state_fingerprint=decision_fingerprint(result),
                    result=result,
                ),
            )

        coordinator = DecisionLifecycleCoordinator[
            OptimizationAssessmentRequest,
            OptimizationProposalDraft,
            OptimizationProposalEffect,
            DecisionGovernanceBinding,
        ](
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=_OptimizationProposalProducer(
                self._capability,
                timeout_seconds=self._timeout_seconds,
                correlation=InferenceCorrelation(
                    run_id=request.correlation.run_id,
                    task_id=request.correlation.task_id,
                ),
            ),
            basis_provider=_OptimizationBasisProvider(self._baseline_provider),
            validator=RuntimeDecisionValidator(
                normalizer=_OptimizationEffectNormalizer()
            ),
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
            checkpoint_store=self._checkpoints,
            checkpoint_type=DecisionCheckpoint[
                OptimizationAssessmentRequest,
                OptimizationProposalDraft,
                OptimizationProposalEffect,
            ],
            trace_writer=RuntimeDecisionTraceWriter(self._trace_sink),
            fault_injector=self._fault_injector,
        )
        return coordinator, recorder, issuer

    async def _result(
        self,
        checkpoint: DecisionCheckpoint[
            OptimizationAssessmentRequest,
            OptimizationProposalDraft,
            OptimizationProposalEffect,
        ],
        recorder: RecordingGovernanceEvaluator,
        issuer: RecordingAuthorizationIssuer,
    ) -> ResearchOptimizationAssessmentResult:
        if checkpoint.stage is DecisionCheckpointStage.REVIEW_PENDING:
            return ResearchOptimizationAssessmentResult(
                proposal=None,
                request_id=checkpoint.request_id,
                draft=checkpoint.proposal.payload if checkpoint.proposal else None,
                eligible_candidate_count=len(checkpoint.request.payload.candidates),
            )
        if (
            checkpoint.result is None
            or checkpoint.result.status is not DecisionResultStatus.APPLIED
            or checkpoint.validated_decision is None
        ):
            reason = checkpoint.result.reason if checkpoint.result else checkpoint.stage
            raise RuntimeError(f"Optimization Assessment did not apply: {reason}")
        normalized = checkpoint.validated_decision.normalized_effect
        proposal = await self._store.load_by_effect(normalized.effect_fingerprint)
        if proposal is None:
            raise RuntimeError("APPLIED Optimization Decision has no stored Proposal")
        governance_record = (
            GovernanceRecord(
                scenario="optimization_proposal",
                request=recorder.request,
                preliminary=recorder.preliminary,
                final=recorder.final,
                authorization=issuer.authorization,
                review=recorder.review,
            )
            if recorder.request is not None
            and recorder.preliminary is not None
            and recorder.final is not None
            else None
        )
        return ResearchOptimizationAssessmentResult(
            proposal=proposal,
            governance_record=governance_record,
            request_id=checkpoint.request_id,
            draft=checkpoint.proposal.payload if checkpoint.proposal else None,
            eligible_candidate_count=len(checkpoint.request.payload.candidates),
        )

    @staticmethod
    def _proposal_result(proposal: OptimizationProposal) -> JsonValue:
        return {
            "proposal_id": str(proposal.proposal_id),
            "effect_fingerprint": proposal.effect_fingerprint,
            "proposal_fingerprint": decision_fingerprint(proposal),
            "scope": proposal.scope.model_dump(mode="json"),
            "target_key": proposal.target_key.value,
        }


def research_initial_planning_optimization_scope() -> OptimizationScope:
    return OptimizationScope(
        tenant="default",
        project="research",
        application="research_agent",
        decision_type="planning.task_graph.initialize",
    )

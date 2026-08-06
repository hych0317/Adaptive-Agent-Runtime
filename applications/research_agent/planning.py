"""Research Application composition for governed Adaptive Planning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4
from pydantic import JsonValue

from adaptive_agent_runtime import AgentTask, TraceSink
from adaptive_agent_runtime.context_memory import MemoryRecallBundle
from adaptive_agent_runtime.decisioning import (
    ContextProjectionPolicy,
    ContextSensitivity,
    DecisionBasis,
    DecisionBudget,
    DecisionCheckpoint,
    DecisionCheckpointStore,
    DecisionCheckpointStage,
    DecisionConstraint,
    DecisionCorrelation,
    DecisionEvidenceReference,
    DecisionLifecycleCoordinator,
    DecisionRequest,
    DecisionApplyReceipt,
    DecisionReconciliation,
    DecisionReconciliationStatus,
    DecisionResultStatus,
    DecisionTarget,
    InMemoryDecisionCheckpointStore,
    PolicyAgentContextBuilder,
    ProjectionSource,
    ProjectionSources,
    RuntimeDecisionTraceWriter,
    RuntimeDecisionValidator,
    NormalizedDecisionEffect,
    decision_fingerprint,
)
from adaptive_agent_runtime.governance import (
    DecisionOutcome,
    DecisionGovernanceBinding,
    GovernanceAuthorization,
    GovernanceAuthorizationIssuer,
    GovernanceDecision,
    GovernanceEvaluator,
    GovernanceTarget,
    GovernanceRequest,
    GovernedDecisionApplier,
    GovernedOperationExecutor,
    HumanReviewDecision,
    HumanReviewService,
    ReviewOutcome,
    ReviewRequest,
    RuntimeDecisionGovernanceAdapter,
    RuntimeCommitPermit,
    CommitPermitValidation,
    governance_fingerprint,
)
from adaptive_agent_runtime.llm import (
    PLANNING_INPUT_SOURCE_TYPE,
    PlannerDecisionProposalProducer,
    PlanningGraphEffectNormalizer,
    TaskGraphDraft,
    TaskGraphDraftAdapter,
    TaskGraphProposalCapability,
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CapabilityTurnResult,
    TaskPlanningRequest,
)
from adaptive_agent_runtime.llm import InferenceCorrelation
from adaptive_agent_runtime.orchestration import (
    PLANNING_DECISION_TYPE,
    GraphInitializationApplier,
    InMemoryTaskGraphStore,
    PlanningDecisionPayload,
    PlanningExecutionPolicy,
    PlanningGraphEffect,
    PlanningGraphLimits,
    PlanningStrategyDescriptor,
    PlanningStrategyRisk,
    RequiredPreparedTaskGraphStore,
    TaskGraphStore,
)
from adaptive_agent_runtime.optimization import (
    OptimizationTargetKey,
    RuntimeConfigurationSnapshot,
)

from applications.research_agent.capabilities import (
    CALCULATION,
    DOCUMENT_ANALYSIS,
    INFORMATION_RETRIEVAL,
)
from applications.research_agent.prompts import CHINESE_OUTPUT_INSTRUCTION
from applications.research_agent.report import GovernanceRecord
from applications.research_agent.tasks import (
    REPORT_STRATEGY_ID,
    RESEARCH_STRATEGY_ID,
    REVIEW_STRATEGY_ID,
    ResearchTaskDefinition,
    build_research_task_from_effect,
    build_research_task_draft,
    validate_research_task_graph_draft,
    validate_deferred_news_research_task_graph_draft,
)


RESEARCH_PLANNING_GRAPH_LIMITS = PlanningGraphLimits(
    max_nodes=8,
    max_depth=5,
    max_fan_out=5,
)


@dataclass(frozen=True)
class ResearchAdaptivePlanningResult:
    definition: ResearchTaskDefinition
    draft: TaskGraphDraft
    governance_record: GovernanceRecord
    graph_store: TaskGraphStore
    configuration_snapshot: RuntimeConfigurationSnapshot


class DeterministicResearchPlanningCapability:
    """Deterministic fallback that still has proposal-only authority."""

    module_id = "research_agent.planning.deterministic_fallback"
    capability_id = "planning.deterministic_fallback"

    def __init__(self, company: str) -> None:
        self._company = company

    async def propose(
        self,
        request: TaskPlanningRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[TaskGraphDraft]:
        del request, invocation
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=build_research_task_draft(
                self._company,
                defer_news=True,
            ),
        )


class _FixedPlanningBasisProvider:
    module_id = "research_agent.planning.basis"

    def __init__(self, basis: DecisionBasis) -> None:
        self._basis = basis

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        del request
        return self._basis


class _RecordingGovernanceEvaluator:
    module_id = "research_agent.planning.governance_recorder"

    def __init__(self, delegate: GovernanceEvaluator) -> None:
        self._delegate = delegate
        self.request: GovernanceRequest | None = None
        self.preliminary: GovernanceDecision | None = None
        self.final: GovernanceDecision | None = None
        self.review: ReviewRequest | None = None

    def evaluate(self, request: GovernanceRequest) -> GovernanceDecision:
        decision = self._delegate.evaluate(request)
        self.request = request
        self.preliminary = decision
        self.final = decision
        return decision

    def finalize_review(
        self,
        request: GovernanceRequest,
        review: ReviewRequest,
    ) -> GovernanceDecision:
        decision = self._delegate.finalize_review(request, review)
        self.review = review
        self.final = decision
        return decision


class _RecordingAuthorizationIssuer:
    module_id = "research_agent.planning.authorization_recorder"

    def __init__(self, delegate: GovernanceAuthorizationIssuer) -> None:
        self._delegate = delegate
        self.authorization: GovernanceAuthorization | None = None

    def issue(
        self,
        request: GovernanceRequest,
        decision: GovernanceDecision,
    ) -> GovernanceAuthorization:
        authorization = self._delegate.issue(request, decision)
        self.authorization = authorization
        return authorization


async def _reconcile_planning_effect(
    applier: GraphInitializationApplier,
    normalized: NormalizedDecisionEffect[PlanningGraphEffect],
) -> DecisionReconciliation:
    effect = normalized.payload
    checkpoint = await applier.load_effect(
        run_id=effect.run_id,
        effect_fingerprint=normalized.effect_fingerprint,
    )
    if checkpoint is None:
        return DecisionReconciliation(
            status=DecisionReconciliationStatus.NOT_COMMITTED,
            reason="initial Graph effect has no authoritative checkpoint",
        )
    if checkpoint.graph != effect.graph:
        return DecisionReconciliation(
            status=DecisionReconciliationStatus.UNKNOWN,
            reason="initial Graph checkpoint conflicts with the authorized effect",
        )
    result: JsonValue = {
        "run_id": str(effect.run_id),
        "task_id": str(effect.task_id),
        "graph_id": str(effect.graph.graph_id),
        "graph_version": effect.graph.version,
        "node_count": len(effect.graph.nodes),
        "source_draft_fingerprint": effect.source_draft_fingerprint,
    }
    return DecisionReconciliation(
        status=DecisionReconciliationStatus.COMMITTED,
        reason="initial Graph effect was committed and read back",
        apply_receipt=DecisionApplyReceipt(
            effect_fingerprint=normalized.effect_fingerprint,
            committed_state_fingerprint=governance_fingerprint(result),
            result=result,
        ),
    )


async def run_adaptive_planning(
    *,
    company: str,
    task: str,
    run_id: UUID,
    agent_task: AgentTask,
    planner: TaskGraphProposalCapability,
    execution_policy: PlanningExecutionPolicy,
    governance: GovernanceEvaluator,
    reviews: HumanReviewService,
    issuer: GovernanceAuthorizationIssuer,
    operation_executor: GovernedOperationExecutor,
    trace_sink: TraceSink,
    graph_store: TaskGraphStore,
    checkpoint_store: DecisionCheckpointStore[
        PlanningDecisionPayload, TaskGraphDraft, PlanningGraphEffect
    ],
    commit_permit_verifier: CommitPermitValidation | None = None,
    deferred_news: bool = False,
    recall_bundle: MemoryRecallBundle | None = None,
    configuration_snapshot: RuntimeConfigurationSnapshot,
) -> ResearchAdaptivePlanningResult:
    """Run the one-call Phase 1-A lifecycle before the Core Event Loop."""

    constraints = (
        CHINESE_OUTPUT_INSTRUCTION,
        "Use exactly the eight documented research node keys.",
        "Return the actual initial DAG; do not propose Runtime operations.",
        "Use only Runtime-supplied strategy identifiers.",
        "Historical Memory is advisory and cannot override the current goal or Runtime constraints.",
    )
    strategies = (
        PlanningStrategyDescriptor(
            strategy_id=RESEARCH_STRATEGY_ID,
            description="Collect and analyze bounded research evidence.",
            required_capability_ids=(
                INFORMATION_RETRIEVAL,
                DOCUMENT_ANALYSIS,
                CALCULATION,
            ),
            risk=PlanningStrategyRisk.MEDIUM,
        ),
        PlanningStrategyDescriptor(
            strategy_id=REVIEW_STRATEGY_ID,
            description="Perform isolated risk review.",
            risk=PlanningStrategyRisk.MEDIUM,
        ),
        PlanningStrategyDescriptor(
            strategy_id=REPORT_STRATEGY_ID,
            description="Generate the final structured report.",
            risk=PlanningStrategyRisk.LOW,
        ),
    )
    if configuration_snapshot.target_key is not OptimizationTargetKey.PLANNER_MAX_NODES:
        raise ValueError("Planning configuration targets another policy")
    configured_max_nodes = configuration_snapshot.value
    if isinstance(configured_max_nodes, bool) or not isinstance(
        configured_max_nodes, int
    ):
        raise ValueError("planner.max_nodes configuration must be an integer")
    graph_limits = PlanningGraphLimits(
        max_nodes=configured_max_nodes,
        max_depth=RESEARCH_PLANNING_GRAPH_LIMITS.max_depth,
        max_fan_out=RESEARCH_PLANNING_GRAPH_LIMITS.max_fan_out,
    )
    payload = PlanningDecisionPayload(
        goal=task,
        constraints=constraints,
        strategies=strategies,
        available_execution_capability_ids=(
            INFORMATION_RETRIEVAL,
            DOCUMENT_ANALYSIS,
            CALCULATION,
        ),
        graph_limits=graph_limits,
        execution_policy=execution_policy,
    )
    basis = DecisionBasis(
        snapshot_fingerprint=decision_fingerprint(
            {
                "task_id": agent_task.task_id,
                "planning_payload": payload,
                "configuration_revision": configuration_snapshot.revision,
                "configuration_fingerprint": (
                    configuration_snapshot.snapshot_fingerprint
                ),
                "configuration_activation_mode": (
                    configuration_snapshot.effective_activation_mode.value
                ),
                "configuration_source_proposal_id": (
                    str(configuration_snapshot.source_proposal_id)
                    if configuration_snapshot.source_proposal_id is not None
                    else None
                ),
                "configuration_trigger_run_id": (
                    str(configuration_snapshot.trigger_run_id)
                    if configuration_snapshot.trigger_run_id is not None
                    else None
                ),
                "auto_adaptation_policy_fingerprint": (
                    configuration_snapshot.auto_adaptation_policy_fingerprint
                ),
                "recall_bundle_fingerprint": (
                    decision_fingerprint(recall_bundle)
                    if recall_bundle is not None
                    else None
                ),
            }
        ),
        configuration_revision=configuration_snapshot.revision,
    )
    planner_action_id = uuid4()
    request = DecisionRequest[PlanningDecisionPayload](
        decision_type=PLANNING_DECISION_TYPE,
        target=DecisionTarget(
            target_type="runtime_run_graph",
            target_id=str(run_id),
        ),
        correlation=DecisionCorrelation(
            run_id=run_id,
            task_id=agent_task.task_id,
            action_id=planner_action_id,
        ),
        basis=basis,
        payload=payload,
        allowed_actions=("graph.initialize",),
        constraints=tuple(
            DecisionConstraint(
                constraint_id=f"planning-{index}",
                description=value,
            )
            for index, value in enumerate(constraints, start=1)
        ),
        evidence=(
            DecisionEvidenceReference(
                evidence_id=f"task:{agent_task.task_id}",
                kind="user.task",
                source="conversation",
                reliability=1.0,
                summary="The planning goal originates from the current user task.",
            ),
            DecisionEvidenceReference(
                evidence_id=f"planning-catalog:{agent_task.task_id}",
                kind="runtime.catalog",
                source="runtime",
                reliability=1.0,
                summary="Runtime supplied the current strategy and capability catalog.",
            ),
            DecisionEvidenceReference(
                evidence_id=(
                    "runtime-configuration:"
                    f"{configuration_snapshot.snapshot_fingerprint}"
                ),
                kind="runtime.configuration_snapshot",
                source="runtime_configuration",
                reliability=1.0,
                summary=(
                    "Planning uses configuration revision "
                    f"{configuration_snapshot.revision} with planner.max_nodes="
                    f"{configured_max_nodes}; origin="
                    f"{configuration_snapshot.effective_activation_mode.value}."
                ),
            ),
            *(
                (
                    DecisionEvidenceReference(
                        evidence_id=f"recall-bundle:{recall_bundle.bundle_id}",
                        kind="memory.recall_bundle",
                        source="memory_runtime",
                        reliability=0.8,
                        summary=(
                            "Committed historical experience; not current factual evidence."
                        ),
                    ),
                )
                if recall_bundle is not None
                else ()
            ),
        ),
        budget=DecisionBudget(
            max_decision_cycles=1,
            max_agent_calls=execution_policy.max_agent_calls,
            max_revision_count=0,
            max_elapsed_seconds=execution_policy.timeout_seconds,
        ),
    )
    sources = ProjectionSources(
        items=(
            ProjectionSource(
                source_id="planning-input",
                source_type=PLANNING_INPUT_SOURCE_TYPE,
                agent_scope="planner",
                content={
                    "goal": task,
                    "constraints": list(constraints),
                    "available_strategies": [
                        item.strategy_id for item in strategies
                    ],
                    "available_execution_capability_ids": [
                        INFORMATION_RETRIEVAL,
                        DOCUMENT_ANALYSIS,
                        CALCULATION,
                    ],
                    "runtime_configuration": {
                        "revision": configuration_snapshot.revision,
                        "fingerprint": configuration_snapshot.snapshot_fingerprint,
                        "planner.max_nodes": configured_max_nodes,
                    },
                    "committed_memory_recall_bundle": (
                        {
                            "section": "Committed Memory Recall Bundle",
                            "bundle_id": str(recall_bundle.bundle_id),
                            "bundle_fingerprint": decision_fingerprint(recall_bundle),
                            "historical_experience": [
                                item.model_dump(mode="json")
                                for item in recall_bundle.items
                            ],
                            "instruction": (
                                "Treat as historical experience; current goal and Runtime "
                                "constraints remain authoritative."
                            ),
                        }
                        if recall_bundle is not None
                        else None
                    ),
                },
                sensitivity=ContextSensitivity.INTERNAL,
                evidence_id=f"task:{agent_task.task_id}",
                priority=100,
                estimated_tokens=(
                    128
                    + (recall_bundle.used_tokens if recall_bundle is not None else 0)
                ),
            ),
            ProjectionSource(
                source_id="planning-catalog",
                source_type="planning_catalog",
                agent_scope="planner",
                content={
                    "strategies": [
                        item.model_dump(mode="json") for item in strategies
                    ]
                },
                sensitivity=ContextSensitivity.INTERNAL,
                evidence_id=f"planning-catalog:{agent_task.task_id}",
                priority=90,
                estimated_tokens=96,
            ),
        )
    )
    context_policy = ContextProjectionPolicy(
        policy_id="research.planner.context",
        version="1",
        agent_scope="planner",
        allowed_decision_types=frozenset({PLANNING_DECISION_TYPE}),
        allowed_source_types=frozenset(
            {PLANNING_INPUT_SOURCE_TYPE, "planning_catalog"}
        ),
        allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
        redact_keys=frozenset(
            {"authorization", "credential", "password", "token"}
        ),
        max_items=2,
        max_context_tokens=(
            512 + (recall_bundle.max_tokens if recall_bundle is not None else 0)
        ),
    )

    graph_applier = GraphInitializationApplier(
        graph_store,
        permit_verifier=commit_permit_verifier,
    )
    recording_evaluator = _RecordingGovernanceEvaluator(governance)
    recording_issuer = _RecordingAuthorizationIssuer(issuer)
    governance_adapter = RuntimeDecisionGovernanceAdapter[
        PlanningDecisionPayload,
        TaskGraphDraft,
        PlanningGraphEffect,
    ](
        evaluator=recording_evaluator,
        authorization_issuer=recording_issuer,
        review_service=reviews,
    )
    coordinator = DecisionLifecycleCoordinator[
        PlanningDecisionPayload,
        TaskGraphDraft,
        PlanningGraphEffect,
        DecisionGovernanceBinding,
    ](
        context_builder=PolicyAgentContextBuilder(),
        proposal_producer=PlannerDecisionProposalProducer(
            capability=planner,
            execution_policy=execution_policy,
            correlation=InferenceCorrelation(
                run_id=run_id,
                task_id=agent_task.task_id,
                action_id=planner_action_id,
            ),
            confidence=0.9,
        ),
        basis_provider=_FixedPlanningBasisProvider(basis),
        validator=RuntimeDecisionValidator(
            normalizer=PlanningGraphEffectNormalizer(
                draft_adapter=TaskGraphDraftAdapter(
                    domain_validator=(
                        validate_deferred_news_research_task_graph_draft
                        if deferred_news
                        else validate_research_task_graph_draft
                    )
                )
            )
        ),
        governance=governance_adapter,
        applier=GovernedDecisionApplier(
            executor=operation_executor,
            apply_effect=graph_applier.apply,
            apply_authorized_effect=lambda normalized, permit: graph_applier.commit(
                normalized.payload,
                effect_fingerprint=normalized.effect_fingerprint,
                permit=permit,
                target=GovernanceTarget(
                    target_type=normalized.target.target_type,
                    target_id=normalized.target.target_id,
                ),
                subject_fingerprint=governance_fingerprint(normalized),
            ),
            reconcile_effect=lambda normalized: _reconcile_planning_effect(
                graph_applier, normalized
            ),
        ),
        checkpoint_store=checkpoint_store,
        checkpoint_type=DecisionCheckpoint[
            PlanningDecisionPayload,
            TaskGraphDraft,
            PlanningGraphEffect,
        ],
        trace_writer=RuntimeDecisionTraceWriter(trace_sink),
    )
    checkpoint = await coordinator.run(
        request,
        sources=sources,
        policy=context_policy,
    )
    if checkpoint.stage is DecisionCheckpointStage.REVIEW_PENDING:
        receipt = checkpoint.governance_receipt
        if receipt is None or receipt.review_request_id is None:
            raise RuntimeError("Planning review checkpoint has no review request")
        review = reviews.resolve(
            receipt.review_request_id,
            HumanReviewDecision(
                outcome=ReviewOutcome.APPROVE,
                reviewer_id="research-demo-reviewer",
                rationale="The initial graph is bounded, reversible, and auditable.",
                decided_at=datetime.now(timezone.utc),
            ),
        )
        recording_evaluator.review = review
        checkpoint = await coordinator.resume_review(request.request_id)

    if (
        checkpoint.result is None
        or checkpoint.result.status is not DecisionResultStatus.APPLIED
        or checkpoint.proposal is None
        or checkpoint.validated_decision is None
    ):
        reason = checkpoint.result.reason if checkpoint.result is not None else checkpoint.stage
        raise RuntimeError(f"Adaptive Planning did not apply: {reason}")
    effect = checkpoint.validated_decision.normalized_effect.payload
    definition = build_research_task_from_effect(
        company,
        effect,
        deferred_news=deferred_news,
    )
    governance_request = recording_evaluator.request
    preliminary = recording_evaluator.preliminary
    final = recording_evaluator.final
    if governance_request is None or preliminary is None or final is None:
        raise RuntimeError("Adaptive Planning Governance record is incomplete")
    record = GovernanceRecord(
        scenario="planning",
        request=governance_request,
        preliminary=preliminary,
        final=final,
        authorization=recording_issuer.authorization,
        review=recording_evaluator.review,
    )
    return ResearchAdaptivePlanningResult(
        definition=definition,
        draft=checkpoint.proposal.payload,
        governance_record=record,
        graph_store=RequiredPreparedTaskGraphStore(
            graph_store,
            run_id=run_id,
            graph_id=effect.graph.graph_id,
        ),
        configuration_snapshot=configuration_snapshot,
    )

"""Research Application composition for governed Recovery decisions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime import TraceSink
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
    governance_fingerprint,
)
from adaptive_agent_runtime.llm import (
    RECOVERY_INPUT_SOURCE_TYPE,
    RecoveryDecisionProposalProducer,
    RecoveryDraft,
    RecoveryEffectNormalizer,
    RecoveryProposalCapability,
    build_recovery_proposal_request,
)
from adaptive_agent_runtime.llm import InferenceCorrelation
from adaptive_agent_runtime.orchestration import (
    RECOVERY_APPLY_OPERATION,
    RECOVERY_DECISION_TYPE,
    AddRecoveryNodeEffect,
    RecoveryActionType,
    RecoveryContext,
    RecoveryDecisionEffect,
    RecoveryDecisionHandler,
    RecoveryDecisionOutcome,
    RecoveryDecisionPayload,
    RecoveryExecutionPolicy,
    RecoveryRecord,
    GraphDecisionCommitter,
    InMemoryGraphDecisionCommitter,
    RecoveryNodeBinding,
    apply_recovery_decision_effect,
    stable_recovery_id,
)

from applications.research_agent.report import GovernanceRecord
from applications.research_agent.strategies import ResearchWorkspace


RECOVERY_EVIDENCE_SOURCE_TYPE = "recovery_evidence"


class RecoveryDiagnosticEvidenceProvider(Protocol):
    async def evidence_for(
        self,
        context: RecoveryContext,
    ) -> tuple[DecisionEvidenceReference, ...]: ...


class _FixedRecoveryBasisProvider:
    module_id = "research_agent.recovery.basis"

    def __init__(self, basis: DecisionBasis) -> None:
        self._basis = basis

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        del request
        return self._basis


class _RecordingGovernanceEvaluator:
    module_id = "research_agent.recovery.governance_recorder"

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
    module_id = "research_agent.recovery.authorization_recorder"

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


class ResearchRecoveryDecisionHandler(RecoveryDecisionHandler):
    """Compose one Recovery Agent decision without becoming another Runtime."""

    module_id = "research_agent.recovery_decision_handler"

    def __init__(
        self,
        *,
        capability: RecoveryProposalCapability,
        execution_policy: RecoveryExecutionPolicy,
        governance: GovernanceEvaluator,
        reviews: HumanReviewService,
        issuer: GovernanceAuthorizationIssuer,
        operation_executor: GovernedOperationExecutor,
        trace_sink: TraceSink,
        workspace: ResearchWorkspace,
        available_strategy_ids: tuple[str, ...],
        diagnostic_evidence_provider: RecoveryDiagnosticEvidenceProvider
        | None = None,
        checkpoint_store: DecisionCheckpointStore[
            RecoveryDecisionPayload,
            RecoveryDraft,
            RecoveryDecisionEffect,
        ]
        | None = None,
    ) -> None:
        self._capability = capability
        self._execution_policy = execution_policy
        self._governance = governance
        self._reviews = reviews
        self._issuer = issuer
        self._operation_executor = operation_executor
        self._trace_sink = trace_sink
        self._workspace = workspace
        self._available_strategy_ids = available_strategy_ids
        self._diagnostic_evidence_provider = diagnostic_evidence_provider
        self._checkpoint_store = (
            checkpoint_store
            or InMemoryDecisionCheckpointStore[
                RecoveryDecisionPayload,
                RecoveryDraft,
                RecoveryDecisionEffect,
            ]()
        )

    async def handle(
        self,
        context: RecoveryContext,
        *,
        committer: GraphDecisionCommitter | None = None,
    ) -> RecoveryDecisionOutcome:
        committer = committer or InMemoryGraphDecisionCommitter()
        request_id = stable_recovery_id(
            "recovery-decision-request",
            context.graph.graph_id,
            context.observation.action_id,
            context.prior_attempts + 1,
        )
        allowed_actions: tuple[RecoveryActionType, ...] = (
            (RecoveryActionType.ABORT,)
            if context.prior_attempts >= self._execution_policy.max_recovery_attempts
            else (
                RecoveryActionType.RETRY_NODE,
                RecoveryActionType.REPLACE_STRATEGY,
                RecoveryActionType.ADD_RECOVERY_NODE,
                RecoveryActionType.REWIRE_DEPENDENCY,
                RecoveryActionType.ABORT,
            )
        )
        payload = RecoveryDecisionPayload(
            task_description=context.state.task.description,
            graph=context.graph,
            failed_node_id=context.failed_node.node_id,
            failed_action_id=context.observation.action_id,
            node_bindings=tuple(
                RecoveryNodeBinding(
                    node_ref=f"node:{node.node_id}",
                    node_id=node.node_id,
                )
                for node in context.graph.nodes
            ),
            available_strategy_ids=self._available_strategy_ids,
            allowed_recovery_actions=allowed_actions,
            prior_attempts=context.prior_attempts,
            execution_policy=self._execution_policy,
        )
        basis = DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(
                {
                    "graph": context.graph,
                    "state_revision": context.state.revision,
                    "failed_action_id": context.observation.action_id,
                    "recovery_attempt": context.prior_attempts + 1,
                }
            ),
            state_revision=context.state.revision,
            graph_version=context.graph.version,
        )
        observation_evidence_id = f"observation:{context.observation.action_id}"
        diagnostic_evidence: tuple[DecisionEvidenceReference, ...] = ()
        if self._diagnostic_evidence_provider is not None:
            try:
                diagnostic_evidence = (
                    await self._diagnostic_evidence_provider.evidence_for(context)
                )
            except Exception:
                # Semantic RCA is advisory; direct Observation evidence is the
                # mandatory Recovery fallback.
                diagnostic_evidence = ()
        evidence = (
            DecisionEvidenceReference(
                evidence_id=observation_evidence_id,
                kind="runtime.observation.failure",
                source="runtime_core",
                reliability=1.0,
                summary=(
                    context.observation.error
                    or "The selected task node returned a failed Observation."
                ),
            ),
            *diagnostic_evidence,
        )
        request = DecisionRequest[RecoveryDecisionPayload](
            request_id=request_id,
            decision_type=RECOVERY_DECISION_TYPE,
            target=DecisionTarget(
                target_type="active_task_graph",
                target_id=str(context.graph.graph_id),
            ),
            correlation=DecisionCorrelation(
                run_id=context.state.run_id,
                task_id=context.state.task.task_id,
                node_id=context.failed_node.node_id,
                action_id=context.observation.action_id,
                decision_cycle=context.prior_attempts,
            ),
            basis=basis,
            payload=payload,
            allowed_actions=(RECOVERY_APPLY_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="recovery-current-graph",
                    description="Use only Runtime-projected node references.",
                ),
                DecisionConstraint(
                    constraint_id="recovery-no-execution",
                    description="Propose one recovery action; do not execute it.",
                ),
            ),
            evidence=evidence,
            budget=DecisionBudget(
                max_decision_cycles=1,
                max_agent_calls=self._execution_policy.max_agent_calls,
                max_revision_count=0,
                max_elapsed_seconds=self._execution_policy.timeout_seconds,
            ),
        )
        projected_request = build_recovery_proposal_request(payload, evidence)
        sources = ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="recovery-input",
                    source_type=RECOVERY_INPUT_SOURCE_TYPE,
                    agent_scope="recovery",
                    content=projected_request.model_dump(mode="json"),
                    sensitivity=ContextSensitivity.INTERNAL,
                    evidence_id=observation_evidence_id,
                    priority=100,
                    estimated_tokens=384,
                ),
                *(
                    ProjectionSource(
                        source_id=f"recovery-evidence:{item.evidence_id}",
                        source_type=RECOVERY_EVIDENCE_SOURCE_TYPE,
                        agent_scope="recovery",
                        content={
                            "reference_id": item.evidence_id,
                            "kind": item.kind,
                            "summary": item.summary,
                            "reliability": item.reliability,
                        },
                        sensitivity=ContextSensitivity.INTERNAL,
                        evidence_id=item.evidence_id,
                        priority=90,
                        estimated_tokens=64,
                    )
                    for item in diagnostic_evidence
                ),
            )
        )
        projection_policy = ContextProjectionPolicy(
            policy_id="research.recovery.context",
            version="1",
            agent_scope="recovery",
            allowed_decision_types=frozenset({RECOVERY_DECISION_TYPE}),
            allowed_source_types=frozenset(
                {RECOVERY_INPUT_SOURCE_TYPE, RECOVERY_EVIDENCE_SOURCE_TYPE}
            ),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            redact_keys=frozenset(
                {"authorization", "credential", "password", "token"}
            ),
            max_items=1 + len(diagnostic_evidence),
            max_context_tokens=512 + (64 * len(diagnostic_evidence)),
        )
        recording_evaluator = _RecordingGovernanceEvaluator(self._governance)
        recording_issuer = _RecordingAuthorizationIssuer(self._issuer)
        governance_adapter = RuntimeDecisionGovernanceAdapter[
            RecoveryDecisionPayload,
            RecoveryDraft,
            RecoveryDecisionEffect,
        ](
            evaluator=recording_evaluator,
            authorization_issuer=recording_issuer,
            review_service=self._reviews,
        )
        applied_graph = None

        async def apply_effect(effect: RecoveryDecisionEffect) -> JsonValue:
            nonlocal applied_graph
            applied_graph = apply_recovery_decision_effect(context.graph, effect)
            if isinstance(effect.effect, AddRecoveryNodeEffect):
                self._workspace.register_recovery_node(
                    effect.effect.recovery_node,
                    context.failed_node,
                )
            return {
                "graph_id": str(applied_graph.graph_id),
                "graph_version": applied_graph.version,
                "effect_type": effect.effect.effect_type,
            }

        async def apply_authorized_effect(
            normalized: NormalizedDecisionEffect[RecoveryDecisionEffect],
            permit: RuntimeCommitPermit,
        ) -> JsonValue:
            nonlocal applied_graph
            effect = normalized.payload
            candidate = apply_recovery_decision_effect(context.graph, effect)
            record = RecoveryRecord(
                plan=effect.plan,
                graph_version_before=context.graph.version,
                graph_version_after=candidate.version,
            )
            applied_graph = await committer.commit_graph_effect(
                state=context.state,
                graph=candidate,
                effect_fingerprint=normalized.effect_fingerprint,
                recovery_record=record,
                permit=permit,
                target=GovernanceTarget(
                    target_type=normalized.target.target_type,
                    target_id=normalized.target.target_id,
                ),
                subject_fingerprint=governance_fingerprint(normalized),
            )
            if isinstance(effect.effect, AddRecoveryNodeEffect):
                self._workspace.register_recovery_node(
                    effect.effect.recovery_node,
                    context.failed_node,
                )
            return {
                "graph_id": str(applied_graph.graph_id),
                "graph_version": applied_graph.version,
                "effect_type": effect.effect.effect_type,
            }

        async def reconcile_effect(
            normalized: NormalizedDecisionEffect[RecoveryDecisionEffect],
        ) -> DecisionReconciliation:
            committed = await committer.load_graph_effect(
                run_id=context.state.run_id,
                effect_fingerprint=normalized.effect_fingerprint,
            )
            if committed is None:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.NOT_COMMITTED,
                    reason="Recovery Graph effect is absent from authoritative storage",
                )
            effect = normalized.payload
            if committed.graph_id != effect.graph_id or committed.version != effect.graph_version_after:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.UNKNOWN,
                    reason="Recovery Graph read-back conflicts with the authorized effect",
                )
            result: JsonValue = {
                "graph_id": str(committed.graph_id),
                "graph_version": committed.version,
                "effect_type": effect.effect.effect_type,
            }
            return DecisionReconciliation(
                status=DecisionReconciliationStatus.COMMITTED,
                reason="Recovery Graph effect was committed and read back",
                apply_receipt=DecisionApplyReceipt(
                    effect_fingerprint=normalized.effect_fingerprint,
                    committed_state_fingerprint=governance_fingerprint(result),
                    result=result,
                ),
            )

        coordinator = DecisionLifecycleCoordinator[
            RecoveryDecisionPayload,
            RecoveryDraft,
            RecoveryDecisionEffect,
            DecisionGovernanceBinding,
        ](
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=RecoveryDecisionProposalProducer(
                capability=self._capability,
                execution_policy=self._execution_policy,
                correlation=InferenceCorrelation(
                    run_id=context.state.run_id,
                    task_id=context.state.task.task_id,
                    node_id=context.failed_node.node_id,
                    action_id=context.observation.action_id,
                ),
            ),
            basis_provider=_FixedRecoveryBasisProvider(basis),
            validator=RuntimeDecisionValidator(
                normalizer=RecoveryEffectNormalizer()
            ),
            governance=governance_adapter,
            applier=GovernedDecisionApplier(
                executor=self._operation_executor,
                apply_effect=apply_effect,
                apply_authorized_effect=apply_authorized_effect,
                reconcile_effect=reconcile_effect,
            ),
            checkpoint_store=self._checkpoint_store,
            checkpoint_type=DecisionCheckpoint[
                RecoveryDecisionPayload,
                RecoveryDraft,
                RecoveryDecisionEffect,
            ],
            trace_writer=RuntimeDecisionTraceWriter(self._trace_sink),
        )
        checkpoint = await coordinator.run(
            request,
            sources=sources,
            policy=projection_policy,
        )
        if checkpoint.stage is DecisionCheckpointStage.REVIEW_PENDING:
            receipt = checkpoint.governance_receipt
            if receipt is None or receipt.review_request_id is None:
                raise RuntimeError("Recovery review checkpoint has no review request")
            review = self._reviews.resolve(
                receipt.review_request_id,
                HumanReviewDecision(
                    outcome=ReviewOutcome.APPROVE,
                    reviewer_id="research-demo-reviewer",
                    rationale=(
                        "The normalized Recovery Effect is bounded and auditable."
                    ),
                    decided_at=datetime.now(timezone.utc),
                ),
            )
            recording_evaluator.review = review
            checkpoint = await coordinator.resume_review(request.request_id)

        if (
            applied_graph is None
            and checkpoint.result is not None
            and checkpoint.result.status is DecisionResultStatus.APPLIED
            and checkpoint.validated_decision is not None
        ):
            applied_graph = await committer.load_graph_effect(
                run_id=context.state.run_id,
                effect_fingerprint=(
                    checkpoint.validated_decision.normalized_effect.effect_fingerprint
                ),
            )

        if (
            checkpoint.result is None
            or checkpoint.result.status is not DecisionResultStatus.APPLIED
            or checkpoint.validated_decision is None
            or applied_graph is None
        ):
            reason = (
                checkpoint.result.reason
                if checkpoint.result is not None
                else checkpoint.stage
            )
            raise RuntimeError(f"Recovery decision did not apply: {reason}")
        governance_request = recording_evaluator.request
        preliminary = recording_evaluator.preliminary
        final = recording_evaluator.final
        if governance_request is None or preliminary is None or final is None:
            raise RuntimeError("Recovery Governance record is incomplete")
        self._workspace.governance_records.append(
            GovernanceRecord(
                scenario="failure_recovery_agent",
                request=governance_request,
                preliminary=preliminary,
                final=final,
                authorization=recording_issuer.authorization,
                review=recording_evaluator.review,
            )
        )
        effect = checkpoint.validated_decision.normalized_effect.payload
        return RecoveryDecisionOutcome(
            request_id=request.request_id,
            effect=effect,
            graph=applied_graph,
        )

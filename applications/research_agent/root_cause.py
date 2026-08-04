"""Research Application composition for semantic Root Cause decisions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime import TraceEntry, TraceSink
from adaptive_agent_runtime.decisioning import (
    ContextProjectionPolicy,
    ContextSensitivity,
    DecisionBasis,
    DecisionBudget,
    DecisionCheckpoint,
    DecisionCheckpointStage,
    DecisionConstraint,
    DecisionCorrelation,
    DecisionEvidenceReference,
    DecisionLifecycleCoordinator,
    DecisionRequest,
    DecisionResultStatus,
    DecisionTarget,
    InMemoryDecisionCheckpointStore,
    PolicyAgentContextBuilder,
    ProjectionSource,
    ProjectionSources,
    RuntimeDecisionTraceWriter,
    RuntimeDecisionValidator,
    decision_fingerprint,
)
from adaptive_agent_runtime.evaluation import (
    ROOT_CAUSE_DECISION_TYPE,
    ROOT_CAUSE_RECORD_OPERATION,
    AgentExecutionTrace,
    EvaluationReport,
    InMemoryRootCauseAssessmentStore,
    RecoveryEvidenceQuery,
    RootCauseAssessment,
    RootCauseAssessmentEffect,
    RootCauseDecisionInput,
    RootCauseDecisionPayload,
    RootCauseExecutionPolicy,
    RootCauseInputAssembler,
    RootCauseRecoveryEvidenceProvider,
    RuntimeTraceAdapter,
    root_cause_failure_signature,
    stable_root_cause_id,
)
from adaptive_agent_runtime.governance import (
    DecisionGovernanceBinding,
    GovernanceAuthorizationIssuer,
    GovernanceEvaluator,
    GovernedDecisionApplier,
    GovernedOperationExecutor,
    HumanReviewService,
    RuntimeDecisionGovernanceAdapter,
)
from adaptive_agent_runtime.llm import (
    ROOT_CAUSE_EVIDENCE_SOURCE_TYPE,
    ROOT_CAUSE_INPUT_SOURCE_TYPE,
    InferenceCorrelation,
    RootCauseAnalysisCapability,
    RootCauseDecisionProposalProducer,
    RootCauseDraft,
    RootCauseEffectNormalizer,
    build_root_cause_analysis_request,
)
from adaptive_agent_runtime.orchestration import RecoveryContext


class _FixedRootCauseBasisProvider:
    module_id = "research_agent.root_cause.basis"

    def __init__(self, basis: DecisionBasis) -> None:
        self._basis = basis

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        del request
        return self._basis


class ResearchRootCauseDecisionHandler:
    """Run one advisory semantic decision without changing Evaluation metrics."""

    module_id = "research_agent.root_cause_decision_handler"

    def __init__(
        self,
        *,
        capability: RootCauseAnalysisCapability,
        execution_policy: RootCauseExecutionPolicy,
        governance: GovernanceEvaluator,
        reviews: HumanReviewService,
        issuer: GovernanceAuthorizationIssuer,
        operation_executor: GovernedOperationExecutor,
        trace_sink: TraceSink,
        assessment_store: InMemoryRootCauseAssessmentStore,
    ) -> None:
        self._capability = capability
        self._execution_policy = execution_policy
        self._governance = governance
        self._reviews = reviews
        self._issuer = issuer
        self._operation_executor = operation_executor
        self._trace_sink = trace_sink
        self._assessment_store = assessment_store
        self._checkpoints = InMemoryDecisionCheckpointStore[
            RootCauseDecisionPayload,
            RootCauseDraft,
            RootCauseAssessmentEffect,
        ]()
        self._input_assembler = RootCauseInputAssembler()

    async def analyze_inline(
        self,
        *,
        run_id: UUID,
        task_id: UUID,
        node_id: UUID,
        action_id: UUID,
        failure_summary: str,
        runtime_entries: Sequence[TraceEntry],
    ) -> RootCauseAssessment | None:
        """Analyze a live failure from a bounded partial Trace snapshot."""

        try:
            trace_batch = RuntimeTraceAdapter().adapt(
                runtime_entries,
                run_id=run_id,
                task_id=task_id,
            )
            decision_input = self._input_assembler.inline_failure(
                run_id=run_id,
                task_id=task_id,
                node_id=node_id,
                action_id=action_id,
                failure_summary=failure_summary,
                trace_batch=trace_batch,
                policy=self._execution_policy,
            )
            return await self._run(decision_input)
        except Exception:
            # Semantic diagnosis is advisory. Recovery retains direct Observation
            # evidence when projection, inference, validation, or Governance fails.
            return None

    async def analyze_post_run(
        self,
        report: EvaluationReport,
        trace: AgentExecutionTrace,
    ) -> RootCauseAssessment | None:
        """Analyze deterministic failures without mutating their report."""

        try:
            decision_input = self._input_assembler.post_run(
                report,
                trace,
                policy=self._execution_policy,
            )
            if decision_input is None:
                return None
            return await self._run(decision_input)
        except Exception:
            return None

    async def _run(
        self,
        decision_input: RootCauseDecisionInput,
    ) -> RootCauseAssessment | None:
        payload = decision_input.payload
        correlation = payload.correlation
        request_id = stable_root_cause_id(
            "request",
            payload.trigger.value,
            correlation.run_id,
            correlation.task_id,
            correlation.node_id,
            correlation.action_id,
            payload.trace_id,
            payload.failure_signature,
        )
        basis = DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(
                {"payload": payload, "evidence": decision_input.evidence}
            )
        )
        request = DecisionRequest(
            request_id=request_id,
            decision_type=ROOT_CAUSE_DECISION_TYPE,
            target=DecisionTarget(
                target_type="root_cause_assessment",
                target_id=(
                    f"{correlation.run_id}:"
                    f"{correlation.action_id or payload.failure_signature}"
                ),
            ),
            correlation=DecisionCorrelation(
                run_id=correlation.run_id,
                task_id=correlation.task_id,
                node_id=correlation.node_id,
                action_id=correlation.action_id,
            ),
            basis=basis,
            payload=payload,
            allowed_actions=(ROOT_CAUSE_RECORD_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="root-cause-evidence-only",
                    description="Cite only Runtime-projected evidence.",
                ),
                DecisionConstraint(
                    constraint_id="root-cause-advisory-only",
                    description=(
                        "Produce a diagnosis only; do not select or execute Recovery."
                    ),
                ),
            ),
            evidence=decision_input.evidence,
            budget=DecisionBudget(
                max_decision_cycles=1,
                max_agent_calls=self._execution_policy.max_agent_calls,
                max_revision_count=0,
                max_elapsed_seconds=self._execution_policy.timeout_seconds,
            ),
        )
        projected = build_root_cause_analysis_request(
            payload,
            decision_input.evidence,
        )
        primary_evidence_id = decision_input.evidence[0].evidence_id
        sources = [
            ProjectionSource(
                source_id="root-cause-input",
                source_type=ROOT_CAUSE_INPUT_SOURCE_TYPE,
                agent_scope="root_cause",
                content=projected.model_dump(mode="json"),
                sensitivity=ContextSensitivity.INTERNAL,
                evidence_id=primary_evidence_id,
                priority=100,
                estimated_tokens=512,
            )
        ]
        sources.extend(
            ProjectionSource(
                source_id=f"root-cause-evidence:{item.evidence_id}",
                source_type=ROOT_CAUSE_EVIDENCE_SOURCE_TYPE,
                agent_scope="root_cause",
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
            for item in decision_input.evidence[1:]
        )
        projection_policy = ContextProjectionPolicy(
            policy_id="research.root_cause.context",
            version="1",
            agent_scope="root_cause",
            allowed_decision_types=frozenset({ROOT_CAUSE_DECISION_TYPE}),
            allowed_source_types=frozenset(
                {
                    ROOT_CAUSE_INPUT_SOURCE_TYPE,
                    ROOT_CAUSE_EVIDENCE_SOURCE_TYPE,
                }
            ),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            redact_keys=frozenset(
                {
                    "agent_id",
                    "authorization",
                    "credential",
                    "governance",
                    "model_id",
                    "password",
                    "reasoning",
                    "token",
                }
            ),
            max_items=len(sources),
            max_context_tokens=512 + (64 * max(0, len(sources) - 1)),
        )
        applied: RootCauseAssessment | None = None

        async def apply_effect(effect: RootCauseAssessmentEffect) -> JsonValue:
            nonlocal applied
            await self._assessment_store.record(effect.assessment)
            applied = effect.assessment
            return {
                "assessment_id": str(effect.assessment.assessment_id),
                "conclusion": effect.assessment.conclusion.value,
                "evidence_strength": effect.assessment.evidence_strength.value,
            }

        coordinator = DecisionLifecycleCoordinator[
            RootCauseDecisionPayload,
            RootCauseDraft,
            RootCauseAssessmentEffect,
            DecisionGovernanceBinding,
        ](
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=RootCauseDecisionProposalProducer(
                capability=self._capability,
                execution_policy=self._execution_policy,
                correlation=InferenceCorrelation(
                    run_id=correlation.run_id,
                    task_id=correlation.task_id,
                    node_id=correlation.node_id,
                    action_id=correlation.action_id,
                ),
            ),
            basis_provider=_FixedRootCauseBasisProvider(basis),
            validator=RuntimeDecisionValidator(
                normalizer=RootCauseEffectNormalizer()
            ),
            governance=RuntimeDecisionGovernanceAdapter(
                evaluator=self._governance,
                authorization_issuer=self._issuer,
                review_service=self._reviews,
            ),
            applier=GovernedDecisionApplier(
                executor=self._operation_executor,
                apply_effect=apply_effect,
            ),
            checkpoint_store=self._checkpoints,
            checkpoint_type=DecisionCheckpoint[
                RootCauseDecisionPayload,
                RootCauseDraft,
                RootCauseAssessmentEffect,
            ],
            trace_writer=RuntimeDecisionTraceWriter(self._trace_sink),
        )
        checkpoint = await coordinator.run(
            request,
            sources=ProjectionSources(items=tuple(sources)),
            policy=projection_policy,
        )
        if (
            checkpoint.stage is not DecisionCheckpointStage.COMPLETED
            or checkpoint.result is None
            or checkpoint.result.status is not DecisionResultStatus.APPLIED
        ):
            return None
        return applied


class ResearchRootCauseRecoveryBridge:
    """Analyze one failure, then admit only validated evidence to Recovery."""

    module_id = "research_agent.root_cause_recovery_bridge"

    def __init__(
        self,
        *,
        handler: ResearchRootCauseDecisionHandler,
        assessment_store: InMemoryRootCauseAssessmentStore,
        trace_reader: Callable[[UUID], tuple[TraceEntry, ...]],
    ) -> None:
        self._handler = handler
        self._provider = RootCauseRecoveryEvidenceProvider(assessment_store)
        self._trace_reader = trace_reader

    async def evidence_for(
        self,
        context: RecoveryContext,
    ) -> tuple[DecisionEvidenceReference, ...]:
        error = context.observation.error or "unknown failure"
        await self._handler.analyze_inline(
            run_id=context.state.run_id,
            task_id=context.state.task.task_id,
            node_id=context.failed_node.node_id,
            action_id=context.observation.action_id,
            failure_summary=error,
            runtime_entries=self._trace_reader(context.state.run_id),
        )
        diagnostics = await self._provider.evidence_for(
            RecoveryEvidenceQuery(
                run_id=context.state.run_id,
                task_id=context.state.task.task_id,
                node_id=context.failed_node.node_id,
                action_id=context.observation.action_id,
                failure_signature=root_cause_failure_signature(error),
            )
        )
        return tuple(
            DecisionEvidenceReference(
                evidence_id=item.evidence_id,
                kind=item.kind,
                source="evaluation_root_cause",
                reliability=item.reliability,
                summary=item.summary,
            )
            for item in diagnostics
        )

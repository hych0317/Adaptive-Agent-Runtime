from __future__ import annotations

import unittest
from datetime import datetime, timezone
from uuid import UUID

from adaptive_agent_runtime import (
    AgentState,
    AgentTask,
    InMemoryTraceSink,
    Observation,
    RunStatus,
    RuntimeEvent,
)
from adaptive_agent_runtime.decisioning import (
    DecisionBasis,
    DecisionCorrelation,
    DecisionProducer,
    DecisionProposal,
    DecisionRequest,
    DecisionTarget,
    decision_fingerprint,
)
from adaptive_agent_runtime.evaluation import (
    AgentEvaluationPipeline,
    AgentExecutionTrace,
    ContextMemoryComponentEvaluator,
    DeterministicOutcomeEvaluator,
    DeterministicTrajectoryEvaluator,
    EvaluationComponent,
    EvaluationCorrelation,
    EvaluationCriteria,
    EvaluationFact,
    EvaluationStateSnapshot,
    EvaluationSubject,
    ExecutionResultSnapshot,
    ExecutionStatus,
    InMemoryRootCauseAssessmentStore,
    OrchestrationComponentEvaluator,
    RootCauseConclusion,
    RootCauseEvidenceStrength,
    RootCauseExecutionPolicy,
    RootCauseInputAssembler,
    ToolComponentEvaluator,
    TraceBatch,
    TraceCategory,
    TraceCompleteness,
    TraceCoverage,
)
from adaptive_agent_runtime.governance import (
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceAuthorizationIssuer,
    GovernedOperationExecutor,
    InMemoryAuthorizationConsumptionStore,
    InMemoryHumanReviewService,
    RuntimeGovernanceEvaluator,
    StrictAuthorizationVerifier,
    default_governance_policy,
)
from adaptive_agent_runtime.llm import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CapabilityTurnResult,
    RecoveryActionDraft,
    RecoveryActionDraftKind,
    RecoveryDraft,
    RecoveryFailureKindDraft,
    RecoveryProposalRequest,
    RootCauseAnalysisRequest,
    RootCauseConclusionDraft,
    RootCauseDraft,
    RootCauseEffectNormalizer,
    RootCauseHypothesisDraft,
)
from adaptive_agent_runtime.orchestration import (
    DynamicTaskGraph,
    RecoveryContext,
    RecoveryExecutionPolicy,
    TaskNode,
)

from applications.research_agent.recovery import ResearchRecoveryDecisionHandler
from applications.research_agent.root_cause import (
    ResearchRootCauseDecisionHandler,
    ResearchRootCauseRecoveryBridge,
)
from applications.research_agent.strategies import ResearchWorkspace
from applications.research_agent.tasks import ResearchTaskDefinition


NOW = datetime(2026, 8, 5, tzinfo=timezone.utc)
RUN_ID = UUID(int=8201)
TASK_ID = UUID(int=8202)
NODE_ID = UUID(int=8203)
ACTION_ID = UUID(int=8204)


def governance_stack():
    reviews = InMemoryHumanReviewService()
    governance = RuntimeGovernanceEvaluator(
        policy=default_governance_policy(),
        rule_evaluator=DeterministicRuleEvaluator(),
        confidence_evaluator=DeterministicConfidenceEvaluator(),
        review_service=reviews,
    )
    issuer = GovernanceAuthorizationIssuer()
    executor = GovernedOperationExecutor(
        verifier=StrictAuthorizationVerifier(),
        consumption_store=InMemoryAuthorizationConsumptionStore(),
    )
    return governance, reviews, issuer, executor


class FakeRootCauseAgent:
    module_id = "test.root_cause_agent"
    capability_id = "test.root_cause_analysis"

    def __init__(self, *, confidence: float = 0.99) -> None:
        self.confidence = confidence
        self.requests: list[RootCauseAnalysisRequest] = []

    async def analyze_root_cause(
        self,
        request: RootCauseAnalysisRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[RootCauseDraft]:
        del invocation
        self.requests.append(request)
        evidence_ids = tuple(
            item.reference_id for item in request.evidence_catalog[:2]
        )
        code = (
            request.deterministic_findings[0].code
            if request.deterministic_findings
            else "source.input_invalid"
        )
        return CapabilityTurnResult[RootCauseDraft](
            kind=CapabilityTurnKind.COMPLETED,
            result=RootCauseDraft(
                conclusion=RootCauseConclusionDraft.SUPPORTED,
                primary=RootCauseHypothesisDraft(
                    code=code,
                    description="The source supplied invalid input to this node.",
                    supporting_evidence_reference_ids=evidence_ids,
                ),
                rationale="Direct failure evidence and Trace facts agree.",
                confidence=self.confidence,
            ),
        )


class EvidenceAwareRecoveryAgent:
    module_id = "test.evidence_aware_recovery_agent"
    capability_id = "test.recovery_proposal"

    def __init__(self) -> None:
        self.requests: list[RecoveryProposalRequest] = []

    async def propose_recovery(
        self,
        request: RecoveryProposalRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[RecoveryDraft]:
        del invocation
        self.requests.append(request)
        diagnosis = next(
            item
            for item in request.evidence
            if item.kind == "evaluation.root_cause.assessment"
        )
        return CapabilityTurnResult[RecoveryDraft](
            kind=CapabilityTurnKind.COMPLETED,
            result=RecoveryDraft(
                failure_kind=RecoveryFailureKindDraft.INVALID_INPUT,
                hypothesis="Validated diagnosis supports one bounded retry.",
                alternatives=("Abort",),
                selected_action=RecoveryActionDraft(
                    kind=RecoveryActionDraftKind.RETRY_NODE,
                    target_node_ref=request.failed_node_ref,
                    reason="Retry with the validated diagnosis available.",
                ),
                rationale="Use admitted Root Cause evidence, not Agent confidence.",
                evidence_reference_ids=(diagnosis.reference_id,),
                confidence=0.88,
            ),
        )


def partial_trace_batch() -> TraceBatch:
    fact = EvaluationFact(
        fact_id=UUID(int=8210),
        component=EvaluationComponent.RUNTIME,
        category=TraceCategory.NODE_EXECUTION,
        kind="observation.received",
        source="runtime",
        occurred_at=NOW,
        correlation=EvaluationCorrelation(
            run_id=RUN_ID,
            task_id=TASK_ID,
            node_id=NODE_ID,
            action_id=ACTION_ID,
        ),
        source_scope=f"runtime:{RUN_ID}",
        source_sequence=2,
        source_record_id="observation-record",
        payload={"error": "source input is invalid"},
    )
    return TraceBatch(
        facts=(fact,),
        coverage=(
            TraceCoverage(
                run_id=RUN_ID,
                component=EvaluationComponent.RUNTIME,
                completeness=TraceCompleteness.PARTIAL,
                diagnostics=("run is still active",),
            ),
        ),
    )


def decision_and_proposal(confidence: float):
    decision_input = RootCauseInputAssembler().inline_failure(
        run_id=RUN_ID,
        task_id=TASK_ID,
        node_id=NODE_ID,
        action_id=ACTION_ID,
        failure_summary="source input is invalid",
        trace_batch=partial_trace_batch(),
        policy=RootCauseExecutionPolicy(),
    )
    basis = DecisionBasis(
        snapshot_fingerprint=decision_fingerprint(decision_input.payload)
    )
    request = DecisionRequest(
        decision_type="evaluation.root_cause",
        target=DecisionTarget(
            target_type="root_cause_assessment",
            target_id=str(ACTION_ID),
        ),
        correlation=DecisionCorrelation(
            run_id=RUN_ID,
            task_id=TASK_ID,
            node_id=NODE_ID,
            action_id=ACTION_ID,
        ),
        basis=basis,
        payload=decision_input.payload,
        allowed_actions=("evaluation.root_cause.record",),
        evidence=decision_input.evidence,
    )
    evidence_ids = tuple(item.evidence_id for item in decision_input.evidence)
    draft = RootCauseDraft(
        conclusion=RootCauseConclusionDraft.SUPPORTED,
        primary=RootCauseHypothesisDraft(
            code="source.input_invalid",
            description="The source returned invalid input.",
            supporting_evidence_reference_ids=evidence_ids,
        ),
        rationale="The direct Observation is corroborated by Trace.",
        confidence=confidence,
    )
    proposal = DecisionProposal(
        request_id=request.request_id,
        proposal_type=request.decision_type,
        producer=DecisionProducer(
            producer_id="fake",
            capability="root_cause_analysis",
        ),
        input_snapshot_fingerprint=basis.snapshot_fingerprint,
        context_fingerprint="a" * 64,
        selected_action="evaluation.root_cause.record",
        payload=draft,
        rationale=draft.rationale,
        evidence_refs=draft.evidence_reference_ids,
        confidence=confidence,
    )
    return request, proposal


class RootCauseNormalizationTests(unittest.TestCase):
    def test_agent_confidence_does_not_change_runtime_evidence_strength_or_risk(
        self,
    ) -> None:
        low_request, low_proposal = decision_and_proposal(0.05)
        high_request, high_proposal = decision_and_proposal(0.99)

        low = RootCauseEffectNormalizer().normalize(low_request, low_proposal)
        high = RootCauseEffectNormalizer().normalize(high_request, high_proposal)

        self.assertEqual(low.risk, high.risk)
        self.assertEqual(low.impact_score, high.impact_score)
        self.assertEqual(
            low.payload.assessment.evidence_strength,
            RootCauseEvidenceStrength.CORROBORATED,
        )
        self.assertEqual(
            high.payload.assessment.evidence_strength,
            RootCauseEvidenceStrength.CORROBORATED,
        )
        self.assertEqual(low.payload.assessment.stated_confidence, 0.05)
        self.assertEqual(high.payload.assessment.stated_confidence, 0.99)

    def test_missing_trace_rejects_supported_claim(self) -> None:
        missing = TraceBatch(
            coverage=(
                TraceCoverage(
                    run_id=RUN_ID,
                    component=EvaluationComponent.RUNTIME,
                    completeness=TraceCompleteness.MISSING,
                ),
            )
        )
        decision_input = RootCauseInputAssembler().inline_failure(
            run_id=RUN_ID,
            task_id=TASK_ID,
            node_id=NODE_ID,
            action_id=ACTION_ID,
            failure_summary="source input is invalid",
            trace_batch=missing,
            policy=RootCauseExecutionPolicy(),
        )
        basis = DecisionBasis(
            snapshot_fingerprint=decision_fingerprint(decision_input.payload)
        )
        request = DecisionRequest(
            decision_type="evaluation.root_cause",
            target=DecisionTarget(target_type="assessment", target_id="missing"),
            correlation=DecisionCorrelation(
                run_id=RUN_ID,
                task_id=TASK_ID,
                node_id=NODE_ID,
                action_id=ACTION_ID,
            ),
            basis=basis,
            payload=decision_input.payload,
            allowed_actions=("evaluation.root_cause.record",),
            evidence=decision_input.evidence,
        )
        draft = RootCauseDraft(
            conclusion=RootCauseConclusionDraft.SUPPORTED,
            primary=RootCauseHypothesisDraft(
                code="source.input_invalid",
                description="The source returned invalid input.",
                supporting_evidence_reference_ids=(
                    decision_input.evidence[0].evidence_id,
                ),
            ),
            rationale="One Observation is enough.",
            confidence=1.0,
        )
        proposal = DecisionProposal(
            request_id=request.request_id,
            proposal_type=request.decision_type,
            producer=DecisionProducer(producer_id="fake", capability="root_cause"),
            input_snapshot_fingerprint=basis.snapshot_fingerprint,
            context_fingerprint="a" * 64,
            selected_action="evaluation.root_cause.record",
            payload=draft,
            rationale=draft.rationale,
            evidence_refs=draft.evidence_reference_ids,
            confidence=draft.confidence,
        )

        with self.assertRaisesRegex(ValueError, "missing Trace coverage"):
            RootCauseEffectNormalizer().normalize(request, proposal)


class RootCauseLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_lifecycle_records_isolated_advisory_assessment(self) -> None:
        trace = InMemoryTraceSink()
        await trace.record(
            RuntimeEvent(
                run_id=RUN_ID,
                kind="runtime.started",
                source="runtime",
                payload={"task_id": str(TASK_ID)},
            )
        )
        agent = FakeRootCauseAgent()
        store = InMemoryRootCauseAssessmentStore()
        governance, reviews, issuer, executor = governance_stack()
        handler = ResearchRootCauseDecisionHandler(
            capability=agent,
            execution_policy=RootCauseExecutionPolicy(),
            governance=governance,
            reviews=reviews,
            issuer=issuer,
            operation_executor=executor,
            trace_sink=trace,
            assessment_store=store,
        )

        assessment = await handler.analyze_inline(
            run_id=RUN_ID,
            task_id=TASK_ID,
            node_id=NODE_ID,
            action_id=ACTION_ID,
            failure_summary="source input is invalid",
            runtime_entries=trace.entries_for(RUN_ID),
        )

        self.assertIsNotNone(assessment)
        assert assessment is not None
        self.assertEqual(assessment.conclusion, RootCauseConclusion.SUPPORTED)
        self.assertEqual(await store.load(assessment.assessment_id), assessment)
        request_json = agent.requests[0].model_dump(mode="json")
        for hidden in (
            "runtime_state",
            "global_memory",
            "governance",
            "model_id",
            "agent_id",
            "reasoning",
        ):
            self.assertNotIn(hidden, request_json)
        kinds = tuple(item.event.kind for item in trace.entries_for(RUN_ID))
        self.assertIn("decision.requested", kinds)
        self.assertIn("decision.validation_passed", kinds)
        self.assertIn("decision.authorized", kinds)
        self.assertIn("decision.applied", kinds)

    async def test_root_cause_enters_recovery_as_validated_evidence_only(self) -> None:
        failed = TaskNode(
            node_id=NODE_ID,
            goal="Collect fragile source",
            expected_output="Source evidence",
            strategy_id="test",
        )
        observation = Observation.failed(
            ACTION_ID,
            error="source input is invalid",
        )
        graph = DynamicTaskGraph(nodes=(failed,)).mark_running(NODE_ID)
        graph = graph.resolve_node(NODE_ID, observation)
        task = AgentTask(task_id=TASK_ID, description="Recover the source")
        state = AgentState(
            run_id=RUN_ID,
            task=task,
            status=RunStatus.RUNNING,
            revision=2,
            step_count=1,
            last_observation=observation,
        )
        context = RecoveryContext(
            graph=graph,
            state=state,
            failed_node=graph.get_node(NODE_ID),
            observation=observation,
            prior_attempts=0,
        )
        trace = InMemoryTraceSink()
        await trace.record(
            RuntimeEvent(
                run_id=RUN_ID,
                kind="runtime.started",
                source="runtime",
                payload={"state": state.model_dump(mode="json")},
            )
        )
        store = InMemoryRootCauseAssessmentStore()
        governance, reviews, issuer, executor = governance_stack()
        root_handler = ResearchRootCauseDecisionHandler(
            capability=FakeRootCauseAgent(),
            execution_policy=RootCauseExecutionPolicy(),
            governance=governance,
            reviews=reviews,
            issuer=issuer,
            operation_executor=executor,
            trace_sink=trace,
            assessment_store=store,
        )
        bridge = ResearchRootCauseRecoveryBridge(
            handler=root_handler,
            assessment_store=store,
            trace_reader=trace.entries_for,
        )
        workspace = ResearchWorkspace(
            ResearchTaskDefinition(
                company="Test Co",
                initial_graph=graph,
                nodes={"failed": failed},
            )
        )
        recovery_agent = EvidenceAwareRecoveryAgent()
        recovery = ResearchRecoveryDecisionHandler(
            capability=recovery_agent,
            execution_policy=RecoveryExecutionPolicy(
                timeout_seconds=5.0,
                max_recovery_attempts=1,
            ),
            governance=governance,
            reviews=reviews,
            issuer=issuer,
            operation_executor=executor,
            trace_sink=trace,
            workspace=workspace,
            available_strategy_ids=("test",),
            diagnostic_evidence_provider=bridge,
        )

        outcome = await recovery.handle(context)

        diagnostic = next(
            item
            for item in recovery_agent.requests[0].evidence
            if item.kind == "evaluation.root_cause.assessment"
        )
        self.assertTrue(diagnostic.reference_id.startswith("root-cause:"))
        self.assertIn(diagnostic.summary, outcome.effect.plan.analysis.evidence)
        self.assertGreater(outcome.graph.version, graph.version)


class DeterministicEvaluationIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_post_run_root_cause_cannot_mutate_deterministic_report(self) -> None:
        fact = EvaluationFact(
            fact_id=UUID(int=8301),
            component=EvaluationComponent.RUNTIME,
            category=TraceCategory.FINAL_RESULT,
            kind="runtime.failed",
            source="runtime",
            occurred_at=NOW,
            correlation=EvaluationCorrelation(run_id=RUN_ID, task_id=TASK_ID),
            source_scope=f"runtime:{RUN_ID}",
            source_sequence=1,
            source_record_id="terminal",
            payload={"error": "execution failed"},
        )
        trace_model = AgentExecutionTrace(
            trace_id=UUID(int=8302),
            run_id=RUN_ID,
            task_id=TASK_ID,
            facts=(fact,),
            coverage=(
                TraceCoverage(
                    run_id=RUN_ID,
                    component=EvaluationComponent.RUNTIME,
                    completeness=TraceCompleteness.COMPLETE,
                ),
            ),
            collected_at=NOW,
        )
        subject = EvaluationSubject(
            trace=trace_model,
            state=EvaluationStateSnapshot(
                run_id=RUN_ID,
                task_id=TASK_ID,
                task_description="Fail deterministically",
                status=ExecutionStatus.FAILED,
                revision=2,
                step_count=1,
                error="execution failed",
                captured_at=NOW,
            ),
            result=ExecutionResultSnapshot(
                run_id=RUN_ID,
                task_id=TASK_ID,
                succeeded=False,
                error="execution failed",
                completed_at=NOW,
            ),
        )
        report = AgentEvaluationPipeline(
            outcome=DeterministicOutcomeEvaluator(),
            trajectory=DeterministicTrajectoryEvaluator(),
            components=(
                OrchestrationComponentEvaluator(),
                ToolComponentEvaluator(),
                ContextMemoryComponentEvaluator(),
            ),
        ).evaluate(subject, EvaluationCriteria())
        before = report.model_dump(mode="json")
        trace_sink = InMemoryTraceSink()
        store = InMemoryRootCauseAssessmentStore()
        governance, reviews, issuer, executor = governance_stack()
        handler = ResearchRootCauseDecisionHandler(
            capability=FakeRootCauseAgent(),
            execution_policy=RootCauseExecutionPolicy(),
            governance=governance,
            reviews=reviews,
            issuer=issuer,
            operation_executor=executor,
            trace_sink=trace_sink,
            assessment_store=store,
        )

        assessment = await handler.analyze_post_run(report, trace_model)

        self.assertIsNotNone(assessment)
        self.assertEqual(report.model_dump(mode="json"), before)
        self.assertEqual(report.overall_score, report.overall_score)


if __name__ == "__main__":
    unittest.main()

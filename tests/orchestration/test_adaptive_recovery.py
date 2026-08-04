from __future__ import annotations

import unittest
from uuid import UUID

from adaptive_agent_runtime import (
    AgentRuntime,
    AgentState,
    AgentTask,
    InMemoryStateStore,
    InMemoryTraceSink,
    Observation,
    RunStatus,
)
from adaptive_agent_runtime.decisioning import (
    DecisionBasis,
    DecisionBudget,
    DecisionCorrelation,
    DecisionEvidenceReference,
    DecisionProducer,
    DecisionProposal,
    DecisionRequest,
    DecisionTarget,
    decision_fingerprint,
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
    RecoveryEffectNormalizer,
    RecoveryFailureKindDraft,
    RecoveryProposalRequest,
    TaskNodeDraft,
)
from adaptive_agent_runtime.orchestration import (
    RECOVERY_APPLY_OPERATION,
    RECOVERY_DECISION_TYPE,
    AddRecoveryNodeEffect,
    DynamicTaskGraph,
    DynamicTaskGraphPlanner,
    NodeExecutionResult,
    RecoveryActionType,
    RecoveryDecisionPayload,
    RecoveryExecutionPolicy,
    RecoveryNodeBinding,
    RetryNodeRecoveryEffect,
    StrategyActionExecutor,
    TaskNode,
    TaskNodeStatus,
)

from applications.research_agent.recovery import ResearchRecoveryDecisionHandler
from applications.research_agent.strategies import ResearchWorkspace
from applications.research_agent.tasks import ResearchTaskDefinition


def failed_graph() -> tuple[DynamicTaskGraph, TaskNode, Observation]:
    node = TaskNode(
        goal="Collect primary evidence",
        expected_output="Primary evidence",
        strategy_id="test",
    )
    graph = DynamicTaskGraph(nodes=(node,)).mark_running(node.node_id)
    observation = Observation.failed(UUID(int=701), error="upstream source failed")
    return graph.resolve_node(node.node_id, observation), node, observation


def recovery_payload(
    graph: DynamicTaskGraph,
    node: TaskNode,
    observation: Observation,
) -> RecoveryDecisionPayload:
    return RecoveryDecisionPayload(
        task_description="Recover the research run",
        graph=graph,
        failed_node_id=node.node_id,
        failed_action_id=observation.action_id,
        node_bindings=tuple(
            RecoveryNodeBinding(
                node_ref=f"node:{item.node_id}",
                node_id=item.node_id,
            )
            for item in graph.nodes
        ),
        available_strategy_ids=("test", "backup"),
        allowed_recovery_actions=tuple(RecoveryActionType),
        prior_attempts=0,
        execution_policy=RecoveryExecutionPolicy(),
    )


def decision_request(
    payload: RecoveryDecisionPayload,
) -> DecisionRequest[RecoveryDecisionPayload]:
    basis = DecisionBasis(
        snapshot_fingerprint=decision_fingerprint(payload),
        state_revision=1,
        graph_version=payload.graph.version,
    )
    return DecisionRequest[RecoveryDecisionPayload](
        decision_type=RECOVERY_DECISION_TYPE,
        target=DecisionTarget(
            target_type="active_task_graph",
            target_id=str(payload.graph.graph_id),
        ),
        correlation=DecisionCorrelation(
            run_id=UUID(int=710),
            task_id=UUID(int=711),
            node_id=payload.failed_node_id,
            action_id=payload.failed_action_id,
        ),
        basis=basis,
        payload=payload,
        allowed_actions=(RECOVERY_APPLY_OPERATION,),
        evidence=(
            DecisionEvidenceReference(
                evidence_id="failure:evidence",
                kind="runtime.observation.failure",
                source="runtime",
                reliability=1.0,
                summary="The source returned a failed Observation.",
            ),
        ),
        budget=DecisionBudget(max_elapsed_seconds=10.0),
    )


def proposal(
    request: DecisionRequest[RecoveryDecisionPayload],
    draft: RecoveryDraft,
) -> DecisionProposal[RecoveryDraft]:
    return DecisionProposal[RecoveryDraft](
        request_id=request.request_id,
        proposal_type=request.decision_type,
        producer=DecisionProducer(
            producer_id="fake-recovery-agent",
            capability="recovery_proposal",
        ),
        input_snapshot_fingerprint=request.basis.snapshot_fingerprint,
        context_fingerprint="a" * 64,
        selected_action=RECOVERY_APPLY_OPERATION,
        payload=draft,
        rationale=draft.rationale,
        evidence_refs=draft.evidence_reference_ids,
        confidence=draft.confidence,
    )


def retry_draft(
    payload: RecoveryDecisionPayload,
    *,
    confidence: float,
) -> RecoveryDraft:
    return RecoveryDraft(
        failure_kind=RecoveryFailureKindDraft.TRANSIENT,
        hypothesis="The failure is transient and a bounded retry may succeed.",
        alternatives=("Change strategy", "Abort the failed branch"),
        selected_action=RecoveryActionDraft(
            kind=RecoveryActionDraftKind.RETRY_NODE,
            target_node_ref=payload.node_ref_for(payload.failed_node_id),
            reason="Retry once on the unchanged strategy.",
        ),
        rationale="Retry has the smallest normalized impact.",
        evidence_reference_ids=("failure:evidence",),
        confidence=confidence,
    )


class RuntimeRecoveryEffectTests(unittest.TestCase):
    def test_runtime_preserves_hypothesis_but_derives_risk_without_confidence(self) -> None:
        graph, node, observation = failed_graph()
        payload = recovery_payload(graph, node, observation)
        request = decision_request(payload)
        normalizer = RecoveryEffectNormalizer()

        low = normalizer.normalize(
            request,
            proposal(request, retry_draft(payload, confidence=0.05)),
        )
        high = normalizer.normalize(
            request,
            proposal(request, retry_draft(payload, confidence=0.99)),
        )

        self.assertEqual(low.risk, high.risk)
        self.assertEqual(low.impact_score, high.impact_score)
        self.assertEqual(low.reversible, high.reversible)
        self.assertIsInstance(low.payload.effect, RetryNodeRecoveryEffect)
        self.assertEqual(
            low.payload.hypothesis,
            "The failure is transient and a bounded retry may succeed.",
        )
        self.assertEqual(
            low.payload.alternatives,
            ("Change strategy", "Abort the failed branch"),
        )
        self.assertEqual(low.payload.agent_confidence, 0.05)

    def test_graph_recovery_is_a_typed_patch_not_a_replacement_graph(self) -> None:
        graph, node, observation = failed_graph()
        payload = recovery_payload(graph, node, observation)
        request = decision_request(payload)
        draft = RecoveryDraft(
            failure_kind=RecoveryFailureKindDraft.INVALID_INPUT,
            hypothesis="A bounded preparation node can repair the missing input.",
            alternatives=("Abort",),
            selected_action=RecoveryActionDraft(
                kind=RecoveryActionDraftKind.ADD_RECOVERY_NODE,
                target_node_ref=payload.node_ref_for(payload.failed_node_id),
                recovery_node=TaskNodeDraft(
                    node_key="prepare-fallback-input",
                    goal="Prepare fallback evidence",
                    expected_output="Validated fallback evidence",
                    requested_strategy_id="test",
                ),
                reason="Insert a fallback preparation step.",
            ),
            rationale="The downstream path can continue through bounded fallback data.",
            evidence_reference_ids=("failure:evidence",),
            confidence=0.8,
        )

        normalized = RecoveryEffectNormalizer().normalize(
            request,
            proposal(request, draft),
        )

        self.assertIsInstance(normalized.payload.effect, AddRecoveryNodeEffect)
        self.assertNotIn("graph", type(normalized.payload).model_fields)
        self.assertGreater(
            normalized.payload.graph_version_after,
            normalized.payload.graph_version_before,
        )


class FakeRecoveryAgent:
    module_id = "test.recovery_agent"
    capability_id = "test.recovery_proposal"

    def __init__(self) -> None:
        self.requests: list[RecoveryProposalRequest] = []
        self.invocations: list[CapabilityInvocationMetadata | None] = []

    async def propose_recovery(
        self,
        request: RecoveryProposalRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[RecoveryDraft]:
        self.requests.append(request)
        self.invocations.append(invocation)
        evidence_id = request.evidence[0].reference_id
        return CapabilityTurnResult[RecoveryDraft](
            kind=CapabilityTurnKind.COMPLETED,
            result=RecoveryDraft(
                failure_kind=RecoveryFailureKindDraft.INVALID_INPUT,
                hypothesis="A preparation node can replace the failed source path.",
                alternatives=("Retry the failed source", "Abort the run"),
                selected_action=RecoveryActionDraft(
                    kind=RecoveryActionDraftKind.ADD_RECOVERY_NODE,
                    target_node_ref=request.failed_node_ref,
                    recovery_node=TaskNodeDraft(
                        node_key="runtime-fallback",
                        goal="Prepare fallback evidence",
                        expected_output="Fallback evidence",
                        requested_strategy_id="test",
                    ),
                    reason="Route future work through fallback evidence.",
                ),
                rationale="The source failure is persistent but has a bounded fallback.",
                evidence_reference_ids=(evidence_id,),
                confidence=0.87,
            ),
        )


class PathRecordingStrategy:
    module_id = "test.path_recording_strategy"
    strategy_id = "test"

    def __init__(self, failed_node_id: UUID) -> None:
        self._failed_node_id = failed_node_id
        self.calls: list[TaskNode] = []

    async def execute(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> NodeExecutionResult:
        del state
        self.calls.append(node)
        if node.node_id == self._failed_node_id:
            return NodeExecutionResult.failed(error="source input is invalid")
        return NodeExecutionResult.ok(output={"goal": node.goal})


class AdaptiveRecoveryEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_changes_future_execution_path_through_decision_lifecycle(
        self,
    ) -> None:
        failed = TaskNode(
            goal="Collect fragile source evidence",
            expected_output="Source evidence",
            strategy_id="test",
        )
        downstream = TaskNode(
            goal="Use recovered evidence",
            dependencies=(failed.node_id,),
            expected_output="Recovered analysis",
            strategy_id="test",
        )
        initial_graph = DynamicTaskGraph(nodes=(failed, downstream))
        workspace = ResearchWorkspace(
            ResearchTaskDefinition(
                company="Test Co",
                initial_graph=initial_graph,
                nodes={"source": failed, "downstream": downstream},
            )
        )
        reviews = InMemoryHumanReviewService()
        trace = InMemoryTraceSink()
        agent = FakeRecoveryAgent()
        handler = ResearchRecoveryDecisionHandler(
            capability=agent,
            execution_policy=RecoveryExecutionPolicy(
                timeout_seconds=5.0,
                max_recovery_attempts=1,
            ),
            governance=RuntimeGovernanceEvaluator(
                policy=default_governance_policy(),
                rule_evaluator=DeterministicRuleEvaluator(),
                confidence_evaluator=DeterministicConfidenceEvaluator(),
                review_service=reviews,
            ),
            reviews=reviews,
            issuer=GovernanceAuthorizationIssuer(),
            operation_executor=GovernedOperationExecutor(
                verifier=StrictAuthorizationVerifier(),
                consumption_store=InMemoryAuthorizationConsumptionStore(),
            ),
            trace_sink=trace,
            workspace=workspace,
            available_strategy_ids=("test",),
        )
        planner = DynamicTaskGraphPlanner(
            initial_graph,
            recovery_decision_handler=handler,
        )
        strategy = PathRecordingStrategy(failed.node_id)
        result = await AgentRuntime(
            planner=planner,
            executor=StrategyActionExecutor((strategy,)),
            state_store=InMemoryStateStore(),
            trace_sink=trace,
            max_steps=8,
        ).run(AgentTask(description="Recover the failed evidence path"))

        self.assertEqual(result.final_state.status, RunStatus.COMPLETED)
        executed_goals = [node.goal for node in strategy.calls]
        self.assertEqual(
            executed_goals,
            [
                "Collect fragile source evidence",
                "Prepare fallback evidence",
                "Use recovered evidence",
            ],
        )
        final_graph = planner.graph_for(result.final_state.run_id)
        self.assertEqual(
            final_graph.get_node(failed.node_id).status,
            TaskNodeStatus.RECOVERED,
        )
        recovery_node = next(
            item for item in final_graph.nodes if item.goal == "Prepare fallback evidence"
        )
        self.assertEqual(
            final_graph.get_node(downstream.node_id).dependencies,
            (recovery_node.node_id,),
        )
        kinds = tuple(
            entry.event.kind for entry in trace.entries_for(result.final_state.run_id)
        )
        self.assertIn("decision.requested", kinds)
        self.assertIn("decision.validation_passed", kinds)
        self.assertIn("decision.review_required", kinds)
        self.assertIn("decision.applied", kinds)
        self.assertEqual(len(agent.requests), 1)
        projected_fields = agent.requests[0].model_dump(mode="json")
        self.assertNotIn("allowed_actions", projected_fields)
        self.assertNotIn("graph", projected_fields)
        self.assertNotIn("state", projected_fields)
        self.assertNotIn("governance", projected_fields)
        self.assertEqual(len(workspace.governance_records), 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from datetime import datetime, timezone
import asyncio
import unittest
from uuid import UUID, uuid5

from pydantic import ValidationError

from adaptive_agent_runtime import AgentTask
from adaptive_agent_runtime.core.state import create_state, start_state
from adaptive_agent_runtime.decisioning import (
    ContextProjectionPolicy,
    ContextSensitivity,
    DecisionBasis,
    DecisionBudget,
    DecisionCorrelation,
    DecisionProducer,
    DecisionProposal,
    DecisionRequest,
    DecisionTarget,
    PolicyAgentContextBuilder,
    ProjectionSource,
    ProjectionSources,
    decision_fingerprint,
)
from adaptive_agent_runtime.llm import (
    PLANNING_INPUT_SOURCE_TYPE,
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CapabilityTurnResult,
    PlannerDecisionProposalProducer,
    PlanningGraphEffectNormalizer,
    TaskGraphDraft,
    TaskGraphDraftAdapter,
    TaskNodeDraft,
    InferenceCorrelation,
)
from adaptive_agent_runtime.orchestration import (
    PLANNING_DECISION_TYPE,
    DynamicTaskGraphPlanner,
    GraphInitializationApplier,
    InMemoryTaskGraphStore,
    PlanningDecisionPayload,
    PlanningExecutionPolicy,
    PlanningGraphLimits,
    PlanningStrategyDescriptor,
    PlanningStrategyRisk,
    RequiredPreparedTaskGraphStore,
)


NOW = datetime(2026, 8, 5, tzinfo=timezone.utc)


def linear_draft() -> TaskGraphDraft:
    return TaskGraphDraft(
        nodes=(
            TaskNodeDraft(
                node_key="research",
                goal="Research the goal",
                expected_output="Research evidence",
                requested_strategy_id="analysis",
            ),
            TaskNodeDraft(
                node_key="report",
                goal="Report the result",
                dependency_keys=("research",),
                expected_output="Final report",
                requested_strategy_id="analysis",
            ),
        ),
        rationale="Use a linear plan.",
    )


def parallel_draft() -> TaskGraphDraft:
    return TaskGraphDraft(
        nodes=(
            TaskNodeDraft(
                node_key="financial",
                goal="Analyze financial evidence",
                expected_output="Financial findings",
                requested_strategy_id="analysis",
            ),
            TaskNodeDraft(
                node_key="industry",
                goal="Analyze industry evidence",
                expected_output="Industry findings",
                requested_strategy_id="analysis",
            ),
            TaskNodeDraft(
                node_key="report",
                goal="Synthesize both branches",
                dependency_keys=("financial", "industry"),
                expected_output="Final report",
                requested_strategy_id="analysis",
            ),
        ),
        rationale="Use two parallel evidence branches.",
    )


def make_payload(
    *,
    risk: PlanningStrategyRisk = PlanningStrategyRisk.LOW,
) -> PlanningDecisionPayload:
    return PlanningDecisionPayload(
        goal="Analyze the same company",
        constraints=("Use only supplied strategies.",),
        strategies=(
            PlanningStrategyDescriptor(
                strategy_id="analysis",
                description="Bounded analysis strategy.",
                required_capability_ids=("read",),
                risk=risk,
            ),
        ),
        available_execution_capability_ids=("read",),
        graph_limits=PlanningGraphLimits(
            max_nodes=8,
            max_depth=5,
            max_fan_out=4,
        ),
        execution_policy=PlanningExecutionPolicy(
            model="test-planner",
            temperature=0.1,
            timeout_seconds=5.0,
        ),
    )


def make_request(
    request_id: UUID,
    *,
    payload: PlanningDecisionPayload | None = None,
) -> DecisionRequest[PlanningDecisionPayload]:
    selected = payload or make_payload()
    basis = DecisionBasis(
        snapshot_fingerprint=decision_fingerprint(
            {"goal": selected.goal, "catalog": selected.strategies}
        ),
        configuration_revision=1,
    )
    return DecisionRequest[PlanningDecisionPayload](
        request_id=request_id,
        decision_type=PLANNING_DECISION_TYPE,
        target=DecisionTarget(
            target_type="runtime_run_graph",
            target_id=str(UUID(int=request_id.int + 100)),
        ),
        correlation=DecisionCorrelation(
            run_id=UUID(int=request_id.int + 100),
            task_id=UUID(int=request_id.int + 200),
        ),
        basis=basis,
        payload=selected,
        allowed_actions=("graph.initialize",),
        budget=DecisionBudget(
            max_decision_cycles=1,
            max_agent_calls=1,
            max_revision_count=0,
            max_elapsed_seconds=5.0,
        ),
        created_at=NOW,
    )


def make_proposal(
    request: DecisionRequest[PlanningDecisionPayload],
    draft: TaskGraphDraft,
    *,
    confidence: float = 0.99,
) -> DecisionProposal[TaskGraphDraft]:
    return DecisionProposal[TaskGraphDraft](
        request_id=request.request_id,
        proposal_type=request.decision_type,
        producer=DecisionProducer(
            producer_id="fake-planner",
            capability="task_graph_proposal",
        ),
        input_snapshot_fingerprint=request.basis.snapshot_fingerprint,
        context_fingerprint="1" * 64,
        selected_action="graph.initialize",
        payload=draft,
        rationale=draft.rationale or "bounded test plan",
        confidence=confidence,
        created_at=NOW,
    )


class FakePlanner:
    module_id = "test.planner_agent"
    capability_id = "test.task_graph_proposal"

    def __init__(self, draft: TaskGraphDraft, *, delay: float = 0.0) -> None:
        self.draft = draft
        self.delay = delay
        self.calls = 0
        self.invocations: list[CapabilityInvocationMetadata | None] = []

    async def propose(self, request, *, invocation=None):
        del request
        self.calls += 1
        self.invocations.append(invocation)
        if self.delay:
            await asyncio.sleep(self.delay)
        return CapabilityTurnResult[TaskGraphDraft](
            kind=CapabilityTurnKind.COMPLETED,
            result=self.draft,
        )


def make_context(request: DecisionRequest[PlanningDecisionPayload]):
    payload = request.payload
    return PolicyAgentContextBuilder().build(
        request,
        ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="planning-input",
                    source_type=PLANNING_INPUT_SOURCE_TYPE,
                    agent_scope="planner",
                    content={
                        "goal": payload.goal,
                        "constraints": payload.constraints,
                        "available_strategies": tuple(
                            item.strategy_id for item in payload.strategies
                        ),
                        "available_execution_capability_ids": (
                            payload.available_execution_capability_ids
                        ),
                    },
                    sensitivity=ContextSensitivity.INTERNAL,
                    estimated_tokens=32,
                ),
            )
        ),
        ContextProjectionPolicy(
            policy_id="test.planner",
            version="1",
            agent_scope="planner",
            allowed_decision_types=frozenset({PLANNING_DECISION_TYPE}),
            allowed_source_types=frozenset({PLANNING_INPUT_SOURCE_TYPE}),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            max_items=1,
            max_context_tokens=128,
        ),
    )


class AdaptivePlanningTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_projects_deterministic_identity_and_exact_checkpoint(self) -> None:
        request = make_request(UUID(int=1))
        proposal = make_proposal(request, linear_draft())
        normalizer = PlanningGraphEffectNormalizer()

        first = normalizer.normalize(request, proposal)
        second = normalizer.normalize(request, proposal)

        self.assertEqual(first, second)
        effect = first.payload
        expected_graph_id = uuid5(request.request_id, "initial-graph")
        self.assertEqual(effect.graph.graph_id, expected_graph_id)
        for binding in effect.node_bindings:
            self.assertEqual(binding.node_id, uuid5(expected_graph_id, binding.node_key))

        store = InMemoryTaskGraphStore()
        await GraphInitializationApplier(store).apply(effect)
        checkpoint = await store.load(effect.run_id)
        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertEqual(checkpoint.graph, effect.graph)
        self.assertEqual(checkpoint.state_revision, 0)

        required_store = RequiredPreparedTaskGraphStore(
            store,
            run_id=effect.run_id,
            graph_id=effect.graph.graph_id,
        )
        planner = DynamicTaskGraphPlanner(
            effect.graph,
            graph_store=required_store,
        )
        state = start_state(
            create_state(AgentTask(description="same goal"), run_id=effect.run_id)
        )
        decision = await planner.plan(state)
        assert decision.action is not None
        self.assertEqual(
            decision.action.arguments["graph_id"],
            str(effect.graph.graph_id),
        )

    async def test_prepared_store_fails_closed_when_approved_graph_is_missing(self) -> None:
        store = RequiredPreparedTaskGraphStore(
            InMemoryTaskGraphStore(),
            run_id=UUID(int=1),
            graph_id=UUID(int=2),
        )
        with self.assertRaisesRegex(Exception, "approved initial"):
            await store.load(UUID(int=1))

    async def test_execution_policy_allows_one_call_and_hides_runtime_action(self) -> None:
        request = make_request(UUID(int=10))
        context = make_context(request)
        self.assertNotIn("allowed_actions", type(context).model_fields)
        planner = FakePlanner(linear_draft())
        producer = PlannerDecisionProposalProducer(
            capability=planner,
            execution_policy=request.payload.execution_policy,
            correlation=InferenceCorrelation(
                run_id=request.correlation.run_id,
                task_id=request.correlation.task_id,
            ),
        )
        result = await producer.propose(context)
        self.assertEqual(result.proposal.payload, linear_draft())
        self.assertEqual(result.proposal.selected_action, "graph.initialize")
        invocation = planner.invocations[0]
        assert invocation is not None
        self.assertEqual(invocation.trace_attributes["planning_max_retries"], 0)
        self.assertEqual(invocation.trace_attributes["planning_temperature"], 0.1)
        with self.assertRaisesRegex(RuntimeError, "one Agent call"):
            await producer.propose(context)

    async def test_execution_policy_times_out_without_retry(self) -> None:
        request = make_request(
            UUID(int=11),
            payload=make_payload().model_copy(
                update={
                    "execution_policy": PlanningExecutionPolicy(
                        model="test-planner",
                        temperature=0.0,
                        timeout_seconds=0.01,
                    )
                }
            ),
        )
        planner = FakePlanner(linear_draft(), delay=0.05)
        producer = PlannerDecisionProposalProducer(
            capability=planner,
            execution_policy=request.payload.execution_policy,
            correlation=InferenceCorrelation(run_id=request.correlation.run_id),
        )
        with self.assertRaises(TimeoutError):
            await producer.propose(make_context(request))
        self.assertEqual(planner.calls, 1)

    async def test_two_fake_planners_can_create_different_valid_graphs(self) -> None:
        effects = []
        for index, draft in enumerate((linear_draft(), parallel_draft()), start=20):
            request = make_request(UUID(int=index))
            context = make_context(request)
            planner = FakePlanner(draft)
            call = await PlannerDecisionProposalProducer(
                capability=planner,
                execution_policy=request.payload.execution_policy,
                correlation=InferenceCorrelation(run_id=request.correlation.run_id),
            ).propose(context)
            effects.append(
                PlanningGraphEffectNormalizer().normalize(
                    request,
                    call.proposal,
                )
            )

        first_graph, second_graph = (item.payload.graph for item in effects)
        self.assertEqual(len(first_graph.nodes), 2)
        self.assertEqual(len(second_graph.nodes), 3)
        self.assertEqual(
            len([node for node in first_graph.nodes if not node.dependencies]),
            1,
        )
        self.assertEqual(
            len([node for node in second_graph.nodes if not node.dependencies]),
            2,
        )
        self.assertNotEqual(effects[0].effect_fingerprint, effects[1].effect_fingerprint)

    async def test_agent_cannot_inject_execution_or_risk_authority(self) -> None:
        with self.assertRaises(ValidationError):
            TaskGraphDraft.model_validate(
                {
                    "nodes": [
                        {
                            "node_key": "node",
                            "goal": "Analyze",
                            "expected_output": "Result",
                            "requested_strategy_id": "analysis",
                        }
                    ],
                    "risk": "low",
                    "runtime_state_change": {"status": "completed"},
                }
            )
        injected = linear_draft().model_copy(
            update={
                "nodes": (
                    linear_draft().nodes[0].model_copy(
                        update={"goal": "execute_tool('admin')"}
                    ),
                    linear_draft().nodes[1],
                )
            }
        )
        with self.assertRaisesRegex(ValueError, "reserved Runtime directive"):
            TaskGraphDraftAdapter().adapt(injected)

        high_risk_request = make_request(
            UUID(int=30),
            payload=make_payload(risk=PlanningStrategyRisk.HIGH),
        )
        normalized = PlanningGraphEffectNormalizer().normalize(
            high_risk_request,
            make_proposal(high_risk_request, linear_draft(), confidence=1.0),
        )
        self.assertEqual(normalized.risk.value, "high")
        self.assertEqual(normalized.payload.risk_assessment.risk.value, "high")


if __name__ == "__main__":
    unittest.main()

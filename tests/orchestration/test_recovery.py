from __future__ import annotations

import unittest
from uuid import UUID

from adaptive_agent_runtime import (
    AgentRuntime,
    AgentState,
    AgentTask,
    InMemoryStateStore,
    InMemoryTraceSink,
    RunStatus,
)
from adaptive_agent_runtime.governance import (
    BoundGovernedOperation,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceAuthorizationIssuer,
    GovernanceHistory,
    GovernedOperationExecutor,
    InMemoryAuthorizationConsumptionStore,
    InMemoryHumanReviewService,
    RecoveryGovernanceAdapter,
    RuntimeGovernanceEvaluator,
    StrictAuthorizationVerifier,
    default_governance_policy,
)
from adaptive_agent_runtime.orchestration import (
    DeterministicFailureDrivenReplanner,
    DynamicTaskGraph,
    DynamicTaskGraphPlanner,
    FailureKind,
    NodeExecutionResult,
    RecoveryAction,
    RecoveryActionType,
    RecoveryPlan,
    StrategyActionExecutor,
    TaskNode,
    TaskNodeStatus,
    apply_recovery_plan,
)


class SequencedStrategy:
    module_id = "test.recovery.sequenced_strategy"
    strategy_id = "mock"

    def __init__(self, results: tuple[NodeExecutionResult, ...]) -> None:
        self._results = results
        self.calls: list[UUID] = []

    async def execute(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> NodeExecutionResult:
        del state
        self.calls.append(node.node_id)
        index = min(len(self.calls) - 1, len(self._results) - 1)
        return self._results[index]


class BackupStrategy:
    module_id = "test.recovery.backup_strategy"
    strategy_id = "backup"

    def __init__(self) -> None:
        self.calls: list[UUID] = []

    async def execute(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> NodeExecutionResult:
        del state
        self.calls.append(node.node_id)
        return NodeExecutionResult.ok(output={"strategy": self.strategy_id})


class GovernedRecoveryApplier:
    module_id = "test.recovery.governed_applier"

    def __init__(self) -> None:
        reviews = InMemoryHumanReviewService()
        self._governor = RuntimeGovernanceEvaluator(
            policy=default_governance_policy(),
            rule_evaluator=DeterministicRuleEvaluator(),
            confidence_evaluator=DeterministicConfidenceEvaluator(),
            review_service=reviews,
        )
        self._issuer = GovernanceAuthorizationIssuer()
        self.store = InMemoryAuthorizationConsumptionStore()
        self._executor = GovernedOperationExecutor(
            verifier=StrictAuthorizationVerifier(),
            consumption_store=self.store,
        )
        self.applied_plans: list[RecoveryPlan] = []

    async def apply(
        self,
        graph: DynamicTaskGraph,
        plan: RecoveryPlan,
        *,
        state: AgentState,
    ) -> DynamicTaskGraph:
        request = RecoveryGovernanceAdapter().to_request(
            plan,
            run_id=state.run_id,
            task_id=state.task.task_id,
            action_id=plan.analysis.action_id,
            history=GovernanceHistory(successful_similar=5),
        )
        decision = self._governor.evaluate(request)
        authorization = self._issuer.issue(request, decision)

        async def apply() -> DynamicTaskGraph:
            self.applied_plans.append(plan)
            return apply_recovery_plan(graph, plan)

        return await self._executor.execute(
            request=request,
            decision=decision,
            authorization=authorization,
            target=BoundGovernedOperation(
                module_id="test.recovery.target",
                operation=request.operation,
                target=request.target,
                subject=plan,
                apply=apply,
            ),
        )


def node(
    goal: str,
    *,
    dependencies: tuple[UUID, ...] = (),
    strategy_id: str = "mock",
) -> TaskNode:
    return TaskNode(
        goal=goal,
        dependencies=dependencies,
        expected_output=f"{goal} output",
        strategy_id=strategy_id,
    )


class FailureDrivenReplanningTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_failure_replans_graph_and_continues(self) -> None:
        root = node("transient root")
        child = node("dependent", dependencies=(root.node_id,))
        strategy = SequencedStrategy(
            (
                NodeExecutionResult.failed(error="temporary timeout"),
                NodeExecutionResult.ok(output={"recovered": True}),
                NodeExecutionResult.ok(output={"child": True}),
            )
        )
        applier = GovernedRecoveryApplier()
        planner = DynamicTaskGraphPlanner(
            DynamicTaskGraph(nodes=(root, child)),
            recovery_planner=DeterministicFailureDrivenReplanner(
                max_attempts_per_node=2
            ),
            recovery_applier=applier,
        )
        runtime = AgentRuntime(
            planner=planner,
            executor=StrategyActionExecutor((strategy,)),
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
        )

        result = await runtime.run(AgentTask(description="recover transient"))

        self.assertEqual(result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(strategy.calls, [root.node_id, root.node_id, child.node_id])
        records = planner.recovery_records_for(result.final_state.run_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].plan.analysis.kind, FailureKind.TRANSIENT)
        self.assertEqual(
            records[0].plan.actions[0].action_type,
            RecoveryActionType.RETRY_NODE,
        )
        self.assertEqual(applier.applied_plans, [records[0].plan])

    async def test_unavailable_strategy_is_replaced_not_retried(self) -> None:
        task = node("replace strategy", strategy_id="primary")
        backup = BackupStrategy()
        applier = GovernedRecoveryApplier()
        planner = DynamicTaskGraphPlanner(
            DynamicTaskGraph(nodes=(task,)),
            recovery_planner=DeterministicFailureDrivenReplanner(
                alternate_strategies={"primary": "backup"},
            ),
            recovery_applier=applier,
        )
        result = await AgentRuntime(
            planner=planner,
            executor=StrategyActionExecutor((backup,)),
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
        ).run(AgentTask(description="replace unavailable strategy"))

        self.assertEqual(result.final_state.status, RunStatus.COMPLETED)
        final_node = planner.graph_for(result.final_state.run_id).get_node(
            task.node_id
        )
        self.assertEqual(final_node.strategy_id, "backup")
        self.assertEqual(backup.calls, [task.node_id])
        record = planner.recovery_records_for(result.final_state.run_id)[0]
        self.assertEqual(
            record.plan.actions[0].action_type,
            RecoveryActionType.REPLACE_STRATEGY,
        )

    async def test_unknown_failure_aborts_without_graph_mutation(self) -> None:
        task = node("unknown failure")
        strategy = SequencedStrategy(
            (NodeExecutionResult.failed(error="unexpected domain failure"),)
        )
        applier = GovernedRecoveryApplier()
        planner = DynamicTaskGraphPlanner(
            DynamicTaskGraph(nodes=(task,)),
            recovery_planner=DeterministicFailureDrivenReplanner(),
            recovery_applier=applier,
        )
        result = await AgentRuntime(
            planner=planner,
            executor=StrategyActionExecutor((strategy,)),
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
        ).run(AgentTask(description="abort unknown failure"))

        self.assertEqual(result.final_state.status, RunStatus.FAILED)
        record = planner.recovery_records_for(result.final_state.run_id)[0]
        self.assertTrue(record.plan.aborts)
        self.assertEqual(applier.applied_plans, [])

    async def test_recovery_budget_stops_repeated_transient_failure(self) -> None:
        task = node("persistent timeout")
        strategy = SequencedStrategy(
            (NodeExecutionResult.failed(error="temporary timeout"),)
        )
        applier = GovernedRecoveryApplier()
        planner = DynamicTaskGraphPlanner(
            DynamicTaskGraph(nodes=(task,)),
            recovery_planner=DeterministicFailureDrivenReplanner(
                max_attempts_per_node=2
            ),
            recovery_applier=applier,
        )
        result = await AgentRuntime(
            planner=planner,
            executor=StrategyActionExecutor((strategy,)),
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
            max_steps=8,
        ).run(AgentTask(description="bounded recovery"))

        self.assertEqual(result.final_state.status, RunStatus.FAILED)
        self.assertEqual(len(strategy.calls), 3)
        records = planner.recovery_records_for(result.final_state.run_id)
        self.assertEqual(len(records), 3)
        self.assertTrue(records[-1].plan.aborts)
        self.assertEqual(len(applier.applied_plans), 2)


class RecoveryGraphTransitionTests(unittest.TestCase):
    def test_recovery_node_supersedes_failure_and_rewires_dependents(self) -> None:
        failed = node("failed")
        dependent = node("dependent", dependencies=(failed.node_id,))
        graph = DynamicTaskGraph(nodes=(failed, dependent)).mark_running(
            failed.node_id
        )
        from adaptive_agent_runtime import Observation

        observation = Observation.failed(UUID(int=500), error="fatal source")
        graph = graph.resolve_node(failed.node_id, observation)
        fallback = node("fallback")

        recovered = graph.add_recovery_node(failed.node_id, fallback)

        self.assertEqual(
            recovered.get_node(failed.node_id).status,
            TaskNodeStatus.RECOVERED,
        )
        self.assertEqual(
            recovered.get_node(dependent.node_id).dependencies,
            (fallback.node_id,),
        )
        self.assertEqual(
            recovered.get_node(dependent.node_id).status,
            TaskNodeStatus.PENDING,
        )


if __name__ == "__main__":
    unittest.main()

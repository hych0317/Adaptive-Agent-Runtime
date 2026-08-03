from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest
from uuid import UUID, uuid4

from adaptive_agent_runtime import (
    AgentRuntime,
    AgentState,
    AgentTask,
    CoreEventKind,
    RunStatus,
    RuntimeResumeBlockedError,
)
from adaptive_agent_runtime.orchestration import (
    DeterministicFailureDrivenReplanner,
    DynamicTaskGraph,
    DynamicTaskGraphPlanner,
    MockExecutionStrategy,
    NodeExecutionResult,
    StrategyActionExecutor,
    TaskNode,
    TaskNodeStatus,
    RecoveryPlan,
    apply_recovery_plan,
)
from adaptive_agent_runtime.governance import (
    BoundGovernedOperation,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceAuthorizationIssuer,
    GovernanceHistory,
    GovernedOperationExecutor,
    RecoveryGovernanceAdapter,
    RuntimeGovernanceEvaluator,
    StrictAuthorizationVerifier,
    default_governance_policy,
)
from adaptive_agent_runtime.persistence import SQLitePersistence


class SimulatedProcessCrash(BaseException):
    pass


class CrashAfterFirstObservationStore:
    module_id = "test.crash_after_observation"

    def __init__(self, delegate: object) -> None:
        self._delegate = delegate
        self._crashed = False

    async def save(self, state: AgentState) -> None:
        await self._delegate.save(state)  # type: ignore[attr-defined]
        if (
            not self._crashed
            and state.status is RunStatus.RUNNING
            and state.step_count == 1
        ):
            self._crashed = True
            raise SimulatedProcessCrash()

    async def load(self, run_id: UUID) -> AgentState | None:
        return await self._delegate.load(run_id)  # type: ignore[attr-defined,no-any-return]


class CrashBeforeObservationStrategy(MockExecutionStrategy):
    async def execute(self, node: TaskNode, state: AgentState):  # type: ignore[no-untyped-def]
        del node, state
        raise SimulatedProcessCrash()


class SQLiteGovernedRecoveryApplier:
    module_id = "test.persistence.recovery_applier"

    def __init__(self, persistence: SQLitePersistence) -> None:
        self._governor = RuntimeGovernanceEvaluator(
            policy=default_governance_policy(),
            rule_evaluator=DeterministicRuleEvaluator(),
            confidence_evaluator=DeterministicConfidenceEvaluator(),
            review_service=persistence.human_review_service,
        )
        self._issuer = GovernanceAuthorizationIssuer()
        self._executor = GovernedOperationExecutor(
            verifier=StrictAuthorizationVerifier(),
            consumption_store=persistence.authorization_store,
        )

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
            return apply_recovery_plan(graph, plan)

        return await self._executor.execute(
            request=request,
            decision=decision,
            authorization=authorization,
            target=BoundGovernedOperation(
                module_id="test.persistence.recovery_target",
                operation=request.operation,
                target=request.target,
                subject=plan,
                apply=apply,
            ),
        )


def task_node(
    goal: str,
    *,
    dependencies: tuple[UUID, ...] = (),
) -> TaskNode:
    return TaskNode(
        goal=goal,
        dependencies=dependencies,
        expected_output=f"{goal} output",
        strategy_id="mock",
    )


class SQLiteRuntimeResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_reopens_database_and_continues_at_next_graph_node(self) -> None:
        first = task_node("first")
        second = task_node("second", dependencies=(first.node_id,))
        template = DynamicTaskGraph(nodes=(first, second))
        run_id = uuid4()

        with TemporaryDirectory() as directory:
            path = f"{directory}/runtime.sqlite3"
            first_persistence = SQLitePersistence(path)
            first_strategy = MockExecutionStrategy()
            first_runtime = AgentRuntime(
                planner=DynamicTaskGraphPlanner(
                    template,
                    graph_store=first_persistence.task_graph_store,
                ),
                executor=StrategyActionExecutor((first_strategy,)),
                state_store=CrashAfterFirstObservationStore(
                    first_persistence.state_store
                ),
                trace_sink=first_persistence.trace_sink,
            )

            with self.assertRaises(SimulatedProcessCrash):
                await first_runtime.run(
                    AgentTask(description="restartable graph"),
                    run_id=run_id,
                )
            self.assertEqual(first_strategy.executed_node_ids, [first.node_id])
            persisted = await first_persistence.state_store.load(run_id)
            self.assertIsNotNone(persisted)
            assert persisted is not None
            self.assertEqual(persisted.step_count, 1)
            first_persistence.close()

            resumed_persistence = SQLitePersistence(path)
            resumed_strategy = MockExecutionStrategy()
            resumed_planner = DynamicTaskGraphPlanner(
                template,
                graph_store=resumed_persistence.task_graph_store,
            )
            resumed_runtime = AgentRuntime(
                planner=resumed_planner,
                executor=StrategyActionExecutor((resumed_strategy,)),
                state_store=resumed_persistence.state_store,
                trace_sink=resumed_persistence.trace_sink,
            )

            result = await resumed_runtime.resume(run_id)

            self.assertEqual(result.final_state.status, RunStatus.COMPLETED)
            self.assertEqual(result.final_state.step_count, 2)
            self.assertEqual(resumed_strategy.executed_node_ids, [second.node_id])
            graph = resumed_planner.graph_for(run_id)
            self.assertTrue(
                all(node.status is TaskNodeStatus.COMPLETED for node in graph.nodes)
            )
            entries = await resumed_persistence.trace_sink.entries_for(run_id)
            history = await resumed_persistence.state_store.history_for(run_id)
            checkpoint = await resumed_persistence.task_graph_store.load(run_id)
            resumed_persistence.close()
            self.assertEqual(
                [entry.sequence for entry in entries],
                list(range(1, len(entries) + 1)),
            )
            self.assertEqual(
                sum(
                    entry.event.kind == CoreEventKind.RUNTIME_RESUMED
                    for entry in entries
                ),
                1,
            )
            self.assertEqual(
                [state.revision for state in history],
                list(range(result.final_state.revision + 1)),
            )
            self.assertIsNotNone(checkpoint)
            assert checkpoint is not None
            self.assertEqual(checkpoint.graph, graph)
            self.assertEqual(checkpoint.in_flight, ())

    async def test_does_not_replay_an_action_with_unknown_outcome(self) -> None:
        only = task_node("external side effect")
        template = DynamicTaskGraph(nodes=(only,))
        run_id = uuid4()

        with TemporaryDirectory() as directory:
            path = f"{directory}/runtime.sqlite3"
            first_persistence = SQLitePersistence(path)
            runtime = AgentRuntime(
                planner=DynamicTaskGraphPlanner(
                    template,
                    graph_store=first_persistence.task_graph_store,
                ),
                executor=StrategyActionExecutor(
                    (CrashBeforeObservationStrategy(),)
                ),
                state_store=first_persistence.state_store,
                trace_sink=first_persistence.trace_sink,
            )
            with self.assertRaises(SimulatedProcessCrash):
                await runtime.run(
                    AgentTask(description="in-doubt action"),
                    run_id=run_id,
                )
            first_persistence.close()

            resumed_persistence = SQLitePersistence(path)
            replay_strategy = MockExecutionStrategy()
            resumed = AgentRuntime(
                planner=DynamicTaskGraphPlanner(
                    template,
                    graph_store=resumed_persistence.task_graph_store,
                ),
                executor=StrategyActionExecutor((replay_strategy,)),
                state_store=resumed_persistence.state_store,
                trace_sink=resumed_persistence.trace_sink,
            )

            with self.assertRaises(RuntimeResumeBlockedError):
                await resumed.resume(run_id)

            self.assertEqual(replay_strategy.executed_node_ids, [])
            state = await resumed_persistence.state_store.load(run_id)
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.status, RunStatus.RUNNING)
            self.assertEqual(state.step_count, 0)
            checkpoint = await resumed_persistence.task_graph_store.load(run_id)
            self.assertIsNotNone(checkpoint)
            assert checkpoint is not None
            self.assertEqual(len(checkpoint.in_flight), 1)
            resumed_persistence.close()

    async def test_resume_classifies_failure_replans_and_persists_recovery(
        self,
    ) -> None:
        failed_then_retried = task_node("restart recovery")
        template = DynamicTaskGraph(nodes=(failed_then_retried,))
        run_id = uuid4()

        with TemporaryDirectory() as directory:
            path = f"{directory}/recovery.sqlite3"
            first = SQLitePersistence(path)
            failing = MockExecutionStrategy(
                {
                    failed_then_retried.node_id: NodeExecutionResult.failed(
                        error="temporary timeout"
                    )
                }
            )
            initial_runtime = AgentRuntime(
                planner=DynamicTaskGraphPlanner(
                    template,
                    graph_store=first.task_graph_store,
                ),
                executor=StrategyActionExecutor((failing,)),
                state_store=CrashAfterFirstObservationStore(first.state_store),
                trace_sink=first.trace_sink,
            )
            with self.assertRaises(SimulatedProcessCrash):
                await initial_runtime.run(
                    AgentTask(description="resume into recovery"),
                    run_id=run_id,
                )
            first.close()

            reopened = SQLitePersistence(path)
            successful = MockExecutionStrategy()
            planner = DynamicTaskGraphPlanner(
                template,
                graph_store=reopened.task_graph_store,
                recovery_planner=DeterministicFailureDrivenReplanner(),
                recovery_applier=SQLiteGovernedRecoveryApplier(reopened),
            )
            result = await AgentRuntime(
                planner=planner,
                executor=StrategyActionExecutor((successful,)),
                state_store=reopened.state_store,
                trace_sink=reopened.trace_sink,
            ).resume(run_id)

            checkpoint = await reopened.task_graph_store.load(run_id)
            self.assertIsNotNone(checkpoint)
            assert checkpoint is not None
            self.assertEqual(result.final_state.status, RunStatus.COMPLETED)
            self.assertEqual(result.final_state.step_count, 2)
            self.assertEqual(
                successful.executed_node_ids,
                [failed_then_retried.node_id],
            )
            self.assertEqual(len(checkpoint.recovery_records), 1)
            self.assertEqual(
                checkpoint.recovery_attempts,
                ((failed_then_retried.node_id, 1),),
            )
            reopened.close()


if __name__ == "__main__":
    unittest.main()

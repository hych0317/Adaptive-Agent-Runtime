from __future__ import annotations

import unittest
from uuid import UUID, uuid4

from adaptive_agent_runtime import (
    AgentState,
    AgentRuntime,
    AgentTask,
    InMemoryStateStore,
    InMemoryTraceSink,
    RunResult,
    RunStatus,
)
from adaptive_agent_runtime.orchestration import (
    DynamicTaskGraph,
    DynamicTaskGraphPlanner,
    GraphMutation,
    ReadyTaskNodeSelector,
    MockExecutionStrategy,
    NodeExecutionResult,
    StrategyActionExecutor,
    TaskNode,
    TaskNodeStatus,
)


class LastReadySelector:
    module_id = "test.ready_node_selector.last"

    def __init__(self) -> None:
        self.ready_sets: list[tuple[UUID, ...]] = []

    async def select_node_id(
        self,
        ready_nodes: tuple[TaskNode, ...],
        state: AgentState,
    ) -> UUID:
        del state
        self.ready_sets.append(tuple(node.node_id for node in ready_nodes))
        return ready_nodes[-1].node_id


class EscapingReadySelector:
    module_id = "test.ready_node_selector.escape"

    async def select_node_id(
        self,
        ready_nodes: tuple[TaskNode, ...],
        state: AgentState,
    ) -> UUID:
        del ready_nodes, state
        return uuid4()


class RaisingExecutionStrategy:
    module_id = "test.raising_strategy"
    strategy_id = "mock"

    async def execute(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> NodeExecutionResult:
        del node, state
        raise RuntimeError("strategy exploded")


def node(
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


async def run_graph(
    graph: DynamicTaskGraph,
    strategy: MockExecutionStrategy,
) -> tuple[RunResult, DynamicTaskGraphPlanner]:
    planner = DynamicTaskGraphPlanner(graph)
    runtime = AgentRuntime(
        planner=planner,
        executor=StrategyActionExecutor((strategy,)),
        state_store=InMemoryStateStore(),
        trace_sink=InMemoryTraceSink(),
        max_steps=32,
    )
    result = await runtime.run(AgentTask(description="execute task graph"))
    return result, planner


class DynamicTaskGraphPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_executes_a_linear_graph(self) -> None:
        first = node("first")
        second = node("second", dependencies=(first.node_id,))
        third = node("third", dependencies=(second.node_id,))
        strategy = MockExecutionStrategy()

        result, planner = await run_graph(
            DynamicTaskGraph(nodes=(first, second, third)),
            strategy,
        )

        self.assertEqual(result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(result.final_state.step_count, 3)
        self.assertEqual(
            strategy.executed_node_ids,
            [first.node_id, second.node_id, third.node_id],
        )
        final_graph = planner.graph_for(result.final_state.run_id)
        self.assertTrue(
            all(
                task.status is TaskNodeStatus.COMPLETED
                for task in final_graph.nodes
            )
        )

    async def test_executes_branch_nodes_before_their_join(self) -> None:
        root = node("root")
        left = node("left", dependencies=(root.node_id,))
        right = node("right", dependencies=(root.node_id,))
        join = node("join", dependencies=(left.node_id, right.node_id))
        strategy = MockExecutionStrategy()

        result, _ = await run_graph(
            DynamicTaskGraph(nodes=(root, left, right, join)),
            strategy,
        )

        self.assertEqual(result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(result.final_state.step_count, 4)
        self.assertEqual(
            strategy.executed_node_ids,
            [root.node_id, left.node_id, right.node_id, join.node_id],
        )

    async def test_optional_selector_proposes_only_within_runtime_ready_set(
        self,
    ) -> None:
        root = node("root")
        left = node("left", dependencies=(root.node_id,))
        right = node("right", dependencies=(root.node_id,))
        selector = LastReadySelector()
        planner = DynamicTaskGraphPlanner(
            DynamicTaskGraph(nodes=(root, left, right)),
            ready_node_selector=selector,
        )
        strategy = MockExecutionStrategy()
        runtime = AgentRuntime(
            planner=planner,
            executor=StrategyActionExecutor((strategy,)),
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
        )

        result = await runtime.run(AgentTask(description="select ready nodes"))

        self.assertEqual(result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(
            strategy.executed_node_ids,
            [root.node_id, right.node_id, left.node_id],
        )
        self.assertIsInstance(selector, ReadyTaskNodeSelector)
        self.assertEqual(
            selector.ready_sets[1],
            (left.node_id, right.node_id),
        )

    async def test_selector_cannot_escape_runtime_ready_set(self) -> None:
        ready = node("ready")
        planner = DynamicTaskGraphPlanner(
            DynamicTaskGraph(nodes=(ready,)),
            ready_node_selector=EscapingReadySelector(),
        )
        strategy = MockExecutionStrategy()
        runtime = AgentRuntime(
            planner=planner,
            executor=StrategyActionExecutor((strategy,)),
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
        )

        result = await runtime.run(AgentTask(description="reject escape"))

        self.assertEqual(result.final_state.status, RunStatus.FAILED)
        self.assertEqual(strategy.executed_node_ids, [])
        self.assertIn("outside the ready set", result.final_state.error or "")

    async def test_execution_can_add_a_dynamic_task_node(self) -> None:
        root = node("discover follow-up")
        follow_up = node("dynamic follow-up", dependencies=(root.node_id,))
        strategy = MockExecutionStrategy(
            {
                root.node_id: NodeExecutionResult.ok(
                    output="follow-up discovered",
                    mutations=(GraphMutation.add_node(follow_up),),
                )
            }
        )

        result, planner = await run_graph(
            DynamicTaskGraph(nodes=(root,)),
            strategy,
        )

        self.assertEqual(result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(result.final_state.step_count, 2)
        self.assertEqual(
            strategy.executed_node_ids,
            [root.node_id, follow_up.node_id],
        )
        final_graph = planner.graph_for(result.final_state.run_id)
        self.assertEqual(len(final_graph.nodes), 2)
        self.assertEqual(
            final_graph.get_node(follow_up.node_id).status,
            TaskNodeStatus.COMPLETED,
        )

    async def test_failed_node_blocks_dependents_and_fails_run(self) -> None:
        root = node("failing root")
        child = node("blocked child", dependencies=(root.node_id,))
        strategy = MockExecutionStrategy(
            {
                root.node_id: NodeExecutionResult.failed(
                    error="mock execution failed"
                )
            }
        )

        result, planner = await run_graph(
            DynamicTaskGraph(nodes=(root, child)),
            strategy,
        )

        self.assertEqual(result.final_state.status, RunStatus.FAILED)
        self.assertEqual(result.final_state.step_count, 1)
        final_graph = planner.graph_for(result.final_state.run_id)
        self.assertEqual(
            final_graph.get_node(root.node_id).status,
            TaskNodeStatus.FAILED,
        )
        self.assertEqual(
            final_graph.get_node(child.node_id).status,
            TaskNodeStatus.BLOCKED,
        )
        self.assertEqual(strategy.executed_node_ids, [root.node_id])

    async def test_strategy_exception_becomes_a_failed_node(self) -> None:
        task = node("strategy failure")
        graph = DynamicTaskGraph(nodes=(task,))
        planner = DynamicTaskGraphPlanner(graph)
        runtime = AgentRuntime(
            planner=planner,
            executor=StrategyActionExecutor((RaisingExecutionStrategy(),)),
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
        )

        result = await runtime.run(AgentTask(description="strategy exception"))

        self.assertEqual(result.final_state.status, RunStatus.FAILED)
        final_node = planner.graph_for(result.final_state.run_id).get_node(
            task.node_id
        )
        self.assertEqual(final_node.status, TaskNodeStatus.FAILED)
        self.assertIn("strategy exploded", final_node.failure_reason or "")
        self.assertNotIn("executor module", result.final_state.error or "")

    async def test_invalid_dynamic_mutation_becomes_a_failed_node(self) -> None:
        root = node("invalid mutation source")
        invalid = node("invalid dynamic node", dependencies=(root.node_id, uuid4()))
        strategy = MockExecutionStrategy(
            {
                root.node_id: NodeExecutionResult.ok(
                    output="invalid proposal",
                    mutations=(GraphMutation.add_node(invalid),),
                )
            }
        )

        result, planner = await run_graph(
            DynamicTaskGraph(nodes=(root,)),
            strategy,
        )

        self.assertEqual(result.final_state.status, RunStatus.FAILED)
        final_graph = planner.graph_for(result.final_state.run_id)
        self.assertEqual(
            final_graph.get_node(root.node_id).status,
            TaskNodeStatus.FAILED,
        )
        self.assertEqual(len(final_graph.nodes), 1)
        self.assertIn("mutation rejected", result.final_state.error or "")

    async def test_planner_creates_an_isolated_graph_for_each_run(self) -> None:
        task = node("shared template node")
        planner = DynamicTaskGraphPlanner(DynamicTaskGraph(nodes=(task,)))
        strategy = MockExecutionStrategy()
        runtime = AgentRuntime(
            planner=planner,
            executor=StrategyActionExecutor((strategy,)),
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
        )

        first = await runtime.run(AgentTask(description="first run"))
        second = await runtime.run(AgentTask(description="second run"))

        first_graph = planner.graph_for(first.final_state.run_id)
        second_graph = planner.graph_for(second.final_state.run_id)
        self.assertNotEqual(first_graph.graph_id, second_graph.graph_id)
        self.assertEqual(first.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(second.final_state.status, RunStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()

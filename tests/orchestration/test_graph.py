from __future__ import annotations

import unittest
from uuid import UUID, uuid4

from pydantic import ValidationError

from adaptive_agent_runtime import Observation
from adaptive_agent_runtime.orchestration import (
    DynamicTaskGraph,
    GraphMutation,
    GraphScheduler,
    TaskNode,
    TaskNodeStatus,
)


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


class DynamicTaskGraphTests(unittest.TestCase):
    def test_scheduler_exposes_parallel_ready_branch(self) -> None:
        root = node("root")
        left = node("left", dependencies=(root.node_id,))
        right = node("right", dependencies=(root.node_id,))
        join = node("join", dependencies=(left.node_id, right.node_id))
        graph = DynamicTaskGraph(nodes=(root, left, right, join))
        scheduler = GraphScheduler()

        running = graph.mark_running(root.node_id)
        completed = running.resolve_node(
            root.node_id,
            Observation.ok(uuid4(), output="root complete"),
        )

        self.assertEqual(
            [ready.node_id for ready in scheduler.ready_nodes(completed)],
            [left.node_id, right.node_id],
        )

    def test_add_node_mutation_returns_a_new_graph(self) -> None:
        root = node("root")
        graph = DynamicTaskGraph(nodes=(root,))
        dynamic = node("dynamic", dependencies=(root.node_id,))

        updated = graph.apply_mutation(GraphMutation.add_node(dynamic))

        self.assertIsNot(updated, graph)
        self.assertEqual(graph.version, 0)
        self.assertEqual(updated.version, 1)
        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(updated.get_node(dynamic.node_id), dynamic)

    def test_failed_dependency_is_propagated_as_blocked(self) -> None:
        root = node("root")
        child = node("child", dependencies=(root.node_id,))
        graph = DynamicTaskGraph(nodes=(root, child)).mark_running(root.node_id)
        graph = graph.resolve_node(
            root.node_id,
            Observation.failed(uuid4(), error="root failed"),
        )

        updated = GraphScheduler().propagate_failed_dependencies(graph)

        self.assertEqual(
            updated.get_node(root.node_id).status,
            TaskNodeStatus.FAILED,
        )
        self.assertEqual(
            updated.get_node(child.node_id).status,
            TaskNodeStatus.BLOCKED,
        )

    def test_graph_rejects_unknown_dependencies(self) -> None:
        with self.assertRaises(ValidationError):
            DynamicTaskGraph(
                nodes=(node("invalid", dependencies=(uuid4(),)),)
            )

    def test_graph_rejects_dependency_cycles(self) -> None:
        first_id = uuid4()
        second_id = uuid4()
        first = TaskNode(
            node_id=first_id,
            goal="first",
            dependencies=(second_id,),
            expected_output="first output",
            strategy_id="mock",
        )
        second = TaskNode(
            node_id=second_id,
            goal="second",
            dependencies=(first_id,),
            expected_output="second output",
            strategy_id="mock",
        )

        with self.assertRaises(ValidationError):
            DynamicTaskGraph(nodes=(first, second))

    def test_dependency_mutation_cannot_introduce_a_cycle(self) -> None:
        first = node("first")
        second = node("second", dependencies=(first.node_id,))
        graph = DynamicTaskGraph(nodes=(first, second))

        with self.assertRaises(ValidationError):
            graph.apply_mutation(
                GraphMutation.add_dependency(first.node_id, second.node_id)
            )

        self.assertEqual(graph.version, 0)
        self.assertEqual(graph.get_node(first.node_id).dependencies, ())

    def test_graph_rejects_resolved_node_with_incomplete_dependency(self) -> None:
        dependency = node("dependency")
        completed = TaskNode(
            goal="invalid completed child",
            dependencies=(dependency.node_id,),
            expected_output="child output",
            strategy_id="mock",
            status=TaskNodeStatus.COMPLETED,
            observation=Observation.ok(uuid4(), output="complete"),
        )

        with self.assertRaises(ValidationError):
            DynamicTaskGraph(nodes=(dependency, completed))


if __name__ == "__main__":
    unittest.main()

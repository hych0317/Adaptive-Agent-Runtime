"""Dependency-aware scheduling for DynamicTaskGraph snapshots."""

from __future__ import annotations

from adaptive_agent_runtime.orchestration.graph import DynamicTaskGraph
from adaptive_agent_runtime.orchestration.models import TaskNode, TaskNodeStatus


class GraphScheduler:
    """Return every currently ready node in stable graph order."""

    module_id = "orchestration.graph_scheduler"

    def ready_nodes(self, graph: DynamicTaskGraph) -> tuple[TaskNode, ...]:
        ready: list[TaskNode] = []
        for node in graph.nodes:
            if node.status is not TaskNodeStatus.PENDING:
                continue
            if all(
                graph.get_node(dependency_id).status
                is TaskNodeStatus.COMPLETED
                for dependency_id in node.dependencies
            ):
                ready.append(node)
        return tuple(ready)

    def propagate_failed_dependencies(
        self,
        graph: DynamicTaskGraph,
    ) -> DynamicTaskGraph:
        """Mark descendants of failed or blocked nodes as blocked."""

        current = graph
        while True:
            blocked_node: TaskNode | None = None
            failed_dependencies: tuple[TaskNode, ...] = ()
            for node in current.nodes:
                if node.status is not TaskNodeStatus.PENDING:
                    continue
                failed_dependencies = tuple(
                    current.get_node(dependency_id)
                    for dependency_id in node.dependencies
                    if current.get_node(dependency_id).status
                    in {TaskNodeStatus.FAILED, TaskNodeStatus.BLOCKED}
                )
                if failed_dependencies:
                    blocked_node = node
                    break

            if blocked_node is None:
                return current

            dependency_text = ", ".join(
                dependency.goal for dependency in failed_dependencies
            )
            current = current.mark_blocked(
                blocked_node.node_id,
                reason=f"blocked by failed dependencies: {dependency_text}",
            )


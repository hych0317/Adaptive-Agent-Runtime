"""Immutable Dynamic Task Graph and its state transitions."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from adaptive_agent_runtime import Observation
from adaptive_agent_runtime.orchestration.errors import GraphTransitionError
from adaptive_agent_runtime.orchestration.models import (
    GraphMutation,
    GraphMutationType,
    OrchestrationModel,
    TaskNode,
    TaskNodeStatus,
)


class DynamicTaskGraph(OrchestrationModel):
    """An immutable DAG whose nodes are goal units."""

    graph_id: UUID = Field(default_factory=uuid4)
    version: int = Field(default=0, ge=0)
    nodes: tuple[TaskNode, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_graph(self) -> DynamicTaskGraph:
        node_ids = [node.node_id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("task graph node_id values must be unique")

        known_ids = set(node_ids)
        for node in self.nodes:
            missing = set(node.dependencies) - known_ids
            if missing:
                missing_text = ", ".join(sorted(str(item) for item in missing))
                raise ValueError(
                    f"node '{node.node_id}' has unknown dependencies: {missing_text}"
                )

        dependencies = {
            node.node_id: node.dependencies
            for node in self.nodes
        }
        visiting: set[UUID] = set()
        visited: set[UUID] = set()

        def visit(node_id: UUID) -> None:
            if node_id in visiting:
                raise ValueError("task graph dependencies must be acyclic")
            if node_id in visited:
                return
            visiting.add(node_id)
            for dependency_id in dependencies[node_id]:
                visit(dependency_id)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in node_ids:
            visit(node_id)

        status_by_id = {
            node.node_id: node.status
            for node in self.nodes
        }
        dependency_bound_statuses = {
            TaskNodeStatus.RUNNING,
            TaskNodeStatus.COMPLETED,
            TaskNodeStatus.FAILED,
            TaskNodeStatus.RECOVERED,
        }
        for node in self.nodes:
            dependency_statuses = tuple(
                status_by_id[dependency_id]
                for dependency_id in node.dependencies
            )
            if node.status in dependency_bound_statuses and any(
                status is not TaskNodeStatus.COMPLETED
                for status in dependency_statuses
            ):
                raise ValueError(
                    "a started or resolved node requires completed dependencies"
                )
            if node.status is TaskNodeStatus.BLOCKED and not any(
                status in {TaskNodeStatus.FAILED, TaskNodeStatus.BLOCKED}
                for status in dependency_statuses
            ):
                raise ValueError(
                    "a blocked node requires a failed or blocked dependency"
                )
        return self

    def get_node(self, node_id: UUID) -> TaskNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise GraphTransitionError(f"task node '{node_id}' does not exist")

    def apply_mutation(self, mutation: GraphMutation) -> DynamicTaskGraph:
        if mutation.mutation_type is GraphMutationType.ADD_NODE:
            if mutation.node is None:
                raise GraphTransitionError("add_node mutation has no task node")
            return self._add_node(mutation.node)

        if mutation.node_id is None or mutation.dependency_id is None:
            raise GraphTransitionError(
                "add_dependency mutation is missing dependency identifiers"
            )
        return self._add_dependency(
            mutation.node_id,
            mutation.dependency_id,
        )

    def mark_running(self, node_id: UUID) -> DynamicTaskGraph:
        node = self.get_node(node_id)
        if node.status is not TaskNodeStatus.PENDING:
            raise GraphTransitionError(
                f"only a pending node can run; '{node_id}' is {node.status.value}"
            )
        incomplete = [
            dependency_id
            for dependency_id in node.dependencies
            if self.get_node(dependency_id).status is not TaskNodeStatus.COMPLETED
        ]
        if incomplete:
            raise GraphTransitionError(
                f"node '{node_id}' has incomplete dependencies"
            )
        updated = self._updated_node(node, status=TaskNodeStatus.RUNNING)
        return self._replace_node(updated)

    def resolve_node(
        self,
        node_id: UUID,
        observation: Observation,
    ) -> DynamicTaskGraph:
        node = self.get_node(node_id)
        if node.status is not TaskNodeStatus.RUNNING:
            raise GraphTransitionError(
                f"only a running node can resolve; '{node_id}' is {node.status.value}"
            )
        if observation.succeeded:
            updated = self._updated_node(
                node,
                status=TaskNodeStatus.COMPLETED,
                observation=observation,
            )
        else:
            updated = self._updated_node(
                node,
                status=TaskNodeStatus.FAILED,
                observation=observation,
                failure_reason=observation.error,
            )
        return self._replace_node(updated)

    def mark_blocked(
        self,
        node_id: UUID,
        *,
        reason: str,
    ) -> DynamicTaskGraph:
        node = self.get_node(node_id)
        if node.status is not TaskNodeStatus.PENDING:
            raise GraphTransitionError(
                f"only a pending node can be blocked; '{node_id}' is "
                f"{node.status.value}"
            )
        updated = self._updated_node(
            node,
            status=TaskNodeStatus.BLOCKED,
            failure_reason=reason,
        )
        return self._replace_node(updated)

    def retry_failed(self, node_id: UUID) -> DynamicTaskGraph:
        """Reopen a failed node while retaining history in prior snapshots."""

        node = self.get_node(node_id)
        if node.status is not TaskNodeStatus.FAILED:
            raise GraphTransitionError("only a failed node can be retried")
        reopened = self._updated_node(
            node,
            status=TaskNodeStatus.PENDING,
            observation=None,
            failure_reason=None,
        )
        return self._replace_node(reopened)

    def replace_failed_strategy(
        self,
        node_id: UUID,
        strategy_id: str,
    ) -> DynamicTaskGraph:
        """Reopen a failed node with a validated alternate strategy id."""

        if not strategy_id:
            raise GraphTransitionError("replacement strategy_id cannot be empty")
        node = self.get_node(node_id)
        if node.status is not TaskNodeStatus.FAILED:
            raise GraphTransitionError(
                "only a failed node can replace its execution strategy"
            )
        reopened = self._updated_node(
            node,
            strategy_id=strategy_id,
            status=TaskNodeStatus.PENDING,
            observation=None,
            failure_reason=None,
        )
        return self._replace_node(reopened)

    def add_recovery_node(
        self,
        failed_node_id: UUID,
        recovery_node: TaskNode,
    ) -> DynamicTaskGraph:
        """Supersede a failure and route its direct dependents through recovery."""

        failed = self.get_node(failed_node_id)
        if failed.status is not TaskNodeStatus.FAILED:
            raise GraphTransitionError("recovery requires a failed source node")
        if recovery_node.status is not TaskNodeStatus.PENDING:
            raise GraphTransitionError("a recovery node must be pending")
        if any(node.node_id == recovery_node.node_id for node in self.nodes):
            raise GraphTransitionError("recovery node already exists")
        if failed_node_id in recovery_node.dependencies:
            raise GraphTransitionError(
                "recovery node cannot depend on the failed node it replaces"
            )
        known = {node.node_id for node in self.nodes}
        if not set(recovery_node.dependencies).issubset(known):
            raise GraphTransitionError("recovery node has unknown dependencies")

        superseded = self._updated_node(
            failed,
            status=TaskNodeStatus.RECOVERED,
        )
        updated_nodes: list[TaskNode] = []
        for node in self.nodes:
            if node.node_id == failed_node_id:
                updated_nodes.append(superseded)
                continue
            if failed_node_id not in node.dependencies:
                updated_nodes.append(node)
                continue
            if node.status not in {
                TaskNodeStatus.PENDING,
                TaskNodeStatus.BLOCKED,
            }:
                raise GraphTransitionError(
                    "only unresolved dependents can be routed through recovery"
                )
            dependencies = tuple(
                recovery_node.node_id if item == failed_node_id else item
                for item in node.dependencies
            )
            updated_nodes.append(
                self._updated_node(
                    node,
                    dependencies=dependencies,
                    status=TaskNodeStatus.PENDING,
                    failure_reason=None,
                )
            )
        updated_nodes.append(recovery_node)
        return self._with_nodes(tuple(updated_nodes))

    def rewire_dependency(
        self,
        node_id: UUID,
        old_dependency_id: UUID,
        new_dependency_id: UUID,
    ) -> DynamicTaskGraph:
        node = self.get_node(node_id)
        self.get_node(new_dependency_id)
        if node.status not in {
            TaskNodeStatus.PENDING,
            TaskNodeStatus.BLOCKED,
        }:
            raise GraphTransitionError(
                "only unresolved nodes can rewire dependencies"
            )
        if old_dependency_id not in node.dependencies:
            raise GraphTransitionError("old dependency is not attached to the node")
        dependencies = tuple(
            new_dependency_id if item == old_dependency_id else item
            for item in node.dependencies
        )
        if len(set(dependencies)) != len(dependencies):
            raise GraphTransitionError("rewire would duplicate a dependency")
        rewired = self._updated_node(
            node,
            dependencies=dependencies,
            status=TaskNodeStatus.PENDING,
            failure_reason=None,
        )
        return self._replace_node(rewired)

    def _add_node(self, node: TaskNode) -> DynamicTaskGraph:
        if any(existing.node_id == node.node_id for existing in self.nodes):
            raise GraphTransitionError(
                f"task node '{node.node_id}' already exists"
            )
        return self._with_nodes((*self.nodes, node))

    def _add_dependency(
        self,
        node_id: UUID,
        dependency_id: UUID,
    ) -> DynamicTaskGraph:
        node = self.get_node(node_id)
        self.get_node(dependency_id)
        if node.status is not TaskNodeStatus.PENDING:
            raise GraphTransitionError(
                "dependencies can only be added to pending task nodes"
            )
        if dependency_id in node.dependencies:
            return self
        updated = self._updated_node(
            node,
            dependencies=(*node.dependencies, dependency_id),
        )
        return self._replace_node(updated)

    def _replace_node(self, updated: TaskNode) -> DynamicTaskGraph:
        nodes = tuple(
            updated if node.node_id == updated.node_id else node
            for node in self.nodes
        )
        return self._with_nodes(nodes)

    def _with_nodes(self, nodes: tuple[TaskNode, ...]) -> DynamicTaskGraph:
        return DynamicTaskGraph(
            graph_id=self.graph_id,
            version=self.version + 1,
            nodes=nodes,
        )

    @staticmethod
    def _updated_node(node: TaskNode, **changes: Any) -> TaskNode:
        values = node.model_dump(mode="python")
        values.update(changes)
        return TaskNode.model_validate(values)

"""Immutable orchestration models with no Tool or Agent binding."""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from adaptive_agent_runtime import Observation


class OrchestrationModel(BaseModel):
    """Base model for immutable orchestration snapshots."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class TaskNodeStatus(StrEnum):
    """Lifecycle states of one goal unit in a task graph."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    RECOVERED = "recovered"


class TaskNode(OrchestrationModel):
    """A goal unit independent of its eventual execution provider."""

    node_id: UUID = Field(default_factory=uuid4)
    goal: str = Field(min_length=1)
    dependencies: tuple[UUID, ...] = ()
    expected_output: str = Field(min_length=1)
    strategy_id: str = Field(min_length=1)
    status: TaskNodeStatus = TaskNodeStatus.PENDING
    observation: Observation | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_node(self) -> TaskNode:
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("task node dependencies must be unique")
        if self.node_id in self.dependencies:
            raise ValueError("a task node cannot depend on itself")

        if self.status is TaskNodeStatus.COMPLETED:
            if self.observation is None or not self.observation.succeeded:
                raise ValueError("a completed node requires a successful observation")
            if self.failure_reason is not None:
                raise ValueError("a completed node cannot contain a failure reason")
        elif self.status in {
            TaskNodeStatus.FAILED,
            TaskNodeStatus.RECOVERED,
        }:
            if self.observation is None or self.observation.succeeded:
                raise ValueError(
                    "a failed or recovered node requires a failed observation"
                )
            if not self.failure_reason:
                raise ValueError(
                    "a failed or recovered node requires a failure reason"
                )
        elif self.status is TaskNodeStatus.BLOCKED:
            if self.observation is not None:
                raise ValueError("a blocked node cannot contain an observation")
            if not self.failure_reason:
                raise ValueError("a blocked node requires a failure reason")
        elif self.observation is not None or self.failure_reason is not None:
            raise ValueError(
                "pending and running nodes cannot contain terminal result data"
            )
        return self


class GraphMutationType(StrEnum):
    """Graph changes supported by the Phase 2 planner."""

    ADD_NODE = "add_node"
    ADD_DEPENDENCY = "add_dependency"


class GraphMutation(OrchestrationModel):
    """A validated proposal to change a DynamicTaskGraph."""

    mutation_id: UUID = Field(default_factory=uuid4)
    mutation_type: GraphMutationType
    node: TaskNode | None = None
    node_id: UUID | None = None
    dependency_id: UUID | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_mutation(self) -> GraphMutation:
        if self.mutation_type is GraphMutationType.ADD_NODE:
            if self.node is None:
                raise ValueError("add_node requires a task node")
            if self.node_id is not None or self.dependency_id is not None:
                raise ValueError("add_node cannot contain dependency fields")
            if self.node.status is not TaskNodeStatus.PENDING:
                raise ValueError("a dynamically added node must be pending")
        elif self.mutation_type is GraphMutationType.ADD_DEPENDENCY:
            if self.node is not None:
                raise ValueError("add_dependency cannot contain a task node")
            if self.node_id is None or self.dependency_id is None:
                raise ValueError(
                    "add_dependency requires node_id and dependency_id"
                )
        return self

    @classmethod
    def add_node(
        cls,
        node: TaskNode,
        *,
        reason: str | None = None,
    ) -> GraphMutation:
        return cls(
            mutation_type=GraphMutationType.ADD_NODE,
            node=node,
            reason=reason,
        )

    @classmethod
    def add_dependency(
        cls,
        node_id: UUID,
        dependency_id: UUID,
        *,
        reason: str | None = None,
    ) -> GraphMutation:
        return cls(
            mutation_type=GraphMutationType.ADD_DEPENDENCY,
            node_id=node_id,
            dependency_id=dependency_id,
            reason=reason,
        )


class NodeExecutionResult(OrchestrationModel):
    """Provider-neutral result returned by an ExecutionStrategy."""

    succeeded: bool
    output: JsonValue = None
    error: str | None = None
    mutations: tuple[GraphMutation, ...] = ()

    @model_validator(mode="after")
    def validate_result(self) -> NodeExecutionResult:
        if self.succeeded and self.error is not None:
            raise ValueError("a successful node result cannot contain an error")
        if not self.succeeded and not self.error:
            raise ValueError("a failed node result requires an error")
        if not self.succeeded and self.mutations:
            raise ValueError("a failed node result cannot mutate the graph")
        return self

    @classmethod
    def ok(
        cls,
        *,
        output: JsonValue = None,
        mutations: tuple[GraphMutation, ...] = (),
    ) -> NodeExecutionResult:
        return cls(succeeded=True, output=output, mutations=mutations)

    @classmethod
    def failed(cls, *, error: str) -> NodeExecutionResult:
        return cls(succeeded=False, error=error)

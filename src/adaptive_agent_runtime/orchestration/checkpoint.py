"""Persistable orchestration cursor kept outside TaskNode domain models."""

from __future__ import annotations

from uuid import UUID

from pydantic import Field, model_validator

from adaptive_agent_runtime.orchestration.graph import DynamicTaskGraph
from adaptive_agent_runtime.orchestration.models import (
    OrchestrationModel,
    TaskNodeStatus,
)
from adaptive_agent_runtime.orchestration.recovery import RecoveryRecord


class InFlightTaskAction(OrchestrationModel):
    """Association between a Core action and a running graph node."""

    action_id: UUID
    node_id: UUID


class TaskGraphCheckpoint(OrchestrationModel):
    """Complete orchestration state needed to continue one run."""

    run_id: UUID
    graph: DynamicTaskGraph
    in_flight: tuple[InFlightTaskAction, ...] = ()
    processed_action_ids: tuple[UUID, ...] = ()
    recovery_attempts: tuple[tuple[UUID, int], ...] = ()
    recovery_records: tuple[RecoveryRecord, ...] = ()
    state_revision: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_cursor(self) -> TaskGraphCheckpoint:
        action_ids = tuple(item.action_id for item in self.in_flight)
        node_ids = tuple(item.node_id for item in self.in_flight)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("in-flight action ids must be unique")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("a task node cannot have multiple in-flight actions")
        if len(set(self.processed_action_ids)) != len(self.processed_action_ids):
            raise ValueError("processed action ids must be unique")
        if set(action_ids).intersection(self.processed_action_ids):
            raise ValueError("an action cannot be in-flight and processed")
        recovery_node_ids = tuple(item[0] for item in self.recovery_attempts)
        if len(set(recovery_node_ids)) != len(recovery_node_ids):
            raise ValueError("recovery attempt node ids must be unique")
        if any(attempt < 1 for _, attempt in self.recovery_attempts):
            raise ValueError("persisted recovery attempts must be positive")
        for item in self.in_flight:
            if self.graph.get_node(item.node_id).status is not TaskNodeStatus.RUNNING:
                raise ValueError("in-flight actions must reference running nodes")
        running_ids = {
            node.node_id
            for node in self.graph.nodes
            if node.status is TaskNodeStatus.RUNNING
        }
        if running_ids != set(node_ids):
            raise ValueError("every running task node requires an in-flight action")
        return self

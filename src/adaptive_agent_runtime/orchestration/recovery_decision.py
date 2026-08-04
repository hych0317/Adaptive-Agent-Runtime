"""Runtime-owned contracts for governed, Agent-proposed failure recovery."""

from __future__ import annotations

from typing import Annotated, Literal, Protocol, runtime_checkable
from uuid import UUID

from pydantic import Field, model_validator

from adaptive_agent_runtime.core.contracts import RuntimeModule
from adaptive_agent_runtime.orchestration.graph import DynamicTaskGraph
from adaptive_agent_runtime.orchestration.models import OrchestrationModel, TaskNode
from adaptive_agent_runtime.orchestration.recovery import (
    RecoveryActionType,
    RecoveryContext,
    RecoveryPlan,
    apply_recovery_plan,
)


RECOVERY_DECISION_TYPE = "recovery.failure"
RECOVERY_APPLY_OPERATION = "recovery.apply"


class RecoveryExecutionPolicy(OrchestrationModel):
    """Runtime-owned bound around one Recovery Agent decision."""

    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_agent_calls: int = Field(default=1, ge=1, le=1)
    max_recovery_attempts: int = Field(default=2, ge=1)


class RecoveryNodeBinding(OrchestrationModel):
    node_ref: str = Field(min_length=1)
    node_id: UUID


class RecoveryDecisionPayload(OrchestrationModel):
    """Full Runtime snapshot retained outside the isolated Agent context."""

    task_description: str = Field(min_length=1)
    graph: DynamicTaskGraph
    failed_node_id: UUID
    failed_action_id: UUID
    node_bindings: tuple[RecoveryNodeBinding, ...] = Field(min_length=1)
    available_strategy_ids: tuple[str, ...] = Field(min_length=1)
    allowed_recovery_actions: tuple[RecoveryActionType, ...] = Field(min_length=1)
    prior_attempts: int = Field(ge=0)
    execution_policy: RecoveryExecutionPolicy

    @model_validator(mode="after")
    def validate_snapshot(self) -> RecoveryDecisionPayload:
        refs = tuple(item.node_ref for item in self.node_bindings)
        ids = tuple(item.node_id for item in self.node_bindings)
        if len(set(refs)) != len(refs) or len(set(ids)) != len(ids):
            raise ValueError("recovery node bindings must be one-to-one")
        if set(ids) != {node.node_id for node in self.graph.nodes}:
            raise ValueError("recovery node bindings must cover the active graph")
        failed = self.graph.get_node(self.failed_node_id)
        if failed.observation is None or failed.observation.succeeded:
            raise ValueError("recovery decision requires a failed node observation")
        if failed.observation.action_id != self.failed_action_id:
            raise ValueError("failed action does not match the graph observation")
        if len(set(self.available_strategy_ids)) != len(
            self.available_strategy_ids
        ):
            raise ValueError("available recovery strategies must be unique")
        if len(set(self.allowed_recovery_actions)) != len(
            self.allowed_recovery_actions
        ):
            raise ValueError("allowed recovery actions must be unique")
        if self.prior_attempts >= self.execution_policy.max_recovery_attempts:
            if self.allowed_recovery_actions != (RecoveryActionType.ABORT,):
                raise ValueError(
                    "exhausted recovery budget may expose only the abort action"
                )
        return self

    def node_id_for(self, node_ref: str) -> UUID:
        for binding in self.node_bindings:
            if binding.node_ref == node_ref:
                return binding.node_id
        raise ValueError(f"unknown recovery node reference: {node_ref}")

    def node_ref_for(self, node_id: UUID) -> str:
        for binding in self.node_bindings:
            if binding.node_id == node_id:
                return binding.node_ref
        raise ValueError(f"unbound recovery node id: {node_id}")


class RetryNodeRecoveryEffect(OrchestrationModel):
    effect_type: Literal["retry_node"] = "retry_node"
    node_id: UUID


class ReplaceStrategyRecoveryEffect(OrchestrationModel):
    effect_type: Literal["replace_strategy"] = "replace_strategy"
    node_id: UUID
    strategy_id: str = Field(min_length=1)


class AddRecoveryNodeEffect(OrchestrationModel):
    effect_type: Literal["add_recovery_node"] = "add_recovery_node"
    failed_node_id: UUID
    recovery_node: TaskNode


class RewireDependencyRecoveryEffect(OrchestrationModel):
    effect_type: Literal["rewire_dependency"] = "rewire_dependency"
    dependent_node_id: UUID
    old_dependency_id: UUID
    new_dependency_id: UUID


class AbortRecoveryEffect(OrchestrationModel):
    effect_type: Literal["abort"] = "abort"
    failed_node_id: UUID
    reason: str = Field(min_length=1)


RecoveryEffect = Annotated[
    RetryNodeRecoveryEffect
    | ReplaceStrategyRecoveryEffect
    | AddRecoveryNodeEffect
    | RewireDependencyRecoveryEffect
    | AbortRecoveryEffect,
    Field(discriminator="effect_type"),
]


class RecoveryDecisionEffect(OrchestrationModel):
    """Normalized effect envelope; no effect embeds or replaces the whole Graph."""

    graph_id: UUID
    graph_version_before: int = Field(ge=0)
    graph_version_after: int = Field(ge=0)
    source_draft_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    hypothesis: str = Field(min_length=1)
    alternatives: tuple[str, ...] = ()
    agent_confidence: float = Field(ge=0.0, le=1.0)
    plan: RecoveryPlan
    effect: RecoveryEffect

    @model_validator(mode="after")
    def validate_effect(self) -> RecoveryDecisionEffect:
        if len(self.plan.actions) != 1:
            raise ValueError("one recovery decision must normalize to one effect")
        action = self.plan.actions[0]
        expected = {
            "retry_node": RecoveryActionType.RETRY_NODE,
            "replace_strategy": RecoveryActionType.REPLACE_STRATEGY,
            "add_recovery_node": RecoveryActionType.ADD_RECOVERY_NODE,
            "rewire_dependency": RecoveryActionType.REWIRE_DEPENDENCY,
            "abort": RecoveryActionType.ABORT,
        }[self.effect.effect_type]
        if action.action_type is not expected:
            raise ValueError("recovery plan and normalized effect types differ")
        if self.plan.aborts:
            if self.graph_version_after != self.graph_version_before:
                raise ValueError("abort recovery cannot advance Graph version")
        elif self.graph_version_after <= self.graph_version_before:
            raise ValueError("applied recovery must advance Graph version")
        return self


class RecoveryDecisionOutcome(OrchestrationModel):
    """Domain result returned to the Graph Planner after governed Apply."""

    request_id: UUID
    effect: RecoveryDecisionEffect
    graph: DynamicTaskGraph

    @model_validator(mode="after")
    def validate_graph(self) -> RecoveryDecisionOutcome:
        if self.graph.graph_id != self.effect.graph_id:
            raise ValueError("recovery outcome belongs to another Graph")
        if self.graph.version != self.effect.graph_version_after:
            raise ValueError("recovery outcome Graph version does not match its effect")
        return self


@runtime_checkable
class RecoveryDecisionHandler(RuntimeModule, Protocol):
    """Domain integration port; this is not a second Runtime."""

    async def handle(self, context: RecoveryContext) -> RecoveryDecisionOutcome: ...


def apply_recovery_decision_effect(
    graph: DynamicTaskGraph,
    effect: RecoveryDecisionEffect,
) -> DynamicTaskGraph:
    """Apply one previously validated effect to the exact Graph basis."""

    if graph.graph_id != effect.graph_id:
        raise ValueError("Recovery Effect targets another Graph")
    if graph.version != effect.graph_version_before:
        raise ValueError("Recovery Effect is stale for the active Graph version")
    recovered = apply_recovery_plan(graph, effect.plan)
    if recovered.version != effect.graph_version_after:
        raise ValueError("Recovery Effect produced an unexpected Graph version")
    return recovered

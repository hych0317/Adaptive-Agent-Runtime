"""Core ActionExecutor adapter and deterministic mock strategy."""

from __future__ import annotations

from typing import Iterable, Mapping
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime import (
    ActionRequest,
    AgentState,
    Observation,
)
from adaptive_agent_runtime.orchestration.contracts import ExecutionStrategy
from adaptive_agent_runtime.orchestration.models import (
    NodeExecutionResult,
    TaskNode,
)

EXECUTE_NODE_ACTION = "orchestration.execute_node"
ORCHESTRATION_METADATA_KEY = "orchestration"


class MockExecutionStrategy:
    """Return predefined provider-neutral outcomes for TaskNodes."""

    module_id = "orchestration.strategy.mock"
    strategy_id = "mock"

    def __init__(
        self,
        outcomes: Mapping[UUID, NodeExecutionResult] | None = None,
    ) -> None:
        self._outcomes = dict(outcomes or {})
        self.executed_node_ids: list[UUID] = []

    async def execute(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> NodeExecutionResult:
        del state
        self.executed_node_ids.append(node.node_id)
        return self._outcomes.get(
            node.node_id,
            NodeExecutionResult.ok(output={"goal": node.goal}),
        )


class StrategyActionExecutor:
    """Adapt abstract ExecutionStrategy instances to Core ActionExecutor."""

    module_id = "orchestration.strategy_executor"

    def __init__(self, strategies: Iterable[ExecutionStrategy]) -> None:
        self._strategies: dict[str, ExecutionStrategy] = {}
        for strategy in strategies:
            if strategy.strategy_id in self._strategies:
                raise ValueError(
                    f"duplicate execution strategy '{strategy.strategy_id}'"
                )
            self._strategies[strategy.strategy_id] = strategy
        if not self._strategies:
            raise ValueError("at least one execution strategy is required")

    async def execute(
        self,
        action: ActionRequest,
        state: AgentState,
    ) -> Observation:
        if action.name != EXECUTE_NODE_ACTION:
            return Observation.failed(
                action.action_id,
                error=f"unsupported orchestration action '{action.name}'",
            )

        arguments = action.model_dump(mode="python")["arguments"]
        raw_node = arguments.get("node")
        try:
            node = TaskNode.model_validate(raw_node)
        except Exception as exc:
            return Observation.failed(
                action.action_id,
                error=f"invalid task node action payload: {exc}",
            )

        strategy = self._strategies.get(node.strategy_id)
        if strategy is None:
            return self._observation(
                action,
                node,
                NodeExecutionResult.failed(
                    error=f"execution strategy '{node.strategy_id}' is unavailable"
                ),
            )

        try:
            result = await strategy.execute(node, state)
            if not isinstance(result, NodeExecutionResult):
                raise TypeError("execution strategy must return NodeExecutionResult")
        except Exception as exc:
            detail = str(exc) or exc.__class__.__name__
            result = NodeExecutionResult.failed(
                error=(
                    f"execution strategy '{strategy.strategy_id}' failed: "
                    f"{exc.__class__.__name__}: {detail}"
                )
            )
        return self._observation(action, node, result)

    @staticmethod
    def _observation(
        action: ActionRequest,
        node: TaskNode,
        result: NodeExecutionResult,
    ) -> Observation:
        metadata: dict[str, JsonValue] = {
            ORCHESTRATION_METADATA_KEY: {
                "node_id": str(node.node_id),
                "mutations": [
                    mutation.model_dump(mode="json")
                    for mutation in result.mutations
                ],
            }
        }
        if result.succeeded:
            return Observation.ok(
                action.action_id,
                output=result.output,
                metadata=metadata,
            )
        return Observation.failed(
            action.action_id,
            error=result.error or "task node execution failed",
            metadata=metadata,
        )

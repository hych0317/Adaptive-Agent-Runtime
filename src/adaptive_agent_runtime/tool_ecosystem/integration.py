"""Adapters from Task requirements and Tool results to stable Runtime contracts."""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime.core.contracts import RuntimeModule
from adaptive_agent_runtime.core.models import AgentState, Observation
from adaptive_agent_runtime.orchestration.models import (
    NodeExecutionResult,
    TaskNode,
)
from adaptive_agent_runtime.tool_ecosystem.contracts import (
    CapabilityCandidateResolver,
    ToolExecutor,
    ToolSelector,
)
from adaptive_agent_runtime.tool_ecosystem.errors import (
    ToolEcosystemError,
    ToolIntegrationError,
    ToolSelectionError,
)
from adaptive_agent_runtime.tool_ecosystem.governance import ToolExecutionPolicy
from adaptive_agent_runtime.tool_ecosystem.models import (
    CapabilityRequest,
    ToolCorrelation,
    ToolInvocation,
    ToolObservation,
)


TOOL_OBSERVATION_METADATA_KEY = "tool"


@runtime_checkable
class TaskCapabilityRequestProvider(RuntimeModule, Protocol):
    def request_for(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> CapabilityRequest: ...


class MappedTaskCapabilityRequestProvider:
    """Keep Capability requests outside provider-neutral TaskNode snapshots."""

    module_id = "tool.task_request.mapped"

    def __init__(self, requests: Mapping[UUID, CapabilityRequest]) -> None:
        self._requests = dict(requests)

    def request_for(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> CapabilityRequest:
        del state
        request = self._requests.get(node.node_id)
        if request is None:
            raise ToolIntegrationError(
                f"task node '{node.node_id}' has no capability request"
            )
        return request


class ToolResultObservationAdapter:
    module_id = "tool.adapter.observation"

    def convert(
        self,
        result: ToolObservation,
        *,
        action_id: UUID,
    ) -> Observation:
        if (
            result.correlation.action_id is not None
            and result.correlation.action_id != action_id
        ):
            raise ToolIntegrationError(
                "tool result action correlation does not match action_id"
            )
        correlation: dict[str, JsonValue] = {
            "action_id": str(action_id),
        }
        for field_name in ("run_id", "task_id", "node_id"):
            value = getattr(result.correlation, field_name)
            if value is not None:
                correlation[field_name] = str(value)
        metadata: dict[str, JsonValue] = {
            TOOL_OBSERVATION_METADATA_KEY: {
                "invocation_id": str(result.invocation_id),
                "requirement_id": str(result.requirement_id),
                "capability_id": result.capability_id,
                "provider_id": result.provider_id,
                "status": result.status.value,
                "retry_status": result.retry_status.value,
                "attempt_count": len(result.attempts),
                "correlation": correlation,
            }
        }
        if result.succeeded:
            return Observation.ok(
                action_id,
                output=result.model_dump(mode="json")["output"],
                metadata=metadata,
            )
        return Observation.failed(
            action_id,
            error=result.error or "tool execution failed",
            metadata=metadata,
        )


class ToolExecutionStrategy:
    """Adapt Capability selection and Tool execution to Orchestration."""

    module_id = "tool.strategy.execution"

    def __init__(
        self,
        *,
        requests: TaskCapabilityRequestProvider,
        resolver: CapabilityCandidateResolver,
        selector: ToolSelector,
        executor: ToolExecutor,
        policy: ToolExecutionPolicy,
        strategy_id: str = "tool",
    ) -> None:
        self._requests = requests
        self._resolver = resolver
        self._selector = selector
        self._executor = executor
        self._policy = policy
        self.strategy_id = strategy_id

    async def execute(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> NodeExecutionResult:
        try:
            request = self._requests.request_for(node, state)
            candidates = self._resolver.candidates(request.requirement)
            selection = self._selector.select(
                request.requirement,
                candidates,
                request.selection_context,
            )
            if selection.requirement_id != request.requirement.requirement_id:
                raise ToolSelectionError(
                    "selector returned a mismatched requirement_id"
                )
            if selection.capability_id != request.requirement.capability_id:
                raise ToolSelectionError(
                    "selector returned a mismatched capability_id"
                )
            if selection.provider_id not in {
                candidate.provider_id for candidate in candidates
            }:
                raise ToolSelectionError(
                    "selector returned a provider outside Runtime candidates"
                )
            invocation = ToolInvocation(
                requirement_id=request.requirement.requirement_id,
                capability_id=request.requirement.capability_id,
                provider_id=selection.provider_id,
                arguments=request.arguments,
                correlation=ToolCorrelation(
                    run_id=state.run_id,
                    task_id=state.task.task_id,
                    node_id=node.node_id,
                ),
            )
            result = await self._executor.execute(invocation, self._policy)
        except ToolEcosystemError as exc:
            return NodeExecutionResult.failed(error=str(exc))
        if result.succeeded:
            return NodeExecutionResult.ok(
                output=result.model_dump(mode="json")["output"]
            )
        return NodeExecutionResult.failed(
            error=result.error or "tool execution failed"
        )

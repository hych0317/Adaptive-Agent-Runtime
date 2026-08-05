"""Interfaces for provider-neutral execution and selective isolation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from adaptive_agent_runtime import AgentState, RuntimeModule
from adaptive_agent_runtime.orchestration.checkpoint import TaskGraphCheckpoint
from adaptive_agent_runtime.orchestration.graph import DynamicTaskGraph
from adaptive_agent_runtime.orchestration.models import (
    GraphMutation,
    NodeExecutionResult,
    TaskNode,
)
from adaptive_agent_runtime.orchestration.recovery import RecoveryRecord


@runtime_checkable
class ExecutionStrategy(RuntimeModule, Protocol):
    """Execute a TaskNode without exposing a concrete provider to the graph."""

    @property
    def strategy_id(self) -> str:
        """Return the identifier referenced by TaskNode.strategy_id."""

        ...

    async def execute(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> NodeExecutionResult:
        """Execute a node against an immutable Runtime state snapshot."""

        ...


@runtime_checkable
class IsolatedAgentExecutor(RuntimeModule, Protocol):
    """Future interface for selectively isolated Agent execution."""

    async def execute_isolated(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> NodeExecutionResult:
        """Execute one node in an isolated Agent boundary."""

        ...


@runtime_checkable
class ReadyTaskNodeSelector(RuntimeModule, Protocol):
    """Propose one node from the Runtime-computed ready set."""

    async def select_node_id(
        self,
        ready_nodes: tuple[TaskNode, ...],
        state: AgentState,
    ) -> UUID:
        """Return a candidate ID; the planner validates membership."""

        ...


@runtime_checkable
class TaskGraphStore(RuntimeModule, Protocol):
    """Persist the graph plus the planner cursor without storage coupling."""

    async def save(self, checkpoint: TaskGraphCheckpoint) -> None: ...

    async def load(self, run_id: UUID) -> TaskGraphCheckpoint | None: ...


@runtime_checkable
class GraphDecisionCommitter(RuntimeModule, Protocol):
    """Narrow authority used only by governed Graph Decision Apply."""

    async def commit_graph_effect(
        self,
        *,
        state: AgentState,
        graph: DynamicTaskGraph,
        effect_fingerprint: str,
        recovery_record: RecoveryRecord | None = None,
    ) -> DynamicTaskGraph: ...

    async def load_graph_effect(
        self,
        *,
        run_id: UUID,
        effect_fingerprint: str,
    ) -> DynamicTaskGraph | None: ...


@runtime_checkable
class GraphMutationApplier(RuntimeModule, Protocol):
    """Apply one validated mutation through a caller-selected policy seam."""

    async def apply(
        self,
        graph: DynamicTaskGraph,
        mutation: GraphMutation,
        *,
        state: AgentState,
        source_node_id: UUID,
    ) -> DynamicTaskGraph: ...

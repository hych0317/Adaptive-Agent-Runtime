"""Narrow interfaces implemented by Runtime Core collaborators."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from adaptive_agent_runtime.core.models import (
    ActionRequest,
    AgentState,
    Observation,
    PlanDecision,
    RuntimeEvent,
    TraceEntry,
)


@runtime_checkable
class RuntimeModule(Protocol):
    """Minimal identity shared by runtime modules."""

    @property
    def module_id(self) -> str:
        """Return the stable identifier used in runtime traces."""

        ...


@runtime_checkable
class Planner(RuntimeModule, Protocol):
    """Create the next domain-neutral decision from a state snapshot."""

    async def plan(self, state: AgentState) -> PlanDecision:
        """Return one action to execute, or complete the run."""

        ...


@runtime_checkable
class ActionExecutor(RuntimeModule, Protocol):
    """Execute a generic action and return its observation."""

    async def execute(
        self,
        action: ActionRequest,
        state: AgentState,
    ) -> Observation:
        """Execute one action against an immutable state snapshot."""

        ...


@runtime_checkable
class StateStore(RuntimeModule, Protocol):
    """Persist and retrieve AgentState snapshots without storage semantics."""

    async def save(self, state: AgentState) -> None:
        """Store the supplied immutable snapshot."""

        ...

    async def load(self, run_id: UUID) -> AgentState | None:
        """Load the latest snapshot for a run, if one exists."""

        ...


@runtime_checkable
class TraceSink(RuntimeModule, Protocol):
    """Append runtime facts to a trace."""

    async def record(self, event: RuntimeEvent) -> TraceEntry:
        """Persist an event and assign its per-run sequence number."""

        ...


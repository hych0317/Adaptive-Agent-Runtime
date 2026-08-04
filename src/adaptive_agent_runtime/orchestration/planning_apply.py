"""Authorized graph initialization commit and fail-closed load boundary."""

from __future__ import annotations

from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime.orchestration.checkpoint import TaskGraphCheckpoint
from adaptive_agent_runtime.orchestration.contracts import TaskGraphStore
from adaptive_agent_runtime.orchestration.errors import OrchestrationStateError
from adaptive_agent_runtime.orchestration.planning import PlanningGraphEffect


class GraphInitializationApplier:
    """Commit only a Runtime-generated graph v0 after Governance approval."""

    module_id = "orchestration.planning.graph_initialization_applier"

    def __init__(self, store: TaskGraphStore) -> None:
        self._store = store

    async def apply(self, effect: PlanningGraphEffect) -> JsonValue:
        if effect.graph.version != 0:
            raise OrchestrationStateError("initial planning effect must be graph v0")
        existing = await self._store.load(effect.run_id)
        checkpoint = TaskGraphCheckpoint(
            run_id=effect.run_id,
            graph=effect.graph,
            state_revision=0,
        )
        if existing is not None and existing != checkpoint:
            raise OrchestrationStateError(
                "run already has a different task graph checkpoint"
            )
        await self._store.save(checkpoint)
        persisted = await self._store.load(effect.run_id)
        if persisted != checkpoint:
            raise OrchestrationStateError(
                "task graph checkpoint did not preserve the approved effect"
            )
        return {
            "run_id": str(effect.run_id),
            "task_id": str(effect.task_id),
            "graph_id": str(effect.graph.graph_id),
            "graph_version": effect.graph.version,
            "node_count": len(effect.graph.nodes),
            "source_draft_fingerprint": effect.source_draft_fingerprint,
        }


class RequiredPreparedTaskGraphStore:
    """Prevent Planner fallback to an unapproved graph when bootstrap is missing."""

    module_id = "orchestration.graph_store.required_prepared"

    def __init__(
        self,
        delegate: TaskGraphStore,
        *,
        run_id: UUID,
        graph_id: UUID,
    ) -> None:
        self._delegate = delegate
        self._run_id = run_id
        self._graph_id = graph_id

    async def save(self, checkpoint: TaskGraphCheckpoint) -> None:
        self._validate(checkpoint)
        await self._delegate.save(checkpoint)

    async def load(self, run_id: UUID) -> TaskGraphCheckpoint | None:
        if run_id != self._run_id:
            raise OrchestrationStateError(
                "prepared graph store cannot serve another Runtime run"
            )
        checkpoint = await self._delegate.load(run_id)
        if checkpoint is None:
            raise OrchestrationStateError(
                "approved initial task graph checkpoint is missing"
            )
        self._validate(checkpoint)
        return checkpoint

    def _validate(self, checkpoint: TaskGraphCheckpoint) -> None:
        if checkpoint.run_id != self._run_id:
            raise OrchestrationStateError("task graph checkpoint run mismatch")
        if checkpoint.graph.graph_id != self._graph_id:
            raise OrchestrationStateError(
                "task graph checkpoint does not belong to the approved graph"
            )

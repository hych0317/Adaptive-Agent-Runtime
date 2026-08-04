"""Small in-memory Task Graph checkpoint adapter."""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from adaptive_agent_runtime.orchestration.checkpoint import TaskGraphCheckpoint
from adaptive_agent_runtime.orchestration.errors import OrchestrationStateError


class InMemoryTaskGraphStore:
    module_id = "orchestration.graph_store.in_memory"

    def __init__(self) -> None:
        self._current: dict[UUID, TaskGraphCheckpoint] = {}
        self._history: defaultdict[UUID, list[TaskGraphCheckpoint]] = defaultdict(list)

    async def save(self, checkpoint: TaskGraphCheckpoint) -> None:
        current = self._current.get(checkpoint.run_id)
        if current is not None:
            if checkpoint.graph.version < current.graph.version:
                raise OrchestrationStateError(
                    "task graph checkpoint would move backwards"
                )
            if checkpoint.graph.version == current.graph.version:
                if checkpoint == current:
                    return
                raise OrchestrationStateError(
                    "task graph version was reused with different content"
                )
        self._current[checkpoint.run_id] = checkpoint
        self._history[checkpoint.run_id].append(checkpoint)

    async def load(self, run_id: UUID) -> TaskGraphCheckpoint | None:
        return self._current.get(run_id)

    def history_for(self, run_id: UUID) -> tuple[TaskGraphCheckpoint, ...]:
        return tuple(self._history.get(run_id, ()))

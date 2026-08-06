"""Small in-memory Task Graph checkpoint adapter."""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from adaptive_agent_runtime.orchestration.checkpoint import TaskGraphCheckpoint
from adaptive_agent_runtime.orchestration.errors import OrchestrationStateError
from adaptive_agent_runtime import AgentState
from adaptive_agent_runtime.orchestration.graph import DynamicTaskGraph
from adaptive_agent_runtime.orchestration.recovery import RecoveryRecord
from adaptive_agent_runtime.governance.models import GovernanceTarget, RuntimeCommitPermit


class InMemoryTaskGraphStore:
    module_id = "orchestration.graph_store.in_memory"

    def __init__(self) -> None:
        self._current: dict[UUID, TaskGraphCheckpoint] = {}
        self._history: defaultdict[UUID, list[TaskGraphCheckpoint]] = defaultdict(list)

    async def save(
        self,
        checkpoint: TaskGraphCheckpoint,
        *,
        permit: RuntimeCommitPermit | None = None,
        target: GovernanceTarget | None = None,
        subject_fingerprint: str | None = None,
    ) -> None:
        del permit, target, subject_fingerprint
        current = self._current.get(checkpoint.run_id)
        if current is not None:
            if checkpoint.checkpoint_revision < current.checkpoint_revision:
                raise OrchestrationStateError(
                    "task graph checkpoint would move backwards"
                )
            if checkpoint.checkpoint_revision == current.checkpoint_revision:
                if checkpoint == current:
                    return
                raise OrchestrationStateError(
                    "task graph checkpoint revision was reused with different content"
                )
        self._current[checkpoint.run_id] = checkpoint
        self._history[checkpoint.run_id].append(checkpoint)

    async def load(self, run_id: UUID) -> TaskGraphCheckpoint | None:
        return self._current.get(run_id)

    def history_for(self, run_id: UUID) -> tuple[TaskGraphCheckpoint, ...]:
        return tuple(self._history.get(run_id, ()))


class InMemoryGraphDecisionCommitter:
    """Standalone authoritative commit seam for domain integration tests."""

    module_id = "orchestration.graph_decision_committer.in_memory"

    def __init__(self) -> None:
        self._commits: dict[tuple[UUID, str], DynamicTaskGraph] = {}

    async def commit_graph_effect(
        self,
        *,
        state: AgentState,
        graph: DynamicTaskGraph,
        effect_fingerprint: str,
        recovery_record: RecoveryRecord | None = None,
        permit: RuntimeCommitPermit | None = None,
        target: GovernanceTarget | None = None,
        subject_fingerprint: str | None = None,
    ) -> DynamicTaskGraph:
        del recovery_record, permit, target, subject_fingerprint
        key = (state.run_id, effect_fingerprint)
        prior = self._commits.get(key)
        if prior is not None and prior != graph:
            raise OrchestrationStateError("effect fingerprint was reused for another Graph")
        self._commits[key] = graph
        return self._commits[key]

    async def load_graph_effect(
        self,
        *,
        run_id: UUID,
        effect_fingerprint: str,
    ) -> DynamicTaskGraph | None:
        return self._commits.get((run_id, effect_fingerprint))

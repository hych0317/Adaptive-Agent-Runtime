"""Pure state transitions and the Phase 1 in-memory state store."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from adaptive_agent_runtime.core.errors import RuntimeInvariantError
from adaptive_agent_runtime.core.models import (
    AgentState,
    AgentTask,
    Observation,
    PlanDecision,
    PlanDecisionType,
    RunStatus,
    utc_now,
)


def create_state(task: AgentTask, *, run_id: UUID | None = None) -> AgentState:
    now = utc_now()
    return AgentState(
        run_id=run_id or uuid4(),
        task=task,
        created_at=now,
        updated_at=now,
    )


def _evolve(
    state: AgentState,
    *,
    at: datetime | None = None,
    **changes: Any,
) -> AgentState:
    values = state.model_dump(mode="python")
    values.update(changes)
    values["revision"] = state.revision + 1
    values["updated_at"] = at or utc_now()
    return AgentState.model_validate(values)


def start_state(state: AgentState) -> AgentState:
    if state.status is not RunStatus.PENDING:
        raise RuntimeInvariantError("only a pending state can be started")
    return _evolve(state, status=RunStatus.RUNNING)


def record_observation(
    state: AgentState,
    plan: PlanDecision,
    observation: Observation,
) -> AgentState:
    if state.status is not RunStatus.RUNNING:
        raise RuntimeInvariantError("observations require a running state")
    if plan.decision is not PlanDecisionType.EXECUTE or plan.action is None:
        raise RuntimeInvariantError("an observation requires an execute decision")
    if observation.action_id != plan.action.action_id:
        raise RuntimeInvariantError("observation action_id does not match the plan")
    return _evolve(
        state,
        step_count=state.step_count + 1,
        last_plan=plan,
        last_observation=observation,
    )


def complete_state(state: AgentState, plan: PlanDecision) -> AgentState:
    if state.status is not RunStatus.RUNNING:
        raise RuntimeInvariantError("only a running state can complete")
    if plan.decision is not PlanDecisionType.COMPLETE:
        raise RuntimeInvariantError("completion requires a complete decision")
    return _evolve(
        state,
        status=RunStatus.COMPLETED,
        last_plan=plan,
        # Re-enter validation through the serialized form rather than passing
        # the PlanDecision's internal immutable mapping proxy across models.
        output=plan.model_dump(mode="python")["output"],
    )


def fail_state(
    state: AgentState,
    error: str,
    *,
    last_plan: PlanDecision | None = None,
) -> AgentState:
    if state.status is not RunStatus.RUNNING:
        raise RuntimeInvariantError("only a running state can fail")
    changes: dict[str, Any] = {"status": RunStatus.FAILED, "error": error}
    if last_plan is not None:
        changes["last_plan"] = last_plan
    return _evolve(state, **changes)


class InMemoryStateStore:
    """Snapshot store used by the minimal runtime and tests."""

    module_id = "state.in_memory"

    def __init__(self) -> None:
        self._current: dict[UUID, AgentState] = {}
        self._history: defaultdict[UUID, list[AgentState]] = defaultdict(list)

    async def save(self, state: AgentState) -> None:
        self._current[state.run_id] = state
        self._history[state.run_id].append(state)

    async def load(self, run_id: UUID) -> AgentState | None:
        return self._current.get(run_id)

    def history_for(self, run_id: UUID) -> tuple[AgentState, ...]:
        """Return snapshots for inspection; this is not part of StateStore."""

        return tuple(self._history.get(run_id, ()))

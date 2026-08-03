"""The minimal sequential event loop for Runtime Core."""

from __future__ import annotations

from typing import Any, Mapping
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime.core.contracts import (
    ActionExecutor,
    Planner,
    StateStore,
    TraceSink,
)
from adaptive_agent_runtime.core.errors import (
    RuntimeInfrastructureError,
    RuntimeInvariantError,
    RuntimeResumeBlockedError,
    RuntimeResumeError,
)
from adaptive_agent_runtime.core.models import (
    AgentState,
    AgentTask,
    CoreEventKind,
    Observation,
    PlanDecision,
    PlanDecisionType,
    RunResult,
    RunStatus,
    RuntimeEvent,
)
from adaptive_agent_runtime.core.state import (
    complete_state,
    create_state,
    fail_state,
    record_observation,
    start_state,
)


class AgentRuntime:
    """Drive one Planner and ActionExecutor through a sequential feedback loop."""

    module_id = "runtime.core"

    def __init__(
        self,
        *,
        planner: Planner,
        executor: ActionExecutor,
        state_store: StateStore,
        trace_sink: TraceSink,
        max_steps: int = 16,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._planner = planner
        self._executor = executor
        self._state_store = state_store
        self._trace_sink = trace_sink
        self._max_steps = max_steps

    async def run(
        self,
        task: AgentTask,
        *,
        run_id: UUID | None = None,
    ) -> RunResult:
        state = create_state(task, run_id=run_id)
        if await self._load_state(state.run_id) is not None:
            raise RuntimeInvariantError(
                f"run_id '{state.run_id}' already exists; "
                "use resume(run_id) to continue it"
            )
        await self._save_state(state)

        state = start_state(state)
        await self._save_state(state)
        await self._emit(
            state,
            CoreEventKind.RUNTIME_STARTED,
            source=self.module_id,
            payload={"state": state.model_dump(mode="json")},
        )

        return await self._drive(state)

    async def resume(self, run_id: UUID) -> RunResult:
        """Continue a persisted run without replaying completed actions.

        Terminal runs are returned unchanged. A pending run is started for the
        first time; a running run continues from its latest immutable snapshot.
        Planners may raise RuntimeResumeBlockedError when an external action is
        in doubt. That signal deliberately leaves the persisted run untouched.
        """

        state = await self._load_state(run_id)
        if state is None:
            raise RuntimeResumeError(f"run '{run_id}' does not exist")
        if state.status in {RunStatus.COMPLETED, RunStatus.FAILED}:
            return RunResult(final_state=state)
        if state.status is RunStatus.PENDING:
            state = start_state(state)
            await self._save_state(state)

        await self._emit(
            state,
            CoreEventKind.RUNTIME_RESUMED,
            source=self.module_id,
            payload={"state": state.model_dump(mode="json")},
        )
        return await self._drive(state)

    async def _drive(self, state: AgentState) -> RunResult:
        """Run the unchanged Plan/Execute/Observe loop from one snapshot."""

        while True:
            try:
                decision = await self._planner.plan(state)
                if not isinstance(decision, PlanDecision):
                    raise TypeError("planner must return PlanDecision")
            except RuntimeResumeBlockedError:
                raise
            except Exception as exc:
                return await self._fail_run(
                    state,
                    self._module_error("planner", self._planner.module_id, exc),
                    source=self._planner.module_id,
                )

            await self._emit(
                state,
                CoreEventKind.PLAN_CREATED,
                source=self._planner.module_id,
                payload={"plan": decision.model_dump(mode="json")},
            )

            if decision.decision is PlanDecisionType.COMPLETE:
                previous = state
                state = complete_state(state, decision)
                await self._save_state(state)
                await self._emit_state_update(previous, state)
                await self._emit(
                    state,
                    CoreEventKind.RUNTIME_COMPLETED,
                    source=self.module_id,
                    payload={"state": state.model_dump(mode="json")},
                )
                return RunResult(final_state=state)

            if decision.decision is PlanDecisionType.FAIL:
                if decision.error is None:
                    return await self._fail_run(
                        state,
                        "planner returned a fail decision without an error",
                        source=self._planner.module_id,
                    )
                return await self._fail_run(
                    state,
                    decision.error,
                    source=self._planner.module_id,
                    last_plan=decision,
                )

            action = decision.action
            if action is None:
                return await self._fail_run(
                    state,
                    "planner returned an execute decision without an action",
                    source=self._planner.module_id,
                )

            if state.step_count >= self._max_steps:
                return await self._fail_run(
                    state,
                    f"maximum action steps reached ({self._max_steps})",
                    source=self.module_id,
                )

            await self._emit(
                state,
                CoreEventKind.ACTION_STARTED,
                source=self._executor.module_id,
                payload={"action": action.model_dump(mode="json")},
            )

            execution_error: str | None = None
            try:
                observation = await self._executor.execute(action, state)
                if not isinstance(observation, Observation):
                    raise TypeError("executor must return Observation")
                if observation.action_id != action.action_id:
                    raise RuntimeInvariantError(
                        "executor returned an observation for a different action"
                    )
            except Exception as exc:
                execution_error = self._module_error(
                    "executor",
                    self._executor.module_id,
                    exc,
                )
                observation = Observation.failed(
                    action.action_id,
                    error=execution_error,
                )

            await self._emit(
                state,
                CoreEventKind.OBSERVATION_RECEIVED,
                source=self._executor.module_id,
                payload={"observation": observation.model_dump(mode="json")},
            )

            previous = state
            state = record_observation(state, decision, observation)
            await self._save_state(state)
            await self._emit_state_update(previous, state)

            if execution_error is not None:
                return await self._fail_run(
                    state,
                    execution_error,
                    source=self._executor.module_id,
                )

    async def _fail_run(
        self,
        state: AgentState,
        error: str,
        *,
        source: str,
        last_plan: PlanDecision | None = None,
    ) -> RunResult:
        previous = state
        state = fail_state(state, error, last_plan=last_plan)
        await self._save_state(state)
        await self._emit_state_update(previous, state)
        await self._emit(
            state,
            CoreEventKind.RUNTIME_FAILED,
            source=source,
            payload={"error": error, "state": state.model_dump(mode="json")},
        )
        return RunResult(final_state=state)

    async def _save_state(self, state: AgentState) -> None:
        try:
            await self._state_store.save(state)
        except Exception as exc:
            raise RuntimeInfrastructureError(
                f"state store '{self._state_store.module_id}' failed"
            ) from exc

    async def _load_state(self, run_id: UUID) -> AgentState | None:
        try:
            return await self._state_store.load(run_id)
        except Exception as exc:
            raise RuntimeInfrastructureError(
                f"state store '{self._state_store.module_id}' failed"
            ) from exc

    async def _emit_state_update(
        self,
        previous: AgentState,
        current: AgentState,
    ) -> None:
        await self._emit(
            current,
            CoreEventKind.STATE_UPDATED,
            source=self.module_id,
            payload={
                "previous_revision": previous.revision,
                "state": current.model_dump(mode="json"),
            },
        )

    async def _emit(
        self,
        state: AgentState,
        kind: str,
        *,
        source: str,
        payload: Mapping[str, JsonValue],
    ) -> None:
        event = RuntimeEvent(
            run_id=state.run_id,
            kind=kind,
            source=source,
            payload=payload,
        )
        try:
            await self._trace_sink.record(event)
        except Exception as exc:
            raise RuntimeInfrastructureError(
                f"trace sink '{self._trace_sink.module_id}' failed"
            ) from exc

    @staticmethod
    def _module_error(stage: str, module_id: str, exc: Exception) -> str:
        detail = str(exc) or exc.__class__.__name__
        exception_name = exc.__class__.__name__
        return f"{stage} module '{module_id}' failed: {exception_name}: {detail}"

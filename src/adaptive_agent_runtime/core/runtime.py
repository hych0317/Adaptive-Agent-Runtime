"""The minimal sequential event loop for Runtime Core."""

from __future__ import annotations

from time import monotonic
from typing import Any, Mapping
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime.core.contracts import (
    ActionExecutor,
    ObservationReconciler,
    Planner,
    StateStore,
    TraceSink,
)
from adaptive_agent_runtime.core.errors import (
    RunBudgetExhaustedError,
    RunUsageAccountingError,
    RuntimeInfrastructureError,
    RuntimeInvariantError,
    RuntimeResumeBlockedError,
    RuntimeResumeError,
)
from adaptive_agent_runtime.core.budget import (
    RunBudgetLedger,
    bind_run_budget_ledger,
)
from adaptive_agent_runtime.core.invocation import (
    RunInvocationGuard,
    bind_run_invocation_guard,
)
from adaptive_agent_runtime.core.models import (
    AgentState,
    AgentTask,
    CoreEventKind,
    Observation,
    PlanDecision,
    PlanDecisionType,
    RunResult,
    RunControlState,
    RunStatus,
    RunStopPolicy,
    RunTerminationReason,
    RuntimeEvent,
    TerminationCheckPhase,
)
from adaptive_agent_runtime.core.state import (
    complete_state,
    create_state,
    fail_state,
    record_observation,
    start_state,
    terminate_state,
    update_control_state,
)
from adaptive_agent_runtime.core.termination import (
    RunTerminationController,
    StopAssessment,
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
        max_steps: int | None = None,
        stop_policy: RunStopPolicy | None = None,
    ) -> None:
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if stop_policy is None:
            stop_policy = RunStopPolicy(max_action_steps=max_steps or 16)
        elif max_steps is not None and max_steps != stop_policy.max_action_steps:
            raise ValueError("max_steps conflicts with RunStopPolicy")
        self._planner = planner
        self._executor = executor
        self._state_store = state_store
        self._trace_sink = trace_sink
        self._stop_policy = stop_policy
        self._termination = RunTerminationController(stop_policy)

    async def run(
        self,
        task: AgentTask,
        *,
        run_id: UUID | None = None,
    ) -> RunResult:
        state = create_state(
            task,
            run_id=run_id,
            stop_policy=self._stop_policy,
        )
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
        in doubt. That signal does not advance or replay an Action; elapsed time
        and authoritative resource usage are still persisted.
        """

        state = await self._load_state(run_id)
        if state is None:
            raise RuntimeResumeError(f"run '{run_id}' does not exist")
        if state.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.TERMINATED,
        }:
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

        try:
            control = self._termination.reconcile_control(state)
        except ValueError as exc:
            raise RuntimeResumeError(
                f"run '{state.run_id}' cannot use this stop policy: {exc}"
            ) from exc
        ledger = RunBudgetLedger(
            run_id=state.run_id,
            policy=self._stop_policy,
            initial_usage=control.usage,
        )
        invocation_guard = RunInvocationGuard(
            repeated_invocation_limit=self._stop_policy.repeated_invocation_limit,
            initial_fingerprint=control.last_tool_invocation_fingerprint,
            initial_count=control.consecutive_identical_tool_invocations,
        )
        with bind_run_budget_ledger(ledger):
            with bind_run_invocation_guard(invocation_guard):
                return await self._drive_bound(
                    state,
                    control,
                    ledger,
                    invocation_guard,
                )

    async def _drive_bound(
        self,
        state: AgentState,
        control: RunControlState,
        ledger: RunBudgetLedger,
        invocation_guard: RunInvocationGuard,
    ) -> RunResult:
        """Drive one Run while its aggregate budget ledger is context-bound."""

        reconciliation_error = await self._reconcile_observation(state)
        if reconciliation_error is not None:
            return await self._fail_run(
                state,
                reconciliation_error,
                source=self._planner.module_id,
                control=control,
            )
        while True:
            usage = await ledger.snapshot()
            control = self._termination.with_usage(control, usage)
            assessment = self._termination.before_planning(
                control,
                phase=TerminationCheckPhase.BEFORE_PLANNING,
            )
            if assessment is not None:
                return await self._stop_run(state, control, assessment)

            planning_started = monotonic()
            try:
                decision = await self._planner.plan(state)
                if not isinstance(decision, PlanDecision):
                    raise TypeError("planner must return PlanDecision")
            except RuntimeResumeBlockedError:
                usage = await ledger.snapshot()
                control = self._termination.after_planning(
                    control,
                    elapsed_seconds=monotonic() - planning_started,
                    usage=usage,
                )
                guard_snapshot = await invocation_guard.snapshot()
                control = control.model_copy(
                    update={
                        "last_tool_invocation_fingerprint": (
                            guard_snapshot.fingerprint
                        ),
                        "consecutive_identical_tool_invocations": (
                            guard_snapshot.consecutive_count
                        ),
                    }
                )
                assessment = self._termination.before_planning(
                    control,
                    phase=TerminationCheckPhase.AFTER_PLANNING,
                )
                if assessment is None and guard_snapshot.blocked:
                    assessment = self._termination.explicit_stop(
                        RunTerminationReason.REPEATED_INVOCATION,
                        phase=TerminationCheckPhase.AFTER_PLANNING,
                        control=control,
                        message=(
                            "consecutive identical Tool invocation limit reached"
                        ),
                    )
                if assessment is not None:
                    return await self._stop_run(state, control, assessment)
                previous = state
                state = update_control_state(state, control)
                await self._save_state(state)
                await self._emit_state_update(previous, state)
                raise
            except Exception as exc:
                usage = await ledger.snapshot()
                control = self._termination.after_planning(
                    control,
                    elapsed_seconds=monotonic() - planning_started,
                    usage=usage,
                )
                assessment = self._termination.before_planning(
                    control,
                    phase=TerminationCheckPhase.AFTER_PLANNING,
                )
                guard_snapshot = await invocation_guard.snapshot()
                control = control.model_copy(
                    update={
                        "last_tool_invocation_fingerprint": (
                            guard_snapshot.fingerprint
                        ),
                        "consecutive_identical_tool_invocations": (
                            guard_snapshot.consecutive_count
                        ),
                    }
                )
                if assessment is None and guard_snapshot.blocked:
                    assessment = self._termination.explicit_stop(
                        RunTerminationReason.REPEATED_INVOCATION,
                        phase=TerminationCheckPhase.AFTER_PLANNING,
                        control=control,
                        message=(
                            "consecutive identical Tool invocation limit reached"
                        ),
                    )
                if assessment is None and isinstance(
                    exc,
                    (RunBudgetExhaustedError, RunUsageAccountingError),
                ):
                    reason = (
                        RunTerminationReason.USAGE_ACCOUNTING_UNAVAILABLE
                        if isinstance(exc, RunUsageAccountingError)
                        else (
                            RunTerminationReason.TOKEN_BUDGET_EXHAUSTED
                            if exc.resource == "tokens"
                            else RunTerminationReason.COST_BUDGET_EXHAUSTED
                        )
                    )
                    assessment = self._termination.explicit_stop(
                        reason,
                        phase=TerminationCheckPhase.AFTER_PLANNING,
                        control=control,
                        message=str(exc) or exc.__class__.__name__,
                    )
                if assessment is not None:
                    return await self._stop_run(state, control, assessment)
                return await self._fail_run(
                    state,
                    self._module_error("planner", self._planner.module_id, exc),
                    source=self._planner.module_id,
                    control=control,
                )

            usage = await ledger.snapshot()
            control = self._termination.after_planning(
                control,
                elapsed_seconds=monotonic() - planning_started,
                usage=usage,
            )
            if (
                decision.decision is PlanDecisionType.EXECUTE
                and decision.action is not None
                and decision.action.timeout_seconds is None
            ):
                decision = decision.model_copy(
                    update={
                        "action": decision.action.model_copy(
                            update={
                                "timeout_seconds": (
                                    self._stop_policy.default_tool_timeout_seconds
                                )
                            }
                        )
                    }
                )

            await self._emit(
                state,
                CoreEventKind.PLAN_CREATED,
                source=self._planner.module_id,
                payload={"plan": decision.model_dump(mode="json")},
            )

            assessment = self._termination.before_planning(
                control,
                phase=TerminationCheckPhase.AFTER_PLANNING,
            )
            guard_snapshot = await invocation_guard.snapshot()
            control = control.model_copy(
                update={
                    "last_tool_invocation_fingerprint": guard_snapshot.fingerprint,
                    "consecutive_identical_tool_invocations": (
                        guard_snapshot.consecutive_count
                    ),
                }
            )
            if assessment is None and guard_snapshot.blocked:
                assessment = self._termination.explicit_stop(
                    RunTerminationReason.REPEATED_INVOCATION,
                    phase=TerminationCheckPhase.AFTER_PLANNING,
                    control=control,
                    message="consecutive identical Tool invocation limit reached",
                )
            if assessment is not None:
                return await self._stop_run(
                    state,
                    control,
                    assessment,
                    last_plan=decision,
                )

            if decision.decision is PlanDecisionType.COMPLETE:
                previous = state
                state = complete_state(state, decision, control=control)
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
                failure = decision.failure
                if (
                    failure is not None
                    and failure.criticality.value == "critical"
                    and not failure.retryable
                    and failure.recovery_status.value in {"exhausted", "unavailable"}
                ):
                    assessment = self._termination.explicit_stop(
                        (
                            RunTerminationReason.RECOVERY_EXHAUSTED
                            if failure.recovery_status.value == "exhausted"
                            else RunTerminationReason.CRITICAL_TOOL_FAILURE
                        ),
                        phase=TerminationCheckPhase.AFTER_PLANNING,
                        control=control,
                        message=decision.error,
                    )
                    return await self._stop_run(
                        state,
                        control,
                        assessment,
                        last_plan=decision,
                    )
                return await self._fail_run(
                    state,
                    decision.error,
                    source=self._planner.module_id,
                    last_plan=decision,
                    control=control,
                )

            action = decision.action
            if action is None:
                return await self._fail_run(
                    state,
                    "planner returned an execute decision without an action",
                    source=self._planner.module_id,
                    control=control,
                )

            control, assessment = self._termination.before_action(
                state,
                action,
                control,
            )
            if assessment is not None:
                return await self._stop_run(
                    state,
                    control,
                    assessment,
                    last_plan=decision,
                )

            await self._emit(
                state,
                CoreEventKind.ACTION_STARTED,
                source=self._executor.module_id,
                payload={"action": action.model_dump(mode="json")},
            )

            execution_error: str | None = None
            action_started = monotonic()
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

            usage = await ledger.snapshot()
            guard_snapshot = await invocation_guard.snapshot()
            control = control.model_copy(
                update={
                    "last_tool_invocation_fingerprint": guard_snapshot.fingerprint,
                    "consecutive_identical_tool_invocations": (
                        guard_snapshot.consecutive_count
                    ),
                }
            )
            control, assessment = self._termination.after_action(
                control,
                action,
                observation,
                elapsed_seconds=monotonic() - action_started,
                usage=usage,
            )
            if assessment is None and guard_snapshot.blocked:
                assessment = self._termination.explicit_stop(
                    RunTerminationReason.REPEATED_INVOCATION,
                    phase=TerminationCheckPhase.AFTER_ACTION,
                    control=control,
                    message=(
                        "consecutive identical Tool invocation limit reached"
                    ),
                )

            previous = state
            state = record_observation(
                state,
                decision,
                observation,
                control=control,
            )
            await self._save_state(state)
            await self._emit_state_update(previous, state)

            reconciliation_error = await self._reconcile_observation(state)
            if reconciliation_error is not None:
                return await self._fail_run(
                    state,
                    reconciliation_error,
                    source=self._planner.module_id,
                    control=control,
                )

            if execution_error is not None:
                return await self._fail_run(
                    state,
                    execution_error,
                    source=self._executor.module_id,
                    control=control,
                )
            if assessment is not None:
                return await self._stop_run(state, control, assessment)

    async def _fail_run(
        self,
        state: AgentState,
        error: str,
        *,
        source: str,
        last_plan: PlanDecision | None = None,
        control: RunControlState | None = None,
        assessment: StopAssessment | None = None,
    ) -> RunResult:
        previous = state
        state = fail_state(
            state,
            error,
            last_plan=last_plan,
            control=control,
            termination=(assessment.termination if assessment is not None else None),
        )
        await self._save_state(state)
        await self._emit_state_update(previous, state)
        if assessment is not None:
            await self._emit(
                state,
                CoreEventKind.STOP_TRIGGERED,
                source=self.module_id,
                payload={
                    "termination": assessment.termination.model_dump(mode="json")
                },
            )
        await self._emit(
            state,
            CoreEventKind.RUNTIME_FAILED,
            source=source,
            payload={"error": error, "state": state.model_dump(mode="json")},
        )
        return RunResult(final_state=state)

    async def _stop_run(
        self,
        state: AgentState,
        control: RunControlState,
        assessment: StopAssessment,
        *,
        last_plan: PlanDecision | None = None,
    ) -> RunResult:
        if assessment.terminal_status is RunStatus.FAILED:
            return await self._fail_run(
                state,
                assessment.termination.evidence[0].message,
                source=self.module_id,
                last_plan=last_plan,
                control=control,
                assessment=assessment,
            )
        previous = state
        state = terminate_state(
            state,
            assessment.termination,
            control=control,
            last_plan=last_plan,
        )
        await self._save_state(state)
        await self._emit_state_update(previous, state)
        await self._emit(
            state,
            CoreEventKind.STOP_TRIGGERED,
            source=self.module_id,
            payload={
                "termination": assessment.termination.model_dump(mode="json")
            },
        )
        await self._emit(
            state,
            CoreEventKind.RUNTIME_TERMINATED,
            source=self.module_id,
            payload={"state": state.model_dump(mode="json")},
        )
        return RunResult(final_state=state)

    async def _save_state(self, state: AgentState) -> None:
        try:
            await self._state_store.save(state)
        except Exception as exc:
            raise RuntimeInfrastructureError(
                f"state store '{self._state_store.module_id}' failed"
            ) from exc

    async def _reconcile_observation(self, state: AgentState) -> str | None:
        if state.last_observation is None or not isinstance(
            self._planner,
            ObservationReconciler,
        ):
            return None
        try:
            await self._planner.reconcile_observation(state)
        except Exception as exc:
            return self._module_error(
                "observation reconciliation",
                self._planner.module_id,
                exc,
            )
        return None

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

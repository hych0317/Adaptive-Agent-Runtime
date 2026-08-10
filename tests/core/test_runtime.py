from __future__ import annotations

import unittest
from unittest.mock import patch
from uuid import uuid4

from adaptive_agent_runtime import (
    ActionRequest,
    AgentRuntime,
    AgentState,
    AgentTask,
    CoreEventKind,
    FailureCriticality,
    FailureDisposition,
    FailureRecoveryStatus,
    InMemoryStateStore,
    InMemoryTraceSink,
    Observation,
    ObservationControl,
    PlanDecision,
    ProgressKind,
    RunBudgetExhaustedError,
    RunStatus,
    RunStopPolicy,
    RunTerminationReason,
    RuntimeInfrastructureError,
    RuntimeInvariantError,
    RuntimeResumeBlockedError,
    RuntimeResumeError,
    RuntimeEvent,
    TraceEntry,
)
from adaptive_agent_runtime.core.budget import current_run_budget_ledger


class FeedbackPlanner:
    module_id = "test.feedback_planner"

    def __init__(self, target_steps: int = 1) -> None:
        self.target_steps = target_steps

    async def plan(self, state: AgentState) -> PlanDecision:
        if state.step_count < self.target_steps:
            return PlanDecision.execute(
                ActionRequest(name="count", arguments={"step": state.step_count + 1})
            )
        return PlanDecision.complete(
            output={
                "steps": state.step_count,
                "last": state.last_observation.output
                if state.last_observation is not None
                else None,
            }
        )


class ReconcilingPlanner(FeedbackPlanner):
    def __init__(self, target_steps: int = 1) -> None:
        super().__init__(target_steps)
        self.reconciled_states: list[AgentState] = []

    async def reconcile_observation(self, state: AgentState) -> None:
        self.reconciled_states.append(state)


class CountingExecutor:
    module_id = "test.counting_executor"

    def __init__(self) -> None:
        self.calls = 0
        self.received_states: list[AgentState] = []

    async def execute(
        self,
        action: ActionRequest,
        state: AgentState,
    ) -> Observation:
        self.calls += 1
        self.received_states.append(state)
        return Observation.ok(
            action.action_id,
            output={"seen_step": state.step_count, "nested": [self.calls]},
        )


class RaisingPlanner:
    module_id = "test.raising_planner"

    async def plan(self, state: AgentState) -> PlanDecision:
        del state
        raise ValueError("planning exploded")


class ActiveExecutionBudgetPlanner:
    module_id = "test.active_execution_budget_planner"

    async def plan(self, state: AgentState) -> PlanDecision:
        del state
        raise RunBudgetExhaustedError("active_execution", "inference timed out")


class FailingDecisionPlanner:
    module_id = "test.failing_decision_planner"

    async def plan(self, state: AgentState) -> PlanDecision:
        del state
        return PlanDecision.fail(error="task cannot be completed")


class RaisingExecutor:
    module_id = "test.raising_executor"

    async def execute(
        self,
        action: ActionRequest,
        state: AgentState,
    ) -> Observation:
        del action, state
        raise RuntimeError("execution exploded")


class FailedObservationExecutor:
    module_id = "test.failed_observation_executor"

    async def execute(
        self,
        action: ActionRequest,
        state: AgentState,
    ) -> Observation:
        del state
        return Observation.failed(action.action_id, error="expected failure")


class RepeatingPlanner:
    module_id = "test.repeating_planner"

    async def plan(self, state: AgentState) -> PlanDecision:
        return PlanDecision.execute(
            ActionRequest(
                name="lookup",
                arguments={
                    "query": "same",
                    # Volatile idempotency identity must not evade the guard.
                    "call_key": f"attempt-{state.step_count + 1}",
                },
            )
        )


class NoProgressExecutor(CountingExecutor):
    async def execute(
        self,
        action: ActionRequest,
        state: AgentState,
    ) -> Observation:
        self.calls += 1
        self.received_states.append(state)
        return Observation.ok(
            action.action_id,
            output={"unchanged": True},
            control=ObservationControl(progress_kind=ProgressKind.NO_PROGRESS),
        )


class CriticalFailureExecutor(CountingExecutor):
    async def execute(
        self,
        action: ActionRequest,
        state: AgentState,
    ) -> Observation:
        self.calls += 1
        self.received_states.append(state)
        return Observation.failed(
            action.action_id,
            error="critical provider is permanently unavailable",
            control=ObservationControl(
                progress_kind=ProgressKind.NO_PROGRESS,
                failure=FailureDisposition(
                    failure_code="tool.provider_unavailable",
                    criticality=FailureCriticality.CRITICAL,
                    retryable=False,
                    recovery_status=FailureRecoveryStatus.UNAVAILABLE,
                ),
            ),
        )


class RecoveringPlanner:
    module_id = "test.recovering_planner"

    async def plan(self, state: AgentState) -> PlanDecision:
        if state.last_observation is None:
            return PlanDecision.execute(ActionRequest(name="may-fail"))
        return PlanDecision.complete(
            output={"handled": not state.last_observation.succeeded}
        )


class FailingTraceSink:
    module_id = "test.failing_trace"

    async def record(self, event: RuntimeEvent) -> TraceEntry:
        del event
        raise OSError("trace unavailable")


class ResumeBlockedPlanner:
    module_id = "test.resume_blocked_planner"

    async def plan(self, state: AgentState) -> PlanDecision:
        del state
        raise RuntimeResumeBlockedError("outcome is in doubt")


class BudgetThenBlockedPlanner:
    module_id = "test.budget_then_blocked_planner"

    async def plan(self, state: AgentState) -> PlanDecision:
        del state
        ledger = current_run_budget_ledger()
        assert ledger is not None
        await ledger.record_usage(
            total_tokens=3,
            monetary_cost=None,
            currency=None,
        )
        raise RuntimeResumeBlockedError("outcome is in doubt")


class RuntimeLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_execution_budget_error_is_controlled_termination(
        self,
    ) -> None:
        trace = InMemoryTraceSink()
        result = await AgentRuntime(
            planner=ActiveExecutionBudgetPlanner(),
            executor=CountingExecutor(),
            state_store=InMemoryStateStore(),
            trace_sink=trace,
        ).run(AgentTask(description="bound inference time"))

        self.assertEqual(result.final_state.status, RunStatus.TERMINATED)
        self.assertIsNone(result.final_state.error)
        assert result.final_state.termination is not None
        self.assertEqual(
            result.final_state.termination.primary_reason,
            RunTerminationReason.ACTIVE_EXECUTION_BUDGET,
        )
        kinds = [
            entry.event.kind
            for entry in trace.entries_for(result.final_state.run_id)
        ]
        self.assertIn(CoreEventKind.RUNTIME_TERMINATED, kinds)
        self.assertNotIn(CoreEventKind.RUNTIME_FAILED, kinds)

    async def test_single_step_loop_and_trace_order(self) -> None:
        store = InMemoryStateStore()
        trace = InMemoryTraceSink()
        executor = CountingExecutor()
        runtime = AgentRuntime(
            planner=FeedbackPlanner(target_steps=1),
            executor=executor,
            state_store=store,
            trace_sink=trace,
        )

        result = await runtime.run(AgentTask(description="single step"))

        self.assertTrue(result.succeeded)
        self.assertEqual(result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(result.final_state.step_count, 1)
        entries = trace.entries_for(result.final_state.run_id)
        self.assertEqual(
            [entry.event.kind for entry in entries],
            [
                CoreEventKind.RUNTIME_STARTED,
                CoreEventKind.PLAN_CREATED,
                CoreEventKind.ACTION_STARTED,
                CoreEventKind.OBSERVATION_RECEIVED,
                CoreEventKind.STATE_UPDATED,
                CoreEventKind.PLAN_CREATED,
                CoreEventKind.STATE_UPDATED,
                CoreEventKind.RUNTIME_COMPLETED,
            ],
        )
        self.assertEqual(
            [entry.sequence for entry in entries],
            list(range(1, len(entries) + 1)),
        )
        self.assertEqual(
            {entry.run_id for entry in entries},
            {result.final_state.run_id},
        )

    async def test_multi_step_feedback_uses_new_snapshots(self) -> None:
        store = InMemoryStateStore()
        trace = InMemoryTraceSink()
        executor = CountingExecutor()
        runtime = AgentRuntime(
            planner=FeedbackPlanner(target_steps=3),
            executor=executor,
            state_store=store,
            trace_sink=trace,
        )

        result = await runtime.run(AgentTask(description="three steps"))

        self.assertEqual(result.final_state.step_count, 3)
        self.assertEqual(
            [state.step_count for state in executor.received_states],
            [0, 1, 2],
        )
        self.assertEqual(len({id(state) for state in executor.received_states}), 3)
        history = store.history_for(result.final_state.run_id)
        self.assertEqual(history[0].status, RunStatus.PENDING)
        self.assertEqual(history[0].step_count, 0)
        self.assertEqual(history[-1].status, RunStatus.COMPLETED)

    async def test_failed_observation_is_feedback_not_runtime_exception(self) -> None:
        store = InMemoryStateStore()
        trace = InMemoryTraceSink()
        runtime = AgentRuntime(
            planner=RecoveringPlanner(),
            executor=FailedObservationExecutor(),
            state_store=store,
            trace_sink=trace,
        )

        result = await runtime.run(AgentTask(description="recover"))

        self.assertTrue(result.succeeded)
        self.assertEqual(result.final_state.output, {"handled": True})
        self.assertIsNotNone(result.final_state.last_observation)
        assert result.final_state.last_observation is not None
        self.assertFalse(result.final_state.last_observation.succeeded)

    async def test_planner_exception_fails_the_run(self) -> None:
        store = InMemoryStateStore()
        trace = InMemoryTraceSink()
        runtime = AgentRuntime(
            planner=RaisingPlanner(),
            executor=CountingExecutor(),
            state_store=store,
            trace_sink=trace,
        )

        result = await runtime.run(AgentTask(description="planner failure"))

        self.assertFalse(result.succeeded)
        self.assertEqual(result.final_state.status, RunStatus.FAILED)
        self.assertIsNotNone(result.final_state.error)
        assert result.final_state.error is not None
        self.assertIn("planning exploded", result.final_state.error)
        self.assertEqual(
            trace.entries_for(result.final_state.run_id)[-1].event.kind,
            CoreEventKind.RUNTIME_FAILED,
        )

    async def test_planner_can_end_with_a_task_failure_decision(self) -> None:
        store = InMemoryStateStore()
        trace = InMemoryTraceSink()
        runtime = AgentRuntime(
            planner=FailingDecisionPlanner(),
            executor=CountingExecutor(),
            state_store=store,
            trace_sink=trace,
        )

        result = await runtime.run(AgentTask(description="cannot complete"))

        self.assertEqual(result.final_state.status, RunStatus.FAILED)
        self.assertEqual(result.final_state.error, "task cannot be completed")
        self.assertIsNotNone(result.final_state.last_plan)
        assert result.final_state.last_plan is not None
        self.assertEqual(result.final_state.last_plan.decision.value, "fail")

    async def test_executor_exception_records_observation_then_fails(self) -> None:
        store = InMemoryStateStore()
        trace = InMemoryTraceSink()
        runtime = AgentRuntime(
            planner=FeedbackPlanner(target_steps=1),
            executor=RaisingExecutor(),
            state_store=store,
            trace_sink=trace,
        )

        result = await runtime.run(AgentTask(description="executor failure"))

        self.assertEqual(result.final_state.status, RunStatus.FAILED)
        self.assertEqual(result.final_state.step_count, 1)
        self.assertIsNotNone(result.final_state.last_observation)
        assert result.final_state.last_observation is not None
        self.assertFalse(result.final_state.last_observation.succeeded)
        kinds = [
            entry.event.kind
            for entry in trace.entries_for(result.final_state.run_id)
        ]
        self.assertIn(CoreEventKind.OBSERVATION_RECEIVED, kinds)
        self.assertEqual(kinds[-1], CoreEventKind.RUNTIME_FAILED)

    async def test_max_steps_limits_executor_calls_but_allows_final_plan(self) -> None:
        store = InMemoryStateStore()
        trace = InMemoryTraceSink()
        executor = CountingExecutor()
        runtime = AgentRuntime(
            planner=FeedbackPlanner(target_steps=3),
            executor=executor,
            state_store=store,
            trace_sink=trace,
            max_steps=2,
        )

        result = await runtime.run(AgentTask(description="bounded"))

        self.assertEqual(executor.calls, 2)
        self.assertEqual(result.final_state.step_count, 2)
        self.assertEqual(result.final_state.status, RunStatus.TERMINATED)
        self.assertIsNone(result.final_state.error)
        self.assertIsNotNone(result.final_state.termination)
        assert result.final_state.termination is not None
        self.assertEqual(
            result.final_state.termination.primary_reason.value,
            "max_action_steps",
        )

    async def test_third_identical_action_is_stopped_before_execution(self) -> None:
        executor = CountingExecutor()
        result = await AgentRuntime(
            planner=RepeatingPlanner(),
            executor=executor,
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
            stop_policy=RunStopPolicy(
                max_action_steps=10,
                repeated_invocation_limit=3,
                max_no_progress_steps=None,
                max_no_progress_seconds=None,
            ),
        ).run(AgentTask(description="detect an invocation loop"))

        self.assertEqual(executor.calls, 2)
        self.assertEqual(result.final_state.status, RunStatus.TERMINATED)
        assert result.final_state.termination is not None
        self.assertEqual(
            result.final_state.termination.primary_reason,
            RunTerminationReason.REPEATED_INVOCATION,
        )

    async def test_no_progress_steps_terminate_even_when_actions_differ(self) -> None:
        executor = NoProgressExecutor()
        result = await AgentRuntime(
            planner=FeedbackPlanner(target_steps=10),
            executor=executor,
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
            stop_policy=RunStopPolicy(
                max_action_steps=10,
                repeated_invocation_limit=None,
                max_no_progress_steps=2,
                max_no_progress_seconds=None,
            ),
        ).run(AgentTask(description="detect stagnation"))

        self.assertEqual(executor.calls, 2)
        self.assertEqual(result.final_state.status, RunStatus.TERMINATED)
        assert result.final_state.termination is not None
        self.assertEqual(
            result.final_state.termination.primary_reason,
            RunTerminationReason.NO_PROGRESS_STEPS,
        )

    async def test_observation_is_reconciled_before_after_action_termination(
        self,
    ) -> None:
        planner = ReconcilingPlanner(target_steps=10)
        result = await AgentRuntime(
            planner=planner,
            executor=NoProgressExecutor(),
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
            stop_policy=RunStopPolicy(
                max_action_steps=10,
                repeated_invocation_limit=None,
                max_no_progress_steps=1,
                max_no_progress_seconds=None,
            ),
        ).run(AgentTask(description="reconcile the terminal observation"))

        self.assertEqual(result.final_state.status, RunStatus.TERMINATED)
        self.assertEqual(len(planner.reconciled_states), 1)
        reconciled = planner.reconciled_states[0]
        self.assertEqual(reconciled.step_count, 1)
        self.assertEqual(
            reconciled.last_observation,
            result.final_state.last_observation,
        )

    async def test_slow_planning_does_not_block_valid_recovery_action(self) -> None:
        executor = CountingExecutor()
        runtime = AgentRuntime(
            planner=FeedbackPlanner(target_steps=1),
            executor=executor,
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
            stop_policy=RunStopPolicy(
                max_action_steps=2,
                repeated_invocation_limit=None,
                max_no_progress_steps=None,
                max_no_progress_seconds=5.0,
            ),
        )

        with patch(
            "adaptive_agent_runtime.core.runtime.monotonic",
            side_effect=(0.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        ):
            result = await runtime.run(
                AgentTask(description="execute a valid recovery after slow planning")
            )

        self.assertEqual(executor.calls, 1)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(
            result.final_state.control.pending_planning_seconds,
            0.0,
        )

    async def test_critical_nonrecoverable_failure_is_structured_failure(self) -> None:
        result = await AgentRuntime(
            planner=FeedbackPlanner(target_steps=2),
            executor=CriticalFailureExecutor(),
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
        ).run(AgentTask(description="critical Tool failure"))

        self.assertEqual(result.final_state.status, RunStatus.FAILED)
        assert result.final_state.termination is not None
        self.assertEqual(
            result.final_state.termination.primary_reason,
            RunTerminationReason.CRITICAL_TOOL_FAILURE,
        )

    async def test_tool_is_not_started_without_its_full_deadline(self) -> None:
        class LongToolPlanner:
            module_id = "test.long_tool_planner"

            async def plan(self, state: AgentState) -> PlanDecision:
                del state
                return PlanDecision.execute(
                    ActionRequest(name="long", timeout_seconds=5.0)
                )

        executor = CountingExecutor()
        result = await AgentRuntime(
            planner=LongToolPlanner(),
            executor=executor,
            state_store=InMemoryStateStore(),
            trace_sink=InMemoryTraceSink(),
            stop_policy=RunStopPolicy(
                max_action_steps=2,
                max_wall_clock_seconds=2.0,
                cleanup_grace_seconds=1.0,
                max_no_progress_steps=None,
                max_no_progress_seconds=None,
            ),
        ).run(AgentTask(description="admit a long Tool"))

        self.assertEqual(executor.calls, 0)
        self.assertEqual(result.final_state.status, RunStatus.TERMINATED)
        assert result.final_state.termination is not None
        self.assertEqual(
            result.final_state.termination.primary_reason,
            RunTerminationReason.WALL_CLOCK_DEADLINE,
        )

    async def test_trace_failure_is_not_reported_as_success(self) -> None:
        runtime = AgentRuntime(
            planner=FeedbackPlanner(),
            executor=CountingExecutor(),
            state_store=InMemoryStateStore(),
            trace_sink=FailingTraceSink(),
        )

        with self.assertRaises(RuntimeInfrastructureError):
            await runtime.run(AgentTask(description="trace failure"))

    async def test_duplicate_run_id_is_rejected_before_trace_is_appended(self) -> None:
        store = InMemoryStateStore()
        trace = InMemoryTraceSink()
        runtime = AgentRuntime(
            planner=FeedbackPlanner(),
            executor=CountingExecutor(),
            state_store=store,
            trace_sink=trace,
        )
        run_id = uuid4()

        first = await runtime.run(AgentTask(description="first"), run_id=run_id)
        first_trace = trace.entries_for(run_id)

        with self.assertRaises(RuntimeInvariantError):
            await runtime.run(AgentTask(description="second"), run_id=run_id)

        self.assertEqual(await store.load(run_id), first.final_state)
        self.assertEqual(trace.entries_for(run_id), first_trace)

    async def test_resume_rejects_a_different_stop_policy(self) -> None:
        store = InMemoryStateStore()
        run_id = uuid4()
        first = AgentRuntime(
            planner=ResumeBlockedPlanner(),
            executor=CountingExecutor(),
            state_store=store,
            trace_sink=InMemoryTraceSink(),
            stop_policy=RunStopPolicy(max_action_steps=4),
        )
        with self.assertRaises(RuntimeResumeBlockedError):
            await first.run(
                AgentTask(description="persist policy"),
                run_id=run_id,
            )

        resumed = AgentRuntime(
            planner=FeedbackPlanner(),
            executor=CountingExecutor(),
            state_store=store,
            trace_sink=InMemoryTraceSink(),
            stop_policy=RunStopPolicy(max_action_steps=5),
        )
        with self.assertRaises(RuntimeResumeError):
            await resumed.resume(run_id)

    async def test_resume_block_persists_usage_without_advancing_action(self) -> None:
        store = InMemoryStateStore()
        run_id = uuid4()
        policy = RunStopPolicy(max_total_tokens=10)
        blocked = AgentRuntime(
            planner=BudgetThenBlockedPlanner(),
            executor=CountingExecutor(),
            state_store=store,
            trace_sink=InMemoryTraceSink(),
            stop_policy=policy,
        )

        with self.assertRaises(RuntimeResumeBlockedError):
            await blocked.run(
                AgentTask(description="account before blocking"),
                run_id=run_id,
            )

        persisted = await store.load(run_id)
        self.assertIsNotNone(persisted)
        assert persisted is not None
        self.assertEqual(persisted.status, RunStatus.RUNNING)
        self.assertEqual(persisted.step_count, 0)
        self.assertEqual(persisted.control.usage.total_tokens, 3)

        resumed = await AgentRuntime(
            planner=FeedbackPlanner(target_steps=0),
            executor=CountingExecutor(),
            state_store=store,
            trace_sink=InMemoryTraceSink(),
            stop_policy=policy,
        ).resume(run_id)
        self.assertEqual(resumed.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(resumed.final_state.control.usage.total_tokens, 3)


if __name__ == "__main__":
    unittest.main()

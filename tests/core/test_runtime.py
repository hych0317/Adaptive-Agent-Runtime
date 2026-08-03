from __future__ import annotations

import unittest
from uuid import uuid4

from adaptive_agent_runtime import (
    ActionRequest,
    AgentRuntime,
    AgentState,
    AgentTask,
    CoreEventKind,
    InMemoryStateStore,
    InMemoryTraceSink,
    Observation,
    PlanDecision,
    RunStatus,
    RuntimeInfrastructureError,
    RuntimeInvariantError,
    RuntimeEvent,
    TraceEntry,
)


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


class RuntimeLoopTests(unittest.IsolatedAsyncioTestCase):
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
        self.assertEqual(result.final_state.status, RunStatus.FAILED)
        self.assertIsNotNone(result.final_state.error)
        assert result.final_state.error is not None
        self.assertIn("maximum action steps", result.final_state.error)

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


if __name__ == "__main__":
    unittest.main()

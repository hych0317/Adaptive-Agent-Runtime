from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from adaptive_agent_runtime import (
    ActionRequest,
    Observation,
    ObservationControl,
    ProgressKind,
    RunStopPolicy,
    RunTerminationReason,
    RunUsage,
)
from adaptive_agent_runtime.core.models import TerminationCheckPhase
from adaptive_agent_runtime.core.termination import RunTerminationController


class MutableClock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current


class RunTerminationControllerTests(unittest.TestCase):
    def test_tool_duration_is_active_time_but_not_no_progress_time(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        controller = RunTerminationController(
            RunStopPolicy(
                max_active_execution_seconds=200.0,
                max_no_progress_seconds=5.0,
                max_no_progress_steps=None,
            ),
            clock=MutableClock(now),
        )
        control = controller.initial_control(now)
        control = controller.after_planning(
            control,
            elapsed_seconds=2.0,
            usage=RunUsage(),
        )
        self.assertEqual(control.no_progress_active_seconds, 0.0)
        self.assertEqual(control.pending_planning_seconds, 2.0)
        action = ActionRequest(name="long-running-tool")
        control, assessment = controller.after_action(
            control,
            action,
            Observation.ok(
                action.action_id,
                control=ObservationControl(
                    progress_kind=ProgressKind.NO_PROGRESS
                ),
            ),
            elapsed_seconds=100.0,
            usage=RunUsage(),
        )

        self.assertIsNone(assessment)
        self.assertEqual(control.active_execution_seconds, 102.0)
        self.assertEqual(control.no_progress_active_seconds, 2.0)
        self.assertEqual(control.pending_planning_seconds, 0.0)
        self.assertEqual(control.no_progress_steps, 1)
    def test_failed_planning_time_is_not_stagnation_evidence(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        controller = RunTerminationController(
            RunStopPolicy(max_no_progress_seconds=5.0),
            clock=MutableClock(now),
        )
        control = controller.after_planning(
            controller.initial_control(now),
            elapsed_seconds=10.0,
            usage=RunUsage(),
        )

        control = controller.resolve_planning_without_action(
            control,
            made_progress=False,
        )

        self.assertEqual(control.pending_planning_seconds, 0.0)
        self.assertEqual(control.no_progress_active_seconds, 0.0)


    def test_wall_clock_deadline_includes_waiting_and_downtime(self) -> None:
        created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        clock = MutableClock(created_at)
        controller = RunTerminationController(
            RunStopPolicy(max_wall_clock_seconds=60.0),
            clock=clock,
        )
        control = controller.initial_control(created_at)
        clock.current += timedelta(seconds=61)

        assessment = controller.before_planning(
            control,
            phase=TerminationCheckPhase.RESUME,
        )

        self.assertIsNotNone(assessment)
        assert assessment is not None
        self.assertEqual(
            assessment.termination.primary_reason,
            RunTerminationReason.WALL_CLOCK_DEADLINE,
        )
        self.assertIsNotNone(control.deadline_at)

    def test_external_job_has_its_own_deadline(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        clock = MutableClock(now)
        controller = RunTerminationController(
            RunStopPolicy(
                external_job_deadline_seconds=10.0,
                max_no_progress_seconds=None,
                max_no_progress_steps=None,
            ),
            clock=clock,
        )
        control = controller.initial_control(now)
        action = ActionRequest(name="submit-job")
        control, assessment = controller.after_action(
            control,
            action,
            Observation.ok(
                action.action_id,
                control=ObservationControl(
                    external_job_id="job-1",
                    external_job_heartbeat=True,
                ),
            ),
            elapsed_seconds=1.0,
            usage=RunUsage(),
        )
        self.assertIsNone(assessment)
        self.assertEqual(control.no_progress_steps, 0)

        clock.current += timedelta(seconds=11)
        poll = ActionRequest(name="poll-job")
        _, assessment = controller.after_action(
            control,
            poll,
            Observation.ok(
                poll.action_id,
                control=ObservationControl(
                    progress_kind=ProgressKind.NO_PROGRESS
                ),
            ),
            elapsed_seconds=0.1,
            usage=RunUsage(),
        )

        self.assertIsNotNone(assessment)
        assert assessment is not None
        self.assertEqual(
            assessment.termination.primary_reason,
            RunTerminationReason.EXTERNAL_JOB_DEADLINE,
        )

    def test_cost_budget_has_a_distinct_stop_reason(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        controller = RunTerminationController(
            RunStopPolicy(max_monetary_cost=0.5, currency="USD"),
            clock=MutableClock(now),
        )
        control = controller.initial_control(now).model_copy(
            update={
                "usage": RunUsage(monetary_cost=0.5, currency="USD")
            }
        )

        assessment = controller.before_planning(
            control,
            phase=TerminationCheckPhase.BEFORE_PLANNING,
        )

        self.assertIsNotNone(assessment)
        assert assessment is not None
        self.assertEqual(
            assessment.termination.primary_reason,
            RunTerminationReason.COST_BUDGET_EXHAUSTED,
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch
from typing import Any, cast
from uuid import uuid4

from pydantic import ValidationError

from adaptive_agent_runtime import (
    ActionRequest,
    AgentState,
    AgentTask,
    Observation,
    PlanDecision,
)
from adaptive_agent_runtime.core.state import record_observation, start_state


NOW = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)


class ImmutableModelTests(unittest.TestCase):
    def test_state_timestamps_do_not_regress_with_wall_clock(self) -> None:
        pending = AgentState(
            run_id=uuid4(),
            task=AgentTask(description="test"),
            created_at=NOW,
            updated_at=NOW,
        )
        with patch(
            "adaptive_agent_runtime.core.state.utc_now",
            return_value=NOW - timedelta(milliseconds=25),
        ):
            running = start_state(pending)

        self.assertEqual(running.updated_at, NOW)
        defaulted = AgentState(
            run_id=uuid4(),
            task=AgentTask(description="default timestamps"),
        )
        self.assertEqual(defaulted.updated_at, defaulted.created_at)

    def test_nested_task_input_is_deeply_immutable(self) -> None:
        source: dict[str, Any] = {"filters": [{"sector": "energy"}]}
        task = AgentTask(description="research", input=source)

        source["filters"][0]["sector"] = "changed"

        frozen_input = cast(Any, task.input)
        self.assertEqual(frozen_input["filters"][0]["sector"], "energy")
        with self.assertRaises(TypeError):
            frozen_input["filters"][0]["sector"] = "changed"

    def test_state_transition_creates_a_new_snapshot(self) -> None:
        pending = AgentState(run_id=uuid4(), task=AgentTask(description="test"))
        running = start_state(pending)
        action = ActionRequest(name="step", arguments={"items": [1, 2]})
        plan = PlanDecision.execute(action)
        observation = Observation.ok(action.action_id, output={"value": [1, 2]})

        updated = record_observation(running, plan, observation)

        self.assertIsNot(updated, running)
        self.assertEqual(running.step_count, 0)
        self.assertIsNone(running.last_observation)
        self.assertEqual(updated.step_count, 1)
        self.assertEqual(updated.revision, running.revision + 1)
        self.assertIsNotNone(updated.last_observation)
        assert updated.last_observation is not None
        output = cast(Any, updated.last_observation.output)
        with self.assertRaises(TypeError):
            output["value"][0] = 9

    def test_failed_observation_requires_error(self) -> None:
        with self.assertRaises(ValidationError):
            Observation(action_id=uuid4(), succeeded=False)

    def test_json_keys_are_not_silently_coerced(self) -> None:
        with self.assertRaises(ValidationError):
            AgentTask(
                description="invalid key",
                input=cast(Any, {1: "integer", "1": "string"}),
            )

    def test_non_finite_json_numbers_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Observation.ok(uuid4(), output={"value": float("nan")})


if __name__ == "__main__":
    unittest.main()

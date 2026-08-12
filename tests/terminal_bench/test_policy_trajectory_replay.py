from __future__ import annotations

import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from adaptive_agent_runtime.core import (
    AgentState,
    AgentTask,
    PlanDecisionType,
    RunStatus,
)
from adaptive_agent_runtime.llm import (
    InferenceExecutionBudgetError,
    InferenceUsage,
)

from applications.terminal_bench.composition import build_terminal_application
from applications.terminal_bench.contracts import TerminalExecutionError
from applications.terminal_bench.deadline import (
    TerminalDeadlineSequence,
    terra_high_deadline_budget_profile,
)
from applications.terminal_bench.models import (
    TERMINAL_COMMAND_ACTION,
    TerminalCommandRole,
    TerminalCompletionDisposition,
    TerminalExecutionPolicy,
    TerminalReconciliationState,
    TerminalTurnProposal,
    utc_now,
)
from applications.terminal_bench.planner import (
    _task_requirements,
    _terminal_behavior_hints,
)
from tests.terminal_bench.fakes import (
    FakeTerminalEnvironment,
    ScriptedTerminalTurnCapability,
    completed_result,
    execute_draft,
    verify_draft,
)


_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "policy_trajectories_v1.json"
)


def _fixtures() -> dict[str, object]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _scenario(name: str) -> dict[str, object]:
    scenarios = _fixtures()["scenarios"]
    assert isinstance(scenarios, dict)
    scenario = scenarios[name]
    assert isinstance(scenario, dict)
    return scenario


class _ProfiledScriptedCapability(ScriptedTerminalTurnCapability):
    deadline_budget_profile = terra_high_deadline_budget_profile()


class _EmergencyWindowCapability:
    """Model one ordinary timeout followed by a useful emergency response."""

    module_id = "test.terminal_turn.trajectory_window"
    deadline_budget_profile = terra_high_deadline_budget_profile()

    def __init__(self, minimum_useful_seconds: float) -> None:
        self.minimum_useful_seconds = minimum_useful_seconds
        self.requests = []

    async def propose(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        if len(self.requests) == 1:
            raise InferenceExecutionBudgetError(
                "scripted ordinary inference exceeded its allocation"
            )
        available = (
            request.execution_limits.max_inference_timeout_sec or 0.0
        )
        if available < self.minimum_useful_seconds:
            raise InferenceExecutionBudgetError(
                "scripted emergency inference requires "
                f"{self.minimum_useful_seconds:.1f} seconds; "
                f"received {available:.1f}"
            )
        return TerminalTurnProposal(
            draft=execute_draft(
                "test ! -e /tmp/policy-worker.pid && test -d /app",
                call_key="reconcile-known-state-1",
                command_role=TerminalCommandRole.INSPECT,
                timeout_sec=15,
            ),
            usage=InferenceUsage(),
            model_id="test/trajectory",
        )


class TerminalPolicyTrajectoryFixtureTests(unittest.TestCase):
    def test_fixture_is_versioned_minimal_and_redacted(self) -> None:
        payload = _fixtures()
        self.assertEqual(payload["schema_version"], 1)
        encoded = json.dumps(payload, sort_keys=True).lower()
        for forbidden in (
            "api_key",
            "access_token",
            "stdout",
            "stderr",
            "model_output",
            "full_command",
        ):
            self.assertNotIn(forbidden, encoded)
        scenarios = payload["scenarios"]
        assert isinstance(scenarios, dict)
        self.assertEqual(
            set(scenarios),
            {
                "build-pmars-reconciliation-window",
                "failed-verify-targeted-repair",
                "cancel-cleanup-coverage",
                "public-api-signature-coverage",
                "exact-output-control",
                "multi-consumer-control",
                "immutable-host-path",
            },
        )

    @unittest.expectedFailure
    def test_cancel_cleanup_requirement_has_behavior_coverage(self) -> None:
        scenario = _scenario("cancel-cleanup-coverage")
        requirements = _task_requirements(str(scenario["instruction"]))
        hints = " ".join(_terminal_behavior_hints(requirements)).lower()
        for term in scenario["expected_hint_terms"]:
            self.assertIn(str(term), hints)

    @unittest.expectedFailure
    def test_public_api_signature_has_behavior_coverage(self) -> None:
        scenario = _scenario("public-api-signature-coverage")
        requirements = _task_requirements(str(scenario["instruction"]))
        hints = " ".join(_terminal_behavior_hints(requirements)).lower()
        for term in scenario["expected_hint_terms"]:
            self.assertIn(str(term), hints)


class TerminalPolicyTrajectoryReplayTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _state(instruction: str) -> AgentState:
        return AgentState(
            run_id=uuid4(),
            task=AgentTask(description=instruction),
            status=RunStatus.RUNNING,
        )

    @staticmethod
    def _policy() -> TerminalExecutionPolicy:
        return TerminalExecutionPolicy(
            max_wall_clock_seconds=840.0,
            max_no_progress_seconds=None,
        )

    @unittest.expectedFailure
    async def test_build_pmars_window_admits_useful_emergency_reconciliation(
        self,
    ) -> None:
        scenario = _scenario("build-pmars-reconciliation-window")
        capability = _EmergencyWindowCapability(
            float(scenario["minimum_useful_emergency_inference_seconds"])
        )
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="trajectory-build-pmars-reconciliation",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(),
                proposal_capability=capability,
                policy=self._policy(),
            )
            try:
                app.journal._session = app.journal.snapshot().model_copy(
                    update={
                        "started_at": utc_now()
                        - timedelta(seconds=float(scenario["elapsed_seconds"])),
                        "committed_commands": int(
                            scenario["committed_commands"]
                        ),
                        "timed_out_commands": 1,
                        "in_doubt_commands": 1,
                        "in_doubt_reconciliation_required": True,
                        "reconciliation_state": (
                            TerminalReconciliationState.REQUIRED
                        ),
                        "reconciliation_proposal_rejections": int(
                            scenario["reconciliation_rejections"]
                        ),
                        "task_generation": int(
                            scenario["task_generation"]
                        ),
                        "known_state_generation": None,
                        "successful_work_generation": None,
                    }
                )
                decision = await app.runtime._planner.plan(
                    self._state(str(scenario["instruction"]))
                )

                self.assertIs(decision.decision, PlanDecisionType.EXECUTE)
                assert decision.action is not None
                self.assertEqual(decision.action.name, TERMINAL_COMMAND_ACTION)
                self.assertEqual(
                    decision.action.arguments["command_role"],
                    scenario["expected_next_role"],
                )
                self.assertEqual(len(capability.requests), 2)
                self.assertTrue(capability.requests[-1].emergency_mode)
                self.assertGreaterEqual(
                    capability.requests[-1]
                    .execution_limits.max_inference_timeout_sec
                    or 0.0,
                    capability.minimum_useful_seconds,
                )
            finally:
                app.close()

    async def test_failed_verify_requests_targeted_work_before_verify(
        self,
    ) -> None:
        scenario = _scenario("failed-verify-targeted-repair")
        capability = _ProfiledScriptedCapability(
            execute_draft(
                "printf repaired > /app/artifact",
                call_key="targeted-repair-1",
                command_role=TerminalCommandRole.WORK,
                timeout_sec=60,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="trajectory-failed-verify-repair",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(),
                proposal_capability=capability,
                policy=self._policy(),
            )
            try:
                app.journal._session = app.journal.snapshot().model_copy(
                    update={
                        "started_at": utc_now()
                        - timedelta(seconds=float(scenario["elapsed_seconds"])),
                        "committed_commands": int(
                            scenario["committed_commands"]
                        ),
                        "task_generation": int(
                            scenario["task_generation"]
                        ),
                        "known_state_generation": int(
                            scenario["task_generation"]
                        ),
                        "successful_work_generation": int(
                            scenario["task_generation"]
                        ),
                        "failed_verification_attempts": int(
                            scenario["failed_verification_attempts"]
                        ),
                        "pending_repair_receipt_id": uuid4(),
                    }
                )
                decision = await app.runtime._planner.plan(
                    self._state(str(scenario["instruction"]))
                )

                self.assertIs(decision.decision, PlanDecisionType.EXECUTE)
                assert decision.action is not None
                self.assertEqual(decision.action.name, TERMINAL_COMMAND_ACTION)
                self.assertEqual(
                    decision.action.arguments["command_role"],
                    scenario["expected_next_role"],
                )
                request = capability.requests[0]
                self.assertTrue(request.repair_mode)
                self.assertIs(
                    request.execution_limits.deadline_sequence,
                    TerminalDeadlineSequence.WORK_THEN_VERIFY,
                )
            finally:
                app.close()

    async def test_control_trajectories_lock_success_without_extra_actions(
        self,
    ) -> None:
        for name in ("exact-output-control", "multi-consumer-control"):
            scenario = _scenario(name)
            replays: list[tuple[object, ...]] = []
            with self.subTest(scenario=name):
                for _ in range(2):
                    with tempfile.TemporaryDirectory() as directory:
                        environment = FakeTerminalEnvironment(
                            completed_result(stdout="artifact ready"),
                            completed_result(stdout="official check passed"),
                        )
                        capability = _ProfiledScriptedCapability(
                            execute_draft(
                                "printf artifact > /app/artifact",
                                call_key="work-1",
                            ),
                            verify_draft(
                                "python3 /tests/scripted_official_test.py",
                                call_key="verify-1",
                            ),
                            execute_draft(
                                "printf forbidden > /app/after-lock",
                                call_key="must-not-run",
                            ),
                        )
                        app = build_terminal_application(
                            trial_id=f"trajectory-{name}",
                            logs_dir=directory,
                            environment=environment,
                            proposal_capability=capability,
                        )
                        try:
                            artifacts = await app.run(
                                str(scenario["instruction"])
                            )
                            self.assertTrue(artifacts.summary.agent_complete)
                            disposition = (
                                app.journal.snapshot().completion_disposition
                            )
                            self.assertEqual(
                                disposition,
                                TerminalCompletionDisposition.SUCCESS_LOCKED,
                            )
                            self.assertEqual(
                                len(environment.calls),
                                scenario["expected_commands"],
                            )
                            self.assertEqual(len(capability.requests), 2)
                            self.assertTrue(artifacts.summary.trace_consistent)
                            replays.append(
                                (
                                    tuple(
                                        (
                                            item.intent.call_key,
                                            item.intent.command_role.value,
                                            item.intent.timeout_sec,
                                            item.intent.advertised_timeout_cap_sec,
                                            item.intent.timeout_cap_reason.value,
                                        )
                                        for item in app.journal.records()
                                    ),
                                    disposition.value,
                                )
                            )
                        finally:
                            app.close()
                self.assertEqual(replays[0], replays[1])

    async def test_unresolved_in_doubt_blocks_work_and_submission(self) -> None:
        environment = FakeTerminalEnvironment(
            TerminalExecutionError(
                "scripted command timeout",
                command_started=True,
                timed_out=True,
            )
        )
        capability = _ProfiledScriptedCapability(
            execute_draft("bounded mutation", call_key="work-1", timeout_sec=30),
            execute_draft("unsafe retry", call_key="work-2", timeout_sec=30),
        )
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="trajectory-in-doubt-hard-protection",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(
                    max_reconciliation_proposal_rejections=0,
                ),
            )
            try:
                artifacts = await app.run(
                    "Create one artifact without replaying an uncertain mutation."
                )
                self.assertEqual(len(environment.calls), 1)
                self.assertEqual(
                    artifacts.runtime_result.final_state.status,
                    RunStatus.FAILED,
                )
                self.assertEqual(
                    app.journal.snapshot().completion_disposition,
                    TerminalCompletionDisposition.IN_PROGRESS,
                )
                self.assertTrue(
                    app.journal.snapshot().in_doubt_reconciliation_required
                )
                self.assertTrue(artifacts.summary.trace_consistent)
            finally:
                app.close()

    async def test_immutable_host_path_fails_before_tool_execution(self) -> None:
        scenario = _scenario("immutable-host-path")
        environment = FakeTerminalEnvironment()
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="trajectory-immutable-host-path",
                logs_dir=directory,
                environment=environment,
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft(
                        "cat /tmp/aar-codex-inference-redacted/artifact",
                        call_key="invalid-host-path-1",
                    )
                ),
                policy=TerminalExecutionPolicy(max_proposal_rejections=0),
            )
            try:
                artifacts = await app.run(str(scenario["instruction"]))
                self.assertEqual(
                    artifacts.runtime_result.final_state.status,
                    RunStatus.FAILED,
                )
                self.assertEqual(
                    len(environment.calls),
                    scenario["expected_commands"],
                )
                self.assertTrue(artifacts.summary.trace_consistent)
            finally:
                app.close()


if __name__ == "__main__":
    unittest.main()

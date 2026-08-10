from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from adaptive_agent_runtime.llm import (
    InferenceExecutionBudgetError,
    InferenceUsage,
    ReasoningEffort,
)

from applications.terminal_bench.harbor_agent import AgentContext
from applications.terminal_bench.models import (
    TerminalCommandRole,
    TerminalExecResult,
    TerminalExecutionPolicy,
    TerminalExecutionState,
    TerminalTrialSummary,
    utc_now,
)
from applications.terminal_bench.plain_harbor_agent import (
    PlainSequentialHarborAgent,
)
from applications.terminal_bench.plain_sequential import (
    PLAIN_SEQUENTIAL_PROFILE,
    PlainRunArtifacts,
    build_plain_sequential_application,
)
from tests.terminal_bench.fakes import (
    FakeTerminalEnvironment,
    ScriptedTerminalTurnCapability,
    complete_draft,
    completed_result,
    execute_draft,
    verify_draft,
)


class _FakePlainApplication:
    def __init__(self, artifacts: PlainRunArtifacts) -> None:
        self.artifacts = artifacts
        self.closed = False

    async def run(self, instruction: str) -> PlainRunArtifacts:
        del instruction
        return self.artifacts

    def close(self) -> None:
        self.closed = True


class _BudgetExhaustedCapability:
    module_id = "test.plain.budget_exhausted"

    def __init__(self) -> None:
        self.requests = []

    async def propose(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        raise InferenceExecutionBudgetError(
            "insufficient wall-clock capacity for another inference"
        )


def _summary() -> TerminalTrialSummary:
    now = utc_now()
    return TerminalTrialSummary(
        profile=PLAIN_SEQUENTIAL_PROFILE,
        trial_id="plain-harbor-test",
        run_id=uuid4(),
        task_id=uuid4(),
        agent_complete=True,
        runtime_status="completed",
        final_output={"agent_complete": True},
        command_count=1,
        denial_count=0,
        timeout_count=0,
        in_doubt_count=0,
        input_tokens=12,
        output_tokens=4,
        total_tokens=16,
        cost_usd=0.0,
        latency_ms=10,
        trace_consistent=True,
        started_at=now,
        completed_at=now,
    )


class PlainSequentialTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_execution_has_no_apply_patch_intercept_or_completion_gate(
        self,
    ) -> None:
        environment = FakeTerminalEnvironment(
            completed_result(
                return_code=127,
                stderr="apply_patch: command not found",
            )
        )
        capability = ScriptedTerminalTurnCapability(
            execute_draft(
                "apply_patch <<'PATCH'\nPATCH",
                call_key="plain-apply-patch",
            ),
            complete_draft("plain agent stopped"),
            usage=InferenceUsage(input_tokens=5, output_tokens=2, total_tokens=7),
        )
        with tempfile.TemporaryDirectory() as directory:
            application = build_plain_sequential_application(
                trial_id="plain-direct",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
            )
            artifacts = await application.run("Create the requested artifact.")

            self.assertEqual(len(environment.calls), 1)
            self.assertIn("apply_patch", environment.calls[0].command)
            self.assertTrue(artifacts.summary.agent_complete)
            self.assertEqual(artifacts.summary.denial_count, 0)
            self.assertEqual(artifacts.summary.command_count, 1)
            self.assertEqual(artifacts.summary.total_tokens, 14)
            self.assertEqual(artifacts.summary.profile, PLAIN_SEQUENTIAL_PROFILE)
            self.assertTrue((Path(directory) / "plain-summary.json").is_file())
            self.assertTrue((Path(directory) / "plain-transcript.jsonl").is_file())
            self.assertFalse((Path(directory) / "aar-runtime.sqlite3").exists())

    async def test_successful_verification_does_not_lock_later_work(self) -> None:
        environment = FakeTerminalEnvironment(
            completed_result(stdout="verified"),
            completed_result(stdout="modified afterwards"),
        )
        capability = ScriptedTerminalTurnCapability(
            verify_draft("true", call_key="plain-verify"),
            execute_draft("printf changed > /app/output", call_key="plain-work"),
            complete_draft(),
        )
        with tempfile.TemporaryDirectory() as directory:
            artifacts = await build_plain_sequential_application(
                trial_id="plain-no-lock",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
            ).run("Produce /app/output.")

        self.assertTrue(artifacts.summary.agent_complete)
        self.assertEqual(len(environment.calls), 2)
        self.assertIn("changed", environment.calls[1].command)

    async def test_in_doubt_command_can_be_replayed_without_fingerprint_guard(
        self,
    ) -> None:
        now = utc_now()
        uncertain = TerminalExecResult(
            stderr="transport timeout",
            started_at=now,
            completed_at=now,
            duration_ms=0,
            execution_state=TerminalExecutionState.IN_DOUBT,
            timed_out=True,
            transport_failed=True,
        )
        environment = FakeTerminalEnvironment(
            uncertain,
            completed_result(stdout="second attempt ran"),
        )
        capability = ScriptedTerminalTurnCapability(
            execute_draft("touch /app/output", call_key="same-call"),
            execute_draft("touch /app/output", call_key="same-call"),
            complete_draft(),
        )
        with tempfile.TemporaryDirectory() as directory:
            artifacts = await build_plain_sequential_application(
                trial_id="plain-replay",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
            ).run("Create /app/output.")

        self.assertTrue(artifacts.summary.agent_complete)
        self.assertEqual(len(environment.calls), 2)
        self.assertEqual(artifacts.summary.in_doubt_count, 1)

    async def test_timeout_above_shared_limit_fails_without_correction_turn(
        self,
    ) -> None:
        environment = FakeTerminalEnvironment()
        capability = ScriptedTerminalTurnCapability(
            execute_draft("sleep 1", timeout_sec=301),
            complete_draft(),
        )
        with tempfile.TemporaryDirectory() as directory:
            artifacts = await build_plain_sequential_application(
                trial_id="plain-timeout-limit",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(max_timeout_sec=300),
            ).run("Run the command.")

        self.assertFalse(artifacts.summary.agent_complete)
        self.assertEqual(artifacts.summary.runtime_status, "invalid_proposal")
        self.assertEqual(len(capability.requests), 1)
        self.assertEqual(environment.calls, [])

    async def test_inference_budget_exhaustion_is_a_controlled_stop(self) -> None:
        environment = FakeTerminalEnvironment()
        capability = _BudgetExhaustedCapability()
        with tempfile.TemporaryDirectory() as directory:
            artifacts = await build_plain_sequential_application(
                trial_id="plain-inference-budget",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
            ).run("Complete the task.")

            transcript = (
                Path(directory) / "plain-transcript.jsonl"
            ).read_text(encoding="utf-8")
            self.assertTrue(
                (Path(directory) / "plain-summary.json").is_file()
            )

        self.assertFalse(artifacts.summary.agent_complete)
        self.assertEqual(
            artifacts.summary.runtime_status,
            "inference_budget_exhausted",
        )
        self.assertEqual(environment.calls, [])
        self.assertEqual(len(capability.requests), 1)
        self.assertIn("plain.inference.budget_exhausted", transcript)

    async def test_shared_request_contract_disables_runtime_recovery_flags(
        self,
    ) -> None:
        environment = FakeTerminalEnvironment(completed_result())
        capability = ScriptedTerminalTurnCapability(
            execute_draft(
                "pwd",
                call_key="inspect",
                command_role=TerminalCommandRole.INSPECT,
            ),
            complete_draft(),
        )
        with tempfile.TemporaryDirectory() as directory:
            await build_plain_sequential_application(
                trial_id="plain-request",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
            ).run("Inspect and solve.")

        request = capability.requests[1]
        self.assertEqual(request.profile, "AAR Terminal Sequential Profile")
        self.assertFalse(request.recovery_mode)
        self.assertFalse(request.repair_mode)
        self.assertFalse(request.verification_due)
        self.assertFalse(request.tool_capabilities.apply_patch)
        self.assertTrue(
            any(
                "apply_patch is not installed" in semantic
                for semantic in request.execution_semantics
            )
        )

    async def test_harbor_agent_fixes_both_reasoning_controls_to_high(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_application = _FakePlainApplication(
                PlainRunArtifacts(summary=_summary())
            )
            captured = {}

            def factory(**kwargs):  # type: ignore[no-untyped-def]
                captured.update(kwargs)
                return fake_application

            agent = PlainSequentialHarborAgent(
                logs_dir=Path(directory),
                model_name="codex-cli/gpt-5.6-luna",
                reasoning_effort="high",
                application_factory=factory,
            )
            context = AgentContext()
            await agent.run(
                "solve",
                SimpleNamespace(context_id=uuid4()),  # type: ignore[arg-type]
                context,
            )

        model_config = captured["model_config"]
        self.assertEqual(model_config.codex_reasoning_effort, ReasoningEffort.HIGH)
        self.assertEqual(
            model_config.deepseek_reasoning_effort,
            ReasoningEffort.HIGH,
        )
        self.assertIn("plain_sequential", context.metadata)
        self.assertTrue(fake_application.closed)

    def test_harbor_agent_rejects_non_high_reasoning_effort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "fixes reasoning_effort=high"):
                PlainSequentialHarborAgent(
                    logs_dir=Path(directory),
                    model_name="codex-cli/gpt-5.6-luna",
                    reasoning_effort="max",
                )


if __name__ == "__main__":
    unittest.main()

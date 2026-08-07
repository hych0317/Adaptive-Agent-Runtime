from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from uuid import UUID, uuid4

from adaptive_agent_runtime.core import (
    ActionRequest,
    AgentState,
    AgentTask,
    RunStatus,
)

from applications.terminal_bench.composition import build_terminal_application
from applications.terminal_bench.contracts import TerminalExecutionError
from applications.terminal_bench.models import (
    TERMINAL_COMMAND_ACTION,
    TerminalCommandIntent,
    TerminalExecutionPolicy,
    TerminalExecutionState,
)
from tests.terminal_bench.fakes import (
    FakeTerminalEnvironment,
    ScriptedTerminalTurnCapability,
    complete_draft,
    completed_result,
    execute_draft,
    verify_draft,
)


class TerminalSequentialProfileTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_command_uses_tool_invocation_decision_lifecycle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = FakeTerminalEnvironment(
                completed_result(stdout="ok"),
                completed_result(stdout="verified"),
            )
            app = build_terminal_application(
                trial_id="trial-lifecycle",
                logs_dir=directory,
                environment=environment,
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("printf ok", cwd="/app", env={}),
                    verify_draft("test -n ok"),
                    complete_draft(),
                ),
            )
            try:
                artifacts = await app.run("finish the task")
                proofs = app.persistence.decision_records.find_completed_decisions(
                    artifacts.summary.run_id,
                    "tool.invocation",
                )
                self.assertEqual(len(proofs), 2)
                proof = proofs[0]
                self.assertEqual(proof.result_status, "applied")
                self.assertIsNotNone(proof.authorization_id)
                self.assertIsNotNone(proof.effect_fingerprint)
                transitions = app.persistence.decision_records.transitions_for(
                    proof.request_id
                )
                stages = {item.to_stage for item in transitions}
                self.assertIn("validated", stages)
                self.assertIn("authorized", stages)
                self.assertIn("applying", stages)
                self.assertIn("effect_committed", stages)
                self.assertIn("completed", stages)
            finally:
                app.close()

    async def test_governed_effect_is_same_effect_executed_by_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = FakeTerminalEnvironment(
                completed_result(stdout="ok"),
                completed_result(stdout="verified"),
            )
            app = build_terminal_application(
                trial_id="trial-effect",
                logs_dir=directory,
                environment=environment,
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft(
                        "python -V",
                        cwd="/app",
                        env={"MODE": "test"},
                        timeout_sec=41,
                    ),
                    verify_draft("python -V", call_key="verification-1"),
                    complete_draft(),
                ),
            )
            try:
                artifacts = await app.run("inspect Python")
                proof = app.persistence.decision_records.find_completed_decisions(
                    artifacts.summary.run_id,
                    "tool.invocation",
                )[0]
                assert proof.effect_fingerprint is not None
                effect = app.persistence.decision_records.load_effect(
                    proof.effect_fingerprint
                )
                assert effect is not None
                arguments = effect["payload"]["invocation"]["arguments"]
                call = environment.calls[0]
                self.assertEqual(arguments["command"], call.command)
                self.assertEqual(arguments["cwd"], call.cwd)
                self.assertEqual(arguments["env"], call.env)
                self.assertEqual(arguments["timeout_sec"], call.timeout_sec)
                self.assertEqual(arguments["trial_id"], "trial-effect")
                self.assertEqual(arguments["command_role"], "work")
                verify_proof = app.persistence.decision_records.find_completed_decisions(
                    artifacts.summary.run_id,
                    "tool.invocation",
                )[1]
                assert verify_proof.effect_fingerprint is not None
                verify_effect = app.persistence.decision_records.load_effect(
                    verify_proof.effect_fingerprint
                )
                assert verify_effect is not None
                self.assertEqual(
                    verify_effect["payload"]["invocation"]["arguments"][
                        "command_role"
                    ],
                    "verify",
                )
                self.assertNotEqual(
                    proof.effect_fingerprint,
                    verify_proof.effect_fingerprint,
                )
            finally:
                app.close()

    async def test_resume_reuses_original_command_effect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = FakeTerminalEnvironment(completed_result(stdout="once"))
            app = build_terminal_application(
                trial_id="trial-resume",
                logs_dir=directory,
                environment=environment,
                proposal_capability=ScriptedTerminalTurnCapability(complete_draft()),
            )
            intent = TerminalCommandIntent(
                trial_id="trial-resume",
                call_key="same-action",
                command="touch /app/once",
                cwd="/app",
                env={},
                timeout_sec=30,
            )
            action = ActionRequest(
                action_id=uuid4(),
                name=TERMINAL_COMMAND_ACTION,
                arguments=intent.model_dump(mode="json"),
            )
            state = AgentState(
                run_id=uuid4(),
                task=AgentTask(description="create once"),
                status=RunStatus.RUNNING,
            )
            executor = app.runtime._executor  # white-box: decision resume boundary
            try:
                first = await executor.execute(action, state)
                second = await executor.execute(action, state)
                self.assertEqual(first.output, second.output)
                self.assertEqual(len(environment.calls), 1)
                first_decision = first.metadata["terminal_decision"]
                second_decision = second.metadata["terminal_decision"]
                self.assertEqual(first_decision, second_decision)
            finally:
                app.close()

    async def test_nonzero_return_code_is_normal_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = FakeTerminalEnvironment(
                completed_result(return_code=7, stderr="not found"),
                completed_result(stdout="handled"),
            )
            app = build_terminal_application(
                trial_id="trial-nonzero",
                logs_dir=directory,
                environment=environment,
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("test -f /app/missing"),
                    verify_draft("test ! -f /app/missing"),
                    complete_draft("handled nonzero"),
                ),
            )
            try:
                artifacts = await app.run("inspect a missing file")
                record = app.journal.records()[0]
                self.assertEqual(record.result.execution_state, TerminalExecutionState.COMPLETED)
                self.assertEqual(record.result.return_code, 7)
                self.assertTrue(artifacts.summary.agent_complete)
            finally:
                app.close()

    async def test_failed_to_start_is_provider_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = FakeTerminalEnvironment(
                TerminalExecutionError(
                    "executable unavailable",
                    command_started=False,
                )
            )
            app = build_terminal_application(
                trial_id="trial-no-start",
                logs_dir=directory,
                environment=environment,
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("missing-program"),
                    complete_draft("observed provider failure"),
                ),
            )
            try:
                await app.run("run missing program")
                result = app.journal.records()[0].result
                self.assertEqual(result.execution_state, TerminalExecutionState.FAILED_TO_START)
                self.assertTrue(result.transport_failed)
            finally:
                app.close()

    async def test_timeout_with_unknown_process_state_is_in_doubt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = FakeTerminalEnvironment(
                TerminalExecutionError(
                    "deadline elapsed",
                    command_started=True,
                    timed_out=True,
                )
            )
            app = build_terminal_application(
                trial_id="trial-timeout",
                logs_dir=directory,
                environment=environment,
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("long-job", timeout_sec=2),
                    complete_draft("stopped after uncertainty"),
                ),
            )
            try:
                await app.run("run a bounded job")
                result = app.journal.records()[0].result
                self.assertEqual(result.execution_state, TerminalExecutionState.IN_DOUBT)
                self.assertTrue(result.timed_out)
            finally:
                app.close()

    async def test_in_doubt_command_is_not_automatically_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = FakeTerminalEnvironment(
                TerminalExecutionError(
                    "connection lost",
                    command_started=None,
                )
            )
            app = build_terminal_application(
                trial_id="trial-no-replay",
                logs_dir=directory,
                environment=environment,
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("mutate-state", call_key="attempt-1"),
                    execute_draft("mutate-state", call_key="attempt-2"),
                ),
            )
            try:
                artifacts = await app.run("perform one mutation")
                self.assertFalse(artifacts.runtime_result.succeeded)
                self.assertIn("IN_DOUBT", artifacts.runtime_result.final_state.error or "")
                self.assertEqual(len(environment.calls), 1)
            finally:
                app.close()

    async def test_cwd_and_environment_are_explicit_between_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = FakeTerminalEnvironment(
                completed_result(stdout="one"),
                completed_result(stdout="two"),
                completed_result(stdout="verified"),
            )
            capability = ScriptedTerminalTurnCapability(
                execute_draft("first", cwd="/app/work", env={"MODE": "test"}),
                execute_draft("second", call_key="command-2"),
                verify_draft(call_key="verification-1"),
                complete_draft(),
            )
            app = build_terminal_application(
                trial_id="trial-explicit",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
            )
            try:
                await app.run("run two commands")
                self.assertEqual(environment.calls[1].cwd, "/app/work")
                self.assertEqual(environment.calls[1].env, {"MODE": "test"})
                self.assertEqual(
                    capability.requests[1].session.current_cwd,
                    "/app/work",
                )
            finally:
                app.close()

    async def test_shell_local_state_is_not_assumed_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = FakeTerminalEnvironment(
                completed_result(),
                completed_result(),
                completed_result(),
            )
            app = build_terminal_application(
                trial_id="trial-shell-state",
                logs_dir=directory,
                environment=environment,
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft(
                        "cd /tmp; export FOO=bar",
                        cwd="/app",
                        env={},
                    ),
                    execute_draft("pwd; env", call_key="command-2"),
                    verify_draft(call_key="verification-1"),
                    complete_draft(),
                ),
            )
            try:
                await app.run("test shell state")
                self.assertEqual(environment.calls[1].cwd, "/app")
                self.assertEqual(environment.calls[1].env, {})
            finally:
                app.close()

    async def test_premature_complete_is_rejected_and_replanned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-artifact"),
                complete_draft("premature"),
                verify_draft("assert-artifact"),
                complete_draft(),
            )
            app = build_terminal_application(
                trial_id="trial-complete-gate",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(),
                ),
                proposal_capability=capability,
            )
            try:
                artifacts = await app.run("create and validate an artifact")
                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertTrue(artifacts.summary.agent_complete)
                self.assertIn(
                    "last committed command is not marked verify",
                    capability.requests[2].session.completion_blocker or "",
                )
            finally:
                app.close()

    async def test_successful_verification_allows_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="trial-verified-complete",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(stdout="9 dates"),
                ),
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("create-artifact"),
                    verify_draft("assert-artifact"),
                    complete_draft(),
                ),
                policy=TerminalExecutionPolicy(max_completion_rejections=0),
            )
            try:
                artifacts = await app.run("create and validate an artifact")
                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertTrue(artifacts.summary.agent_complete)
            finally:
                app.close()

    async def test_failed_verification_does_not_allow_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="trial-failed-verification",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(return_code=1, stderr="assertion failed"),
                ),
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("create-artifact"),
                    verify_draft("assert-artifact"),
                    complete_draft(),
                ),
                policy=TerminalExecutionPolicy(max_completion_rejections=0),
            )
            try:
                artifacts = await app.run("create and validate an artifact")
                self.assertFalse(artifacts.runtime_result.succeeded)
                self.assertFalse(artifacts.summary.agent_complete)
                self.assertIn(
                    "verification command returned a non-zero status",
                    artifacts.runtime_result.final_state.error or "",
                )
            finally:
                app.close()

    async def test_work_after_verification_invalidates_completion_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="trial-stale-verification",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(),
                    completed_result(),
                ),
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("create-artifact"),
                    verify_draft("assert-artifact"),
                    execute_draft("change-artifact", call_key="command-2"),
                    complete_draft(),
                ),
                policy=TerminalExecutionPolicy(max_completion_rejections=0),
            )
            try:
                artifacts = await app.run("create and validate an artifact")
                self.assertFalse(artifacts.runtime_result.succeeded)
                self.assertFalse(artifacts.summary.agent_complete)
                self.assertIn(
                    "last committed command is not marked verify",
                    artifacts.runtime_result.final_state.error or "",
                )
            finally:
                app.close()


if __name__ == "__main__":
    unittest.main()

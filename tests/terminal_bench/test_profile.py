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
from adaptive_agent_runtime.llm import InferenceExecutionBudgetError, InferenceUsage

from applications.terminal_bench.composition import build_terminal_application
from applications.terminal_bench.contracts import TerminalExecutionError
from applications.terminal_bench.models import (
    TERMINAL_COMMAND_ACTION,
    TerminalCommandIntent,
    TerminalCommandRole,
    TerminalExecutionPolicy,
    TerminalExecutionState,
    TerminalTurnDraft,
    TerminalTurnProposal,
    TerminalTurnRequest,
    TerminalVerificationContract,
)
from applications.terminal_bench.planner import _identifies_official_test_source
from tests.terminal_bench.fakes import (
    FakeTerminalEnvironment,
    ScriptedTerminalTurnCapability,
    complete_draft,
    completed_result,
    execute_draft,
    verify_draft,
)


class TimeoutThenScriptedCapability(ScriptedTerminalTurnCapability):
    def __init__(self, *drafts: TerminalTurnDraft) -> None:
        super().__init__(*drafts)
        self._timed_out = False

    async def propose(self, request: TerminalTurnRequest) -> TerminalTurnProposal:
        self.requests.append(request)
        if not self._timed_out:
            self._timed_out = True
            raise InferenceExecutionBudgetError("elapsed-time limit 360 seconds")
        self.requests.pop()
        return await super().propose(request)


class AlwaysTimeoutCapability:
    module_id = "test.terminal_turn.always_timeout"

    def __init__(self) -> None:
        self.requests: list[TerminalTurnRequest] = []

    async def propose(self, request: TerminalTurnRequest) -> TerminalTurnProposal:
        self.requests.append(request)
        raise InferenceExecutionBudgetError("elapsed-time limit 360 seconds")


class TerminalSequentialProfileTests(unittest.IsolatedAsyncioTestCase):
    def test_eval_py_is_recognized_as_an_official_test_source(self) -> None:
        self.assertTrue(_identifies_official_test_source("/app/eval.py"))
        self.assertTrue(
            _identifies_official_test_source("python3 /app/eval.py")
        )
        self.assertFalse(
            _identifies_official_test_source("/app/evaluate.py")
        )

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

    async def test_all_call_keys_remain_visible_when_history_is_bounded(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("printf one", call_key="work-1"),
                execute_draft("printf two", call_key="work-2"),
                verify_draft("true", call_key="verification-1"),
            )
            app = build_terminal_application(
                trial_id="trial-used-call-keys",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(stdout="one"),
                    completed_result(stdout="two"),
                    completed_result(stdout="verified"),
                ),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(max_context_records=1),
            )
            try:
                await app.run("run two commands and verify")
                last_request = capability.requests[-1]
                self.assertEqual(
                    last_request.used_call_keys,
                    ("work-1", "work-2"),
                )
                self.assertEqual(
                    tuple(item.call_key for item in last_request.recent_history),
                    ("work-2",),
                )
                self.assertEqual(len(capability.requests), 3)
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

    async def test_no_progress_termination_commits_last_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="trial-no-progress-termination",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(return_code=1, stderr="unchanged")
                ),
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("false")
                ),
                policy=TerminalExecutionPolicy(
                    max_no_progress_steps=1,
                    max_no_progress_seconds=None,
                ),
            )
            try:
                artifacts = await app.run("stop after no progress")

                self.assertEqual(
                    artifacts.runtime_result.final_state.status,
                    RunStatus.TERMINATED,
                )
                self.assertEqual(artifacts.summary.command_count, 1)
                self.assertTrue(artifacts.summary.trace_consistent)
                self.assertIsNone(app.journal.pending())
                self.assertEqual(len(app.journal.records()), 1)
            finally:
                app.close()

    async def test_before_action_deadline_abandons_pending_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = FakeTerminalEnvironment()
            app = build_terminal_application(
                trial_id="trial-action-abandoned",
                logs_dir=directory,
                environment=environment,
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("sleep 1", timeout_sec=10)
                ),
                policy=TerminalExecutionPolicy(
                    default_timeout_sec=10,
                    max_timeout_sec=10,
                    max_wall_clock_seconds=5.0,
                    cleanup_grace_seconds=1.0,
                    max_no_progress_steps=None,
                    max_no_progress_seconds=None,
                ),
            )
            try:
                artifacts = await app.run("run a command only when admitted")

                self.assertEqual(
                    artifacts.runtime_result.final_state.status,
                    RunStatus.TERMINATED,
                )
                self.assertEqual(environment.calls, [])
                self.assertIsNone(app.journal.pending())
                self.assertTrue(artifacts.summary.trace_consistent)
                transcript = Path(directory, "aar-transcript.jsonl").read_text(
                    encoding="utf-8"
                )
                self.assertIn('"kind": "pending.abandoned"', transcript)
            finally:
                app.close()

    async def test_repeated_inspection_counts_as_no_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_terminal_application(
                trial_id="trial-repeated-inspection",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(stdout="same diagnostic"),
                    completed_result(stdout="same diagnostic"),
                ),
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft(
                        "inspect-state",
                        call_key="inspection-1",
                        command_role=TerminalCommandRole.INSPECT,
                    ),
                    execute_draft(
                        "inspect-state",
                        call_key="inspection-2",
                        command_role=TerminalCommandRole.INSPECT,
                    ),
                ),
                policy=TerminalExecutionPolicy(
                    max_no_progress_steps=1,
                    max_no_progress_seconds=None,
                ),
            )
            try:
                artifacts = await app.run("diagnose before making a repair")

                self.assertEqual(
                    artifacts.runtime_result.final_state.status,
                    RunStatus.TERMINATED,
                )
                self.assertEqual(artifacts.summary.command_count, 2)
                self.assertTrue(artifacts.summary.trace_consistent)
                self.assertEqual(len(app.journal.records()), 2)
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
                policy=TerminalExecutionPolicy(max_proposal_rejections=0),
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

    async def test_timeout_above_runtime_maximum_is_correctable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft(
                    "long-build",
                    call_key="invalid-timeout",
                    timeout_sec=301,
                ),
                execute_draft(
                    "bounded-build",
                    call_key="bounded-timeout",
                    timeout_sec=300,
                ),
                verify_draft("test -f /app/result"),
                complete_draft(),
            )
            environment = FakeTerminalEnvironment(
                completed_result(stdout="built"),
                completed_result(stdout="verified"),
            )
            app = build_terminal_application(
                trial_id="trial-timeout-correction",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(max_proposal_rejections=1),
            )
            try:
                artifacts = await app.run("build with a bounded timeout")
                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertEqual(len(environment.calls), 2)
                self.assertEqual(environment.calls[0].command, "bounded-build")
                rejection = capability.requests[1].session.last_proposal_rejection
                self.assertIsNotNone(rejection)
                assert rejection is not None
                self.assertEqual(rejection.code, "terminal.timeout.above_maximum")
                self.assertEqual(rejection.rejected_value, "301")
            finally:
                app.close()

    async def test_unavailable_apply_patch_is_rejected_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft(
                    "apply_patch <<'PATCH'\n*** Begin Patch\n*** End Patch\nPATCH",
                    call_key="unavailable-editor",
                ),
                execute_draft(
                    "python3 -c 'open(\"/app/result\", \"w\").write(\"ok\")'",
                    call_key="portable-editor",
                ),
                verify_draft("test \"$(cat /app/result)\" = ok"),
                complete_draft(),
            )
            environment = FakeTerminalEnvironment(
                completed_result(),
                completed_result(),
            )
            app = build_terminal_application(
                trial_id="trial-tool-capability",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(max_proposal_rejections=1),
            )
            try:
                artifacts = await app.run("edit a task file portably")
                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertEqual(len(environment.calls), 2)
                self.assertNotIn("apply_patch", environment.calls[0].command)
                rejection = capability.requests[1].session.last_proposal_rejection
                self.assertIsNotNone(rejection)
                assert rejection is not None
                self.assertEqual(rejection.code, "terminal.tool.unavailable")
                self.assertEqual(rejection.draft.environment_keys, ())
            finally:
                app.close()

    async def test_mutating_git_verification_is_rejected_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-artifact"),
                verify_draft(
                    "git branch verifier-created-ref",
                    call_key="polluting-verification",
                ),
                verify_draft(
                    "git show-ref --verify refs/heads/expected",
                    call_key="read-only-verification",
                ),
                complete_draft(),
            )
            environment = FakeTerminalEnvironment(
                completed_result(),
                completed_result(),
            )
            app = build_terminal_application(
                trial_id="trial-verification-purity",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(max_proposal_rejections=1),
            )
            try:
                artifacts = await app.run("create and verify repository state")
                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertEqual(len(environment.calls), 2)
                self.assertEqual(
                    environment.calls[1].command,
                    "git show-ref --verify refs/heads/expected",
                )
                rejection = capability.requests[2].session.last_proposal_rejection
                self.assertIsNotNone(rejection)
                assert rejection is not None
                self.assertEqual(
                    rejection.code,
                    "terminal.verification.persistent_mutation",
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
                ),
                policy=TerminalExecutionPolicy(max_completion_rejections=0),
            )
            try:
                artifacts = await app.run("create and validate an artifact")
                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertTrue(artifacts.summary.agent_complete)
                self.assertIn(
                    "Runtime finalized",
                    str(artifacts.runtime_result.final_state.output),
                )
            finally:
                app.close()

    async def test_successful_verification_completes_at_exact_token_budget(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-artifact"),
                verify_draft("assert-artifact"),
                usage=InferenceUsage(total_tokens=1),
            )
            app = build_terminal_application(
                trial_id="trial-exact-token-budget",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(),
                ),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(max_total_tokens=2),
            )
            try:
                artifacts = await app.run("create and validate an artifact")
                self.assertTrue(
                    artifacts.runtime_result.succeeded,
                    artifacts.runtime_result.final_state.model_dump_json(indent=2),
                )
                self.assertTrue(artifacts.summary.agent_complete)
                self.assertEqual(artifacts.summary.total_tokens, 2)
                self.assertEqual(len(capability.requests), 2)
            finally:
                app.close()

    async def test_exact_token_budget_blocks_another_model_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-artifact"),
                usage=InferenceUsage(total_tokens=1),
            )
            environment = FakeTerminalEnvironment(completed_result())
            app = build_terminal_application(
                trial_id="trial-exact-token-budget-without-verification",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(max_total_tokens=1),
            )
            try:
                artifacts = await app.run("create an artifact")
                self.assertFalse(artifacts.runtime_result.succeeded)
                self.assertEqual(
                    artifacts.runtime_result.final_state.error,
                    "terminal model token budget exhausted",
                )
                self.assertEqual(len(capability.requests), 1)
                self.assertEqual(len(environment.calls), 1)
            finally:
                app.close()

    async def test_over_budget_model_turn_does_not_execute_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-artifact"),
                usage=InferenceUsage(total_tokens=2),
            )
            environment = FakeTerminalEnvironment(completed_result())
            app = build_terminal_application(
                trial_id="trial-over-token-budget",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(max_total_tokens=1),
            )
            try:
                artifacts = await app.run("create an artifact")
                self.assertFalse(artifacts.runtime_result.succeeded)
                self.assertEqual(
                    artifacts.runtime_result.final_state.error,
                    "terminal model token budget exhausted",
                )
                self.assertEqual(len(capability.requests), 1)
                self.assertEqual(len(environment.calls), 0)
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

    async def test_multiple_verification_repair_cycles_are_allowed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-artifact", call_key="work-1"),
                verify_draft("assert-artifact", call_key="verification-1"),
                execute_draft("repair-artifact-1", call_key="work-2"),
                verify_draft("assert-artifact", call_key="verification-2"),
                execute_draft("repair-artifact-2", call_key="work-3"),
                verify_draft("assert-artifact", call_key="verification-3"),
            )
            app = build_terminal_application(
                trial_id="trial-multiple-verification-repairs",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(return_code=1, stderr="first failure"),
                    completed_result(),
                    completed_result(return_code=1, stderr="second failure"),
                    completed_result(),
                    completed_result(stdout="verification passed"),
                ),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(
                    max_no_progress_seconds=None,
                ),
            )
            try:
                artifacts = await app.run("create and validate an artifact")
                session = app.journal.snapshot()

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertTrue(artifacts.summary.agent_complete)
                self.assertEqual(artifacts.summary.command_count, 6)
                self.assertEqual(session.failed_verification_attempts, 2)
                self.assertEqual(session.verification_corrections, 2)
                self.assertEqual(len(capability.requests), 6)
                self.assertIsNotNone(session.verified_checkpoint)
                self.assertTrue(artifacts.summary.trace_consistent)
            finally:
                app.close()

    async def test_successful_verification_auto_completes_and_preserves_checkpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-artifact"),
                verify_draft("assert-artifact"),
                execute_draft("change-artifact", call_key="command-2"),
                complete_draft(),
            )
            app = build_terminal_application(
                trial_id="trial-stale-verification",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(),
                ),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(max_completion_rejections=0),
            )
            try:
                artifacts = await app.run("create and validate an artifact")
                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertTrue(artifacts.summary.agent_complete)
                self.assertEqual(len(app.journal.records()), 2)
                checkpoint = app.journal.snapshot().verified_checkpoint
                self.assertIsNotNone(checkpoint)
                self.assertEqual(len(capability.requests), 2)
                self.assertIsNone(
                    app.journal.snapshot().last_proposal_rejection
                )
            finally:
                app.close()

    async def test_verification_must_cover_runtime_requirement_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verification = TerminalVerificationContract(
                evidence_kind="independent_check",
                evidence_sources=("task-local check",),
                artifact_paths=("/app/hello.html",),
                requirement_coverage=("req-001",),
                coverage_dimensions=(
                    "artifact",
                    "format",
                    "semantic",
                    "end_to_end",
                ),
                validation_methods=("HTTP request plus content assertion",),
            )
            environment = FakeTerminalEnvironment(completed_result())
            app = build_terminal_application(
                trial_id="trial-requirement-coverage",
                logs_dir=directory,
                environment=environment,
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("touch /app/hello.html"),
                    verify_draft(
                        "test -f /app/hello.html",
                        verification=verification,
                    ),
                ),
                policy=TerminalExecutionPolicy(
                    max_proposal_rejections=0,
                    max_no_progress_seconds=None,
                ),
            )
            try:
                artifacts = await app.run(
                    "Create the artifact. Serve hello.html over HTTP."
                )

                self.assertEqual(
                    artifacts.runtime_result.final_state.status,
                    RunStatus.FAILED,
                )
                self.assertEqual(len(environment.calls), 1)
                self.assertIn(
                    "verification requirement coverage is incomplete",
                    artifacts.runtime_result.final_state.error or "",
                )
            finally:
                app.close()

    async def test_inference_timeout_retries_once_in_delivery_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = TimeoutThenScriptedCapability(
                execute_draft("create-artifact"),
                verify_draft("test -e /app"),
            )
            app = build_terminal_application(
                trial_id="trial-inference-delivery-retry",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(),
                ),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(max_no_progress_seconds=None),
            )
            try:
                artifacts = await app.run("create and validate an artifact")
                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertFalse(capability.requests[0].delivery_mode)
                self.assertTrue(capability.requests[1].delivery_mode)
                self.assertTrue(capability.requests[1].recovery_mode)
                self.assertEqual(len(app.journal.records()), 2)
            finally:
                app.close()

    async def test_repeated_inference_timeout_is_controlled_termination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = AlwaysTimeoutCapability()
            app = build_terminal_application(
                trial_id="trial-inference-budget-termination",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(max_no_progress_seconds=None),
            )
            try:
                artifacts = await app.run("create an artifact")
                state = artifacts.runtime_result.final_state
                self.assertEqual(state.status, RunStatus.TERMINATED)
                self.assertIsNone(state.error)
                self.assertIsNotNone(state.termination)
                assert state.termination is not None
                self.assertEqual(
                    state.termination.primary_reason.value,
                    "active_execution_budget",
                )
                self.assertEqual(len(capability.requests), 2)
                self.assertFalse(capability.requests[0].delivery_mode)
                self.assertTrue(capability.requests[1].delivery_mode)
                self.assertTrue(artifacts.summary.trace_consistent)
            finally:
                app.close()

    async def test_consecutive_unique_inspections_enter_recovery_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = FakeTerminalEnvironment(
                completed_result(stdout="first"),
                completed_result(stdout="second"),
                completed_result(stdout="third"),
            )
            capability = ScriptedTerminalTurnCapability(
                execute_draft(
                    "inspect-1",
                    call_key="inspect-1",
                    command_role=TerminalCommandRole.INSPECT,
                ),
                execute_draft(
                    "inspect-2",
                    call_key="inspect-2",
                    command_role=TerminalCommandRole.INSPECT,
                ),
                execute_draft(
                    "inspect-3",
                    call_key="inspect-3",
                    command_role=TerminalCommandRole.INSPECT,
                ),
                execute_draft(
                    "inspect-4",
                    call_key="inspect-4",
                    command_role=TerminalCommandRole.INSPECT,
                ),
            )
            app = build_terminal_application(
                trial_id="trial-inspection-recovery",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(
                    max_consecutive_inspections=3,
                    max_proposal_rejections=0,
                    max_no_progress_seconds=None,
                ),
            )
            try:
                artifacts = await app.run("Inspect, then repair the artifact.")

                self.assertEqual(
                    artifacts.runtime_result.final_state.status,
                    RunStatus.FAILED,
                )
                self.assertEqual(len(environment.calls), 3)
                self.assertEqual(app.journal.snapshot().consecutive_inspections, 3)
                self.assertTrue(capability.requests[-1].recovery_mode)
            finally:
                app.close()

    async def test_total_inspections_enter_recovery_across_work_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            drafts: list[TerminalTurnDraft] = []
            results = []
            for index in range(1, 6):
                drafts.extend(
                    (
                        execute_draft(
                            f"inspect-{index}",
                            call_key=f"inspect-{index}",
                            command_role=TerminalCommandRole.INSPECT,
                        ),
                        execute_draft(
                            f"work-{index}",
                            call_key=f"work-{index}",
                        ),
                    )
                )
                results.extend((completed_result(), completed_result()))
            drafts.append(
                execute_draft(
                    "inspect-6",
                    call_key="inspect-6",
                    command_role=TerminalCommandRole.INSPECT,
                )
            )
            capability = ScriptedTerminalTurnCapability(*drafts)
            app = build_terminal_application(
                trial_id="trial-total-inspection-recovery",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(*results),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(
                    max_consecutive_inspections=3,
                    max_total_inspections=5,
                    max_proposal_rejections=0,
                    max_no_progress_steps=None,
                    max_no_progress_seconds=None,
                ),
            )
            try:
                artifacts = await app.run("Inspect incrementally, then repair.")
                self.assertEqual(
                    artifacts.runtime_result.final_state.status,
                    RunStatus.FAILED,
                )
                self.assertEqual(len(app.journal.records()), 10)
                session = app.journal.snapshot()
                self.assertEqual(session.inspection_commands, 5)
                self.assertEqual(session.consecutive_inspections, 0)
                self.assertTrue(capability.requests[-1].recovery_mode)
            finally:
                app.close()


if __name__ == "__main__":
    unittest.main()

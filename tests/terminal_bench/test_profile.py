from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4

from adaptive_agent_runtime.core import (
    ActionRequest,
    AgentState,
    AgentTask,
    RunStatus,
)
from adaptive_agent_runtime.llm import InferenceExecutionBudgetError, InferenceUsage
from adaptive_agent_runtime.tool_ecosystem.invocation_decision import (
    _validate_arguments,
)

from applications.terminal_bench.composition import build_terminal_application
from applications.terminal_bench.contracts import TerminalExecutionError
from applications.terminal_bench.deadline import terra_high_deadline_budget_profile
from applications.terminal_bench.models import (
    TERMINAL_COMMAND_ACTION,
    TerminalCommandIntent,
    TerminalCommandRole,
    TerminalCompletionDisposition,
    TerminalExecutionPolicy,
    TerminalExecutionState,
    TerminalSessionSnapshot,
    TerminalReconciliationState,
    TerminalTimeoutCapReason,
    TerminalTurnDraft,
    TerminalTurnProposal,
    TerminalTurnRequest,
    TerminalVerificationContract,
    utc_now,
)
from applications.terminal_bench.planner import (
    JsonlTerminalTrialJournal,
    _identifies_official_test_source,
)
from applications.terminal_bench.tools import terminal_provider_metadata
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
    def test_provider_schema_accepts_the_full_verification_contract(self) -> None:
        policy = TerminalExecutionPolicy()
        metadata = terminal_provider_metadata(policy)
        intent = TerminalCommandIntent(
            trial_id="trial-provider-verification-contract",
            call_key="verification-1",
            command="true",
            command_role=TerminalCommandRole.VERIFY,
            timeout_sec=policy.default_timeout_sec,
            verification=TerminalVerificationContract(
                evidence_kind="independent_check",
                evidence_sources=("independent oracle",),
                artifact_paths=("/app",),
                requirement_coverage=("req-001",),
                coverage_dimensions=(
                    "artifact",
                    "format",
                    "semantic",
                    "end_to_end",
                ),
                validation_methods=("fresh-process assertion",),
            ),
        )

        _validate_arguments(metadata, intent.tool_arguments())

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
                completed_result(stdout="handled"),
            )
            app = build_terminal_application(
                trial_id="trial-nonzero",
                logs_dir=directory,
                environment=environment,
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("test -f /app/missing"),
                    execute_draft(
                        "test ! -f /app/missing",
                        call_key="work-2",
                    ),
                    verify_draft(
                        "test ! -f /app/missing",
                        call_key="verification-1",
                    ),
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
                    completed_result(return_code=1, stderr="unchanged"),
                    completed_result(return_code=1, stderr="unchanged")
                ),
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("false", call_key="failure-1"),
                    execute_draft("false", call_key="failure-2"),
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
                self.assertEqual(artifacts.summary.command_count, 2)
                self.assertTrue(artifacts.summary.trace_consistent)
                self.assertIsNone(app.journal.pending())
                records = app.journal.records()
                self.assertEqual(len(records), 2)
                self.assertEqual(records[-1].result.return_code, 1)
                self.assertEqual(records[-1].result.stderr, "unchanged")
                self.assertEqual(
                    records[-1].action_id,
                    artifacts.runtime_result.final_state.last_observation.action_id,
                )
            finally:
                app.close()

    async def test_deadline_timeout_is_clamped_before_pending_save(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = FakeTerminalEnvironment(
                completed_result(stdout="corrected"),
                completed_result(stdout="verified"),
            )
            app = build_terminal_application(
                trial_id="trial-action-abandoned",
                logs_dir=directory,
                environment=environment,
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft(
                        "sleep 1",
                        call_key="too-wide",
                        timeout_sec=30,
                    ),
                    verify_draft("true", timeout_sec=1),
                ),
                policy=TerminalExecutionPolicy(
                    default_timeout_sec=10,
                    max_timeout_sec=30,
                    max_wall_clock_seconds=20.0,
                    cleanup_grace_seconds=1.0,
                    max_no_progress_steps=None,
                    max_no_progress_seconds=None,
                ),
            )
            try:
                artifacts = await app.run("run a command only when admitted")

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertEqual(len(environment.calls), 2)
                self.assertIsNone(app.journal.pending())
                self.assertTrue(artifacts.summary.trace_consistent)
                record = app.journal.records()[0]
                self.assertEqual(record.intent.requested_timeout_sec, 30)
                self.assertEqual(
                    record.intent.timeout_sec,
                    record.intent.advertised_timeout_cap_sec,
                )
                self.assertLess(record.intent.timeout_sec, 30)
                self.assertEqual(
                    record.intent.timeout_cap_reason,
                    TerminalTimeoutCapReason.ADVERTISED_CAP,
                )
                self.assertEqual(app.journal.snapshot().proposal_rejections, 0)
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
                    max_artifact_first_inspections=None,
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
                policy=TerminalExecutionPolicy(max_proposal_rejections=0),
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
                policy=TerminalExecutionPolicy(
                    max_proposal_rejections=0,
                    max_reconciliation_proposal_rejections=0,
                ),
            )
            try:
                artifacts = await app.run("perform one mutation")
                self.assertFalse(artifacts.runtime_result.succeeded)
                self.assertIn("IN_DOUBT", artifacts.runtime_result.final_state.error or "")
                self.assertEqual(len(environment.calls), 1)
            finally:
                app.close()

    async def test_in_doubt_requires_read_only_reconciliation_before_work(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = FakeTerminalEnvironment(
                TerminalExecutionError(
                    "connection lost",
                    command_started=True,
                ),
                completed_result(stdout="process stopped; artifacts known"),
                completed_result(stdout="verified"),
            )
            capability = ScriptedTerminalTurnCapability(
                execute_draft("mutate-state", call_key="attempt-1"),
                execute_draft(
                    "test ! -e /tmp/worker.pid && test -e /app",
                    call_key="reconcile-1",
                    command_role=TerminalCommandRole.INSPECT,
                ),
                verify_draft("true", call_key="verification-1"),
            )
            app = build_terminal_application(
                trial_id="trial-in-doubt-reconciliation",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(max_no_progress_seconds=None),
            )
            try:
                artifacts = await app.run("repair and verify the artifact")

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertEqual(len(environment.calls), 3)
                self.assertTrue(capability.requests[1].reconciliation_mode)
                self.assertFalse(capability.requests[2].reconciliation_mode)
                self.assertTrue(capability.requests[2].verification_due)
                self.assertFalse(
                    app.journal.snapshot().in_doubt_reconciliation_required
                )
                self.assertEqual(
                    app.journal.snapshot().reconciliation_state,
                    "stable_unverified",
                )
                self.assertIsNotNone(
                    app.journal.snapshot().latest_reconciliation_receipt
                )
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

    async def test_timeout_above_runtime_maximum_is_clamped_locally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft(
                    "long-build",
                    call_key="invalid-timeout",
                    timeout_sec=301,
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
                self.assertEqual(environment.calls[0].command, "long-build")
                record = app.journal.records()[0]
                self.assertEqual(record.intent.requested_timeout_sec, 301)
                self.assertEqual(record.intent.advertised_timeout_cap_sec, 300)
                self.assertEqual(record.intent.timeout_sec, 300)
                self.assertEqual(
                    record.intent.timeout_cap_reason,
                    TerminalTimeoutCapReason.ADVERTISED_CAP,
                )
                self.assertEqual(app.journal.snapshot().proposal_rejections, 0)
                self.assertEqual(len(capability.requests), 2)
            finally:
                app.close()

    async def test_verification_timeout_is_clamped_without_replanning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-artifact", call_key="work-1"),
                verify_draft(
                    "test -f /app/result",
                    call_key="verify-wide",
                    timeout_sec=300,
                ),
            )
            environment = FakeTerminalEnvironment(
                completed_result(stdout="built"),
                completed_result(stdout="verified"),
            )
            app = build_terminal_application(
                trial_id="trial-verify-timeout-clamp",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
            )
            try:
                artifacts = await app.run("build and verify an artifact")

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertEqual(len(environment.calls), 2)
                verify_record = app.journal.records()[1]
                self.assertEqual(verify_record.intent.requested_timeout_sec, 300)
                self.assertEqual(
                    verify_record.intent.advertised_timeout_cap_sec,
                    120,
                )
                self.assertEqual(verify_record.intent.timeout_sec, 120)
                self.assertEqual(
                    verify_record.intent.timeout_cap_reason,
                    TerminalTimeoutCapReason.ADVERTISED_CAP,
                )
                self.assertEqual(app.journal.snapshot().proposal_rejections, 0)
                self.assertEqual(len(capability.requests), 2)
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

    async def test_exact_token_budget_submits_known_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-artifact"),
                usage=InferenceUsage(total_tokens=1),
            )
            environment = FakeTerminalEnvironment(
                completed_result(),
                completed_result(stdout="benchmark completed"),
            )
            app = build_terminal_application(
                trial_id="trial-exact-token-budget-without-verification",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(max_total_tokens=1),
            )
            try:
                artifacts = await app.run("create an artifact")
                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertFalse(artifacts.summary.agent_complete)
                self.assertIs(
                    artifacts.summary.completion_disposition,
                    TerminalCompletionDisposition.SUBMITTED_UNVERIFIED,
                )
                self.assertEqual(len(capability.requests), 1)
                self.assertEqual(len(environment.calls), 1)
            finally:
                app.close()

    async def test_over_budget_model_turn_submits_without_command(self) -> None:
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
                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertFalse(artifacts.summary.agent_complete)
                self.assertIs(
                    artifacts.summary.completion_disposition,
                    TerminalCompletionDisposition.SUBMITTED_UNVERIFIED,
                )
                self.assertEqual(len(capability.requests), 1)
                self.assertEqual(len(environment.calls), 0)
            finally:
                app.close()

    async def test_failed_verification_complete_submits_without_success_claim(
        self,
    ) -> None:
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
                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertFalse(artifacts.summary.agent_complete)
                self.assertIs(
                    artifacts.summary.completion_disposition,
                    TerminalCompletionDisposition.SUBMITTED_KNOWN_FAILED,
                )
                self.assertEqual(artifacts.summary.command_count, 2)
                session = app.journal.snapshot()
                self.assertIsNotNone(session.pending_repair_receipt_id)
                self.assertTrue(session.latest_failure_signatures)
                replayed = JsonlTerminalTrialJournal(
                    "trial-failed-verification",
                    Path(directory) / "aar-transcript.jsonl",
                )
                self.assertIs(
                    replayed.snapshot().completion_disposition,
                    TerminalCompletionDisposition.SUBMITTED_KNOWN_FAILED,
                )
                self.assertTrue(replayed.trace_consistent)
            finally:
                app.close()

    def test_legacy_submitted_unverified_session_remains_loadable(self) -> None:
        session = TerminalSessionSnapshot.model_validate(
            {
                "trial_id": "legacy-unverified",
                "completion_disposition": "submitted_unverified",
                "task_generation": 0,
                "known_state_generation": 0,
            }
        )

        self.assertIs(
            session.completion_disposition,
            TerminalCompletionDisposition.SUBMITTED_UNVERIFIED,
        )

    async def test_failed_verification_allows_changed_read_only_reverify(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-artifact", call_key="work-1"),
                verify_draft("assert-artifact", call_key="verification-1"),
                verify_draft(
                    "assert-artifact-with-fresh-input",
                    call_key="verification-2",
                ),
            )
            app = build_terminal_application(
                trial_id="trial-repair-before-reverify",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(return_code=1, stderr="assertion failed"),
                    completed_result(stdout="fresh verification passed"),
                ),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(
                    max_proposal_rejections=0,
                    max_no_progress_seconds=None,
                ),
            )
            try:
                artifacts = await app.run("create and validate an artifact")

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertEqual(artifacts.summary.command_count, 3)
                self.assertTrue(capability.requests[2].repair_mode)
                self.assertEqual(
                    app.journal.snapshot().proposal_rejections,
                    0,
                )
                self.assertTrue(artifacts.summary.trace_consistent)
            finally:
                app.close()

    async def test_settled_nonzero_work_allows_read_only_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("best-effort-repair", call_key="work-1"),
                verify_draft("official-check", call_key="verification-1"),
            )
            app = build_terminal_application(
                trial_id="trial-nonzero-work-verify",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(return_code=1, stderr="partial repair"),
                    completed_result(stdout="artifact is valid"),
                ),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(max_no_progress_seconds=None),
            )
            try:
                artifacts = await app.run("repair and validate an artifact")
                session = app.journal.snapshot()

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertEqual(artifacts.summary.command_count, 2)
                self.assertEqual(session.task_generation, 1)
                self.assertEqual(session.known_state_generation, 1)
                self.assertIsNone(session.successful_work_generation)
                self.assertEqual(session.proposal_rejections, 0)
                self.assertTrue(artifacts.summary.trace_consistent)
            finally:
                app.close()

    async def test_known_initial_generation_allows_direct_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                verify_draft("official-check", call_key="verification-1"),
            )
            app = build_terminal_application(
                trial_id="trial-initial-direct-verify",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(stdout="existing artifact is valid"),
                ),
                proposal_capability=capability,
            )
            try:
                artifacts = await app.run("validate the existing artifact")
                session = app.journal.snapshot()

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertEqual(artifacts.summary.command_count, 1)
                self.assertEqual(session.task_generation, 0)
                self.assertEqual(session.known_state_generation, 0)
                self.assertIsNone(session.successful_work_generation)
                self.assertEqual(session.proposal_rejections, 0)
            finally:
                app.close()

    async def test_unchanged_failed_verify_retry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-artifact", call_key="work-1"),
                verify_draft("assert-artifact", call_key="verification-1"),
                verify_draft("assert-artifact", call_key="verification-2"),
            )
            app = build_terminal_application(
                trial_id="trial-unchanged-reverify",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(return_code=1, stderr="assertion failed"),
                ),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(
                    max_proposal_rejections=0,
                    max_no_progress_seconds=None,
                ),
            )
            try:
                artifacts = await app.run("create and validate an artifact")

                self.assertEqual(
                    artifacts.runtime_result.final_state.status,
                    RunStatus.FAILED,
                )
                self.assertEqual(artifacts.summary.command_count, 2)
                self.assertIn(
                    "command, inputs, or evidence changes",
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
                self.assertTrue(capability.requests[2].repair_mode)
                self.assertTrue(capability.requests[2].recovery_mode)
                self.assertFalse(capability.requests[2].verification_due)
                self.assertFalse(capability.requests[3].repair_mode)
                self.assertTrue(capability.requests[3].verification_due)
                self.assertTrue(capability.requests[3].recovery_mode)
                self.assertTrue(capability.requests[4].repair_mode)
                self.assertTrue(capability.requests[5].verification_due)
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

    async def test_partial_verification_is_audited_without_success_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verification = TerminalVerificationContract(
                evidence_kind="official_tests",
                evidence_provenance="task_provided",
                evidence_sources=("/tests/official_verify.py",),
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
            environment = FakeTerminalEnvironment(
                completed_result(),
                completed_result(stdout="partial verification passed"),
            )
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

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertFalse(artifacts.summary.agent_complete)
                self.assertIs(
                    artifacts.summary.completion_disposition,
                    TerminalCompletionDisposition.SUBMITTED_UNVERIFIED,
                )
                self.assertEqual(len(environment.calls), 2)
                session = app.journal.snapshot()
                self.assertIsNone(session.verified_checkpoint)
                self.assertIsNotNone(session.latest_verification_receipt)
                assert session.latest_verification_receipt is not None
                self.assertTrue(session.latest_verification_receipt.passed)
                self.assertEqual(
                    session.latest_verification_receipt.covered_requirement_ids,
                    ("req-001",),
                )
            finally:
                app.close()

    async def test_inference_timeout_retries_once_in_emergency_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = TimeoutThenScriptedCapability(
                execute_draft("create-artifact"),
                verify_draft("test -e /app"),
            )
            capability.deadline_budget_profile = (
                terra_high_deadline_budget_profile()
            )
            app = build_terminal_application(
                trial_id="trial-inference-delivery-retry",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(),
                ),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(
                    max_wall_clock_seconds=840,
                    max_no_progress_seconds=None,
                ),
            )
            try:
                artifacts = await app.run("create and validate an artifact")
                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertFalse(capability.requests[0].delivery_mode)
                self.assertTrue(capability.requests[1].delivery_mode)
                self.assertTrue(capability.requests[1].emergency_mode)
                self.assertTrue(capability.requests[1].recovery_mode)
                first_cap = capability.requests[0].execution_limits.max_inference_timeout_sec
                emergency_cap = capability.requests[1].execution_limits.max_inference_timeout_sec
                self.assertIsNotNone(first_cap)
                self.assertIsNotNone(emergency_cap)
                assert first_cap is not None
                assert emergency_cap is not None
                self.assertLess(emergency_cap, first_cap)
                self.assertLessEqual(emergency_cap, 60)
                self.assertEqual(len(app.journal.records()), 2)
            finally:
                app.close()

    async def test_repeated_inference_timeout_submits_known_state(self) -> None:
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
                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertFalse(artifacts.summary.agent_complete)
                self.assertIs(
                    artifacts.summary.completion_disposition,
                    TerminalCompletionDisposition.SUBMITTED_UNVERIFIED,
                )
                self.assertEqual(len(capability.requests), 2)
                self.assertFalse(capability.requests[0].delivery_mode)
                self.assertTrue(capability.requests[1].delivery_mode)
                self.assertTrue(capability.requests[1].emergency_mode)
                self.assertTrue(artifacts.summary.trace_consistent)
            finally:
                app.close()

    async def test_deadline_pressure_submits_without_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("too-late", call_key="work-1"),
            )
            capability.deadline_budget_profile = (
                terra_high_deadline_budget_profile()
            )
            app = build_terminal_application(
                trial_id="trial-deadline-submit",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(
                    max_wall_clock_seconds=840,
                    max_no_progress_seconds=None,
                ),
            )
            app.journal._session = app.journal.snapshot().model_copy(
                update={"started_at": utc_now() - timedelta(seconds=830)}
            )
            try:
                artifacts = await app.run("preserve the current artifact")

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertFalse(artifacts.summary.agent_complete)
                self.assertIs(
                    artifacts.summary.completion_disposition,
                    TerminalCompletionDisposition.SUBMITTED_UNVERIFIED,
                )
                self.assertEqual(len(capability.requests), 0)
                self.assertEqual(len(app.journal.records()), 0)
                self.assertTrue(artifacts.summary.trace_consistent)
            finally:
                app.close()

    async def test_deadline_pressure_never_submits_unresolved_in_doubt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft(
                    "inspect-state",
                    call_key="inspect-1",
                    command_role=TerminalCommandRole.INSPECT,
                ),
            )
            capability.deadline_budget_profile = (
                terra_high_deadline_budget_profile()
            )
            app = build_terminal_application(
                trial_id="trial-in-doubt-deadline",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(
                    max_wall_clock_seconds=840,
                    max_no_progress_seconds=None,
                ),
            )
            app.journal._session = app.journal.snapshot().model_copy(
                update={
                    "started_at": utc_now() - timedelta(seconds=830),
                    "known_state_generation": None,
                    "in_doubt_reconciliation_required": True,
                    "reconciliation_state": TerminalReconciliationState.REQUIRED,
                }
            )
            try:
                artifacts = await app.run("preserve the uncertain artifact")

                self.assertFalse(artifacts.runtime_result.succeeded)
                self.assertIs(
                    artifacts.summary.completion_disposition,
                    TerminalCompletionDisposition.IN_PROGRESS,
                )
                self.assertEqual(len(capability.requests), 0)
                self.assertEqual(len(app.journal.records()), 0)
                self.assertTrue(artifacts.summary.trace_consistent)
            finally:
                app.close()

    async def test_delivery_timeout_retries_once_in_emergency_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = TimeoutThenScriptedCapability(
                execute_draft("create-artifact", timeout_sec=60),
                verify_draft("test -e /app", timeout_sec=60),
            )
            app = build_terminal_application(
                trial_id="trial-inference-emergency-retry",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(),
                ),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(
                    max_wall_clock_seconds=840,
                    max_no_progress_seconds=None,
                ),
            )
            app.journal._session = app.journal.snapshot().model_copy(
                update={"started_at": utc_now() - timedelta(seconds=400)}
            )
            try:
                artifacts = await app.run("create and verify an artifact")

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertEqual(len(capability.requests), 3)
                self.assertTrue(capability.requests[0].delivery_mode)
                self.assertFalse(capability.requests[0].emergency_mode)
                self.assertTrue(capability.requests[1].delivery_mode)
                self.assertTrue(capability.requests[1].emergency_mode)
                self.assertTrue(artifacts.summary.trace_consistent)
            finally:
                app.close()

    async def test_deadline_timeout_is_clamped_before_core_termination(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft(
                    "slow-repair",
                    call_key="work-too-wide",
                    timeout_sec=300,
                ),
                verify_draft("test -e /app", timeout_sec=60),
            )
            environment = FakeTerminalEnvironment(
                completed_result(stdout="repaired"),
                completed_result(stdout="verified"),
            )
            app = build_terminal_application(
                trial_id="trial-deadline-timeout-correction",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(
                    max_wall_clock_seconds=840,
                    max_no_progress_seconds=None,
                ),
            )
            app.journal._session = app.journal.snapshot().model_copy(
                update={"started_at": utc_now() - timedelta(seconds=650)}
            )
            try:
                artifacts = await app.run("repair and verify the artifact")
                session = app.journal.snapshot()

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertEqual(len(environment.calls), 2)
                self.assertEqual(session.proposal_rejections, 0)
                self.assertEqual(
                    capability.requests[0]
                    .execution_limits.timeout_admission_margin_seconds,
                    8.0,
                )
                self.assertLessEqual(
                    capability.requests[0].execution_limits.max_timeout_sec,
                    172,
                )
                record = app.journal.records()[0]
                self.assertEqual(record.intent.requested_timeout_sec, 300)
                self.assertEqual(
                    record.intent.timeout_sec,
                    record.intent.advertised_timeout_cap_sec,
                )
                self.assertLessEqual(record.intent.timeout_sec, 172)
                self.assertEqual(
                    record.intent.timeout_cap_reason,
                    TerminalTimeoutCapReason.ADVERTISED_CAP,
                )
                self.assertTrue(artifacts.summary.trace_consistent)
            finally:
                app.close()

    async def test_finalization_advises_verification_without_blocking_work(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-artifact", call_key="work-1"),
                execute_draft("explore-more", call_key="work-2"),
                verify_draft("test -e /app", call_key="verification-1"),
            )
            app = build_terminal_application(
                trial_id="trial-finalization-verification",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(),
                ),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(
                    max_wall_clock_seconds=840,
                    max_no_progress_seconds=None,
                ),
            )
            app.journal._session = app.journal.snapshot().model_copy(
                update={"started_at": utc_now() - timedelta(seconds=600)}
            )
            try:
                artifacts = await app.run("create and verify the artifact")

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertEqual(len(app.journal.records()), 3)
                self.assertEqual(app.journal.snapshot().proposal_rejections, 0)
                self.assertTrue(capability.requests[1].finalization_mode)
                self.assertTrue(capability.requests[1].verification_due)
                self.assertTrue(capability.requests[2].verification_due)
            finally:
                app.close()

    async def test_late_work_preserves_finalization_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-artifact", call_key="work-1"),
                execute_draft(
                    "wide-repair",
                    call_key="work-too-wide",
                    timeout_sec=180,
                ),
                verify_draft(
                    "test -e /app",
                    call_key="verification-1",
                    timeout_sec=60,
                ),
            )
            app = build_terminal_application(
                trial_id="trial-finalization-reserve",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(),
                ),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(
                    max_wall_clock_seconds=840,
                    max_no_progress_seconds=None,
                ),
            )
            app.journal._session = app.journal.snapshot().model_copy(
                update={"started_at": utc_now() - timedelta(seconds=500)}
            )
            try:
                artifacts = await app.run("repair and verify the artifact")

                self.assertTrue(artifacts.runtime_result.succeeded)
                records = app.journal.records()
                self.assertEqual(len(records), 3)
                self.assertEqual(app.journal.snapshot().proposal_rejections, 0)
                self.assertEqual(records[1].intent.requested_timeout_sec, 180)
                self.assertEqual(records[1].intent.timeout_sec, 180)
                self.assertEqual(
                    records[1].intent.timeout_cap_reason,
                    TerminalTimeoutCapReason.MODEL_REQUESTED,
                )
            finally:
                app.close()

    async def test_failed_verification_signatures_are_structured_for_repair(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft("create-artifact", call_key="work-1"),
                verify_draft("check-all", call_key="verification-1"),
                execute_draft("repair-all", call_key="work-2"),
                verify_draft("check-all", call_key="verification-2"),
            )
            app = build_terminal_application(
                trial_id="trial-failure-signatures",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(),
                    completed_result(
                        return_code=1,
                        stdout="FAIL: np.int alias remains\nERROR: np.float alias remains",
                    ),
                    completed_result(),
                    completed_result(),
                ),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(max_no_progress_seconds=None),
            )
            try:
                artifacts = await app.run("create and verify the artifact")

                self.assertTrue(artifacts.runtime_result.succeeded)
                signatures = capability.requests[2].session.latest_failure_signatures
                self.assertEqual(len(signatures), 2)
                self.assertTrue(any("np.int" in item for item in signatures))
                self.assertTrue(any("np.float" in item for item in signatures))
                self.assertEqual(
                    app.journal.snapshot().latest_failure_signatures,
                    (),
                )
            finally:
                app.close()

    async def test_performance_protocol_is_assurance_advice_not_execution_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verification = TerminalVerificationContract(
                evidence_kind="independent_check",
                evidence_sources=("fresh benchmark process",),
                artifact_paths=("/app/eigen.py",),
                requirement_coverage=("req-001",),
                coverage_dimensions=(
                    "artifact",
                    "format",
                    "semantic",
                    "end_to_end",
                ),
                validation_methods=("time repeated warmed input",),
            )
            environment = FakeTerminalEnvironment(completed_result())
            app = build_terminal_application(
                trial_id="trial-performance-protocol",
                logs_dir=directory,
                environment=environment,
                proposal_capability=ScriptedTerminalTurnCapability(
                    execute_draft("create-eigen-implementation"),
                    verify_draft("benchmark", verification=verification),
                ),
                policy=TerminalExecutionPolicy(
                    max_proposal_rejections=0,
                    max_no_progress_seconds=None,
                ),
            )
            try:
                artifacts = await app.run(
                    "Make the implementation consistently faster than the reference benchmark."
                )

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertFalse(artifacts.summary.agent_complete)
                self.assertIs(
                    artifacts.summary.completion_disposition,
                    TerminalCompletionDisposition.SUBMITTED_UNVERIFIED,
                )
                self.assertEqual(len(environment.calls), 2)
                self.assertIsNone(app.journal.snapshot().verified_checkpoint)
            finally:
                app.close()

    async def test_verify_without_evidence_contract_can_submit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capability = ScriptedTerminalTurnCapability(
                execute_draft(
                    "read-only-check",
                    call_key="verification-1",
                    command_role=TerminalCommandRole.VERIFY,
                    verification=None,
                ),
                complete_draft(),
            )
            app = build_terminal_application(
                trial_id="trial-uncontracted-verify",
                logs_dir=directory,
                environment=FakeTerminalEnvironment(
                    completed_result(stdout="check passed"),
                ),
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(max_completion_rejections=0),
            )
            try:
                artifacts = await app.run("validate the existing artifact")

                self.assertTrue(artifacts.runtime_result.succeeded)
                self.assertFalse(artifacts.summary.agent_complete)
                self.assertIs(
                    artifacts.summary.completion_disposition,
                    TerminalCompletionDisposition.SUBMITTED_UNVERIFIED,
                )
                self.assertEqual(artifacts.summary.command_count, 1)
                self.assertEqual(len(capability.requests), 2)
                session = app.journal.snapshot()
                self.assertIsNone(session.latest_verification_receipt)
                self.assertIsNone(session.verified_checkpoint)
                self.assertEqual(session.proposal_rejections, 0)
            finally:
                app.close()

    async def test_artifact_first_recovery_is_advisory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = FakeTerminalEnvironment(
                completed_result(stdout="capabilities discovered"),
            )
            capability = ScriptedTerminalTurnCapability(
                execute_draft(
                    "inspect-capabilities",
                    call_key="inspect-1",
                    command_role=TerminalCommandRole.INSPECT,
                ),
                execute_draft(
                    "inspect-more",
                    call_key="inspect-2",
                    command_role=TerminalCommandRole.INSPECT,
                ),
            )
            app = build_terminal_application(
                trial_id="trial-artifact-first-recovery",
                logs_dir=directory,
                environment=environment,
                proposal_capability=capability,
                policy=TerminalExecutionPolicy(
                    max_proposal_rejections=0,
                    max_no_progress_seconds=None,
                ),
            )
            try:
                artifacts = await app.run("Create /app/filter.py.")

                self.assertEqual(
                    artifacts.runtime_result.final_state.status,
                    RunStatus.FAILED,
                )
                self.assertEqual(len(environment.calls), 2)
                self.assertEqual(len(capability.requests), 3)
                self.assertTrue(
                    capability.requests[0].artifact_first_mode
                )
                self.assertFalse(capability.requests[0].recovery_mode)
                self.assertTrue(
                    capability.requests[1].artifact_first_mode
                )
                self.assertTrue(capability.requests[1].recovery_mode)
            finally:
                app.close()

    async def test_consecutive_inspection_recovery_is_advisory(self) -> None:
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
                    max_artifact_first_inspections=None,
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
                self.assertEqual(len(environment.calls), 4)
                self.assertEqual(app.journal.snapshot().consecutive_inspections, 4)
                self.assertTrue(capability.requests[-1].recovery_mode)
            finally:
                app.close()

    async def test_total_inspection_recovery_is_advisory(self) -> None:
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
                self.assertEqual(len(app.journal.records()), 11)
                session = app.journal.snapshot()
                self.assertEqual(session.inspection_commands, 6)
                self.assertEqual(session.consecutive_inspections, 1)
                self.assertTrue(capability.requests[-1].recovery_mode)
            finally:
                app.close()


if __name__ == "__main__":
    unittest.main()

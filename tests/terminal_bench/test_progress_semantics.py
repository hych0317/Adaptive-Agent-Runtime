from __future__ import annotations

import unittest
from uuid import uuid4

from adaptive_agent_runtime.core import OutcomeCertainty, ProgressKind
from adaptive_agent_runtime.tool_ecosystem import (
    ToolInvocation,
    ToolProviderOutcome,
)
from adaptive_agent_runtime.tool_ecosystem.models import (
    ToolAttemptStatus,
    ToolExecutionStatus,
)

from applications.terminal_bench.contracts import (
    TerminalExecutionError,
    terminal_exception_indicates_timeout,
    terminal_exception_outcome,
)
from applications.terminal_bench.harbor_agent import (
    HarborTerminalEnvironmentAdapter,
)
from applications.terminal_bench.models import (
    TERMINAL_COMMAND_CAPABILITY,
    TERMINAL_COMMAND_PROVIDER,
    TerminalCommandIntent,
    TerminalCommandRecord,
    TerminalCommandRole,
    TerminalExecResult,
    TerminalExecutionPolicy,
    TerminalExecutionState,
    utc_now,
)
from applications.terminal_bench.planner import JsonlTerminalTrialJournal
from applications.terminal_bench.progress import (
    assess_terminal_progress,
    terminal_failure_code,
    terminal_outcome_certainty,
)
from applications.terminal_bench.tools import (
    TerminalCommandProvider,
    terminal_tool_observation_from_result,
)
from tests.terminal_bench.fakes import completed_result


def _intent(
    *,
    role: TerminalCommandRole = TerminalCommandRole.WORK,
    call_key: str = "call-1",
) -> TerminalCommandIntent:
    return TerminalCommandIntent(
        trial_id="trial-progress",
        call_key=call_key,
        command="run-check",
        timeout_sec=30,
        command_role=role,
    )


def _record(
    result: TerminalExecResult,
    *,
    role: TerminalCommandRole = TerminalCommandRole.WORK,
    call_key: str = "prior-call",
) -> TerminalCommandRecord:
    return TerminalCommandRecord(
        action_id=uuid4(),
        invocation_id=uuid4(),
        decision_request_id=uuid4(),
        effect_fingerprint="0" * 64,
        intent=_intent(role=role, call_key=call_key),
        result=result,
        governance_status="applied",
    )


def _uncertain_result(
    *,
    state: TerminalExecutionState,
    timed_out: bool = False,
) -> TerminalExecResult:
    now = utc_now()
    return TerminalExecResult(
        stderr="provider unavailable",
        started_at=now,
        completed_at=now,
        duration_ms=0,
        execution_state=state,
        timed_out=timed_out,
        transport_failed=True,
    )


class TerminalProgressSemanticsTests(unittest.TestCase):
    def test_new_failure_is_recovery_evidence(self) -> None:
        result = completed_result(return_code=1, stderr="ERROR missing input")

        assessment = assess_terminal_progress(_intent(), result, ())

        self.assertIs(assessment.kind, ProgressKind.RECOVERY_PROGRESS)
        self.assertTrue(assessment.novel)
        self.assertEqual(assessment.basis, "new_failure_evidence")

    def test_repeated_failure_is_no_progress(self) -> None:
        result = completed_result(return_code=1, stderr="ERROR missing input")

        assessment = assess_terminal_progress(
            _intent(),
            result,
            (_record(result),),
        )

        self.assertIs(assessment.kind, ProgressKind.NO_PROGRESS)
        self.assertFalse(assessment.novel)
        self.assertEqual(assessment.basis, "repeated_failure")

    def test_failure_seen_before_intervening_error_is_still_repeated(self) -> None:
        first = completed_result(return_code=1, stderr="ERROR missing input")
        second = completed_result(return_code=1, stderr="ERROR invalid format")

        assessment = assess_terminal_progress(
            _intent(),
            first,
            (
                _record(first, call_key="first"),
                _record(second, call_key="second"),
            ),
        )

        self.assertIs(assessment.kind, ProgressKind.NO_PROGRESS)
        self.assertFalse(assessment.novel)

    def test_distinct_unmarked_diagnostics_are_new_recovery_evidence(self) -> None:
        first = completed_result(return_code=1, stderr="invalid option")
        second = completed_result(return_code=1, stderr="permission denied")

        assessment = assess_terminal_progress(
            _intent(call_key="second"),
            second,
            (_record(first, call_key="first"),),
        )

        self.assertIs(assessment.kind, ProgressKind.RECOVERY_PROGRESS)
        self.assertTrue(assessment.novel)

    def test_stdout_changes_are_kept_when_stderr_is_constant(self) -> None:
        first = completed_result(
            return_code=1,
            stdout="invalid option",
            stderr="warning",
        )
        second = completed_result(
            return_code=1,
            stdout="permission denied",
            stderr="warning",
        )

        assessment = assess_terminal_progress(
            _intent(call_key="second"),
            second,
            (_record(first, call_key="first"),),
        )

        self.assertIs(assessment.kind, ProgressKind.RECOVERY_PROGRESS)
        self.assertTrue(assessment.novel)

    def test_stable_tmp_paths_are_not_normalized_as_random_values(self) -> None:
        first = completed_result(
            return_code=1,
            stderr="cannot read /tmp/input.csv",
        )
        second = completed_result(
            return_code=1,
            stderr="cannot read /tmp/output.csv",
        )

        assessment = assess_terminal_progress(
            _intent(call_key="second"),
            second,
            (_record(first, call_key="first"),),
        )

        self.assertIs(assessment.kind, ProgressKind.RECOVERY_PROGRESS)
        self.assertTrue(assessment.novel)

    def test_volatile_diagnostic_values_do_not_manufacture_novelty(self) -> None:
        first = completed_result(
            return_code=1,
            stderr=(
                "2026-08-11T10:11:12Z permission denied "
                "/tmp/tmpabcdef"
            ),
        )
        second = completed_result(
            return_code=1,
            stderr=(
                "2026-08-12T13:14:15Z permission denied "
                "/tmp/tmpghijkl"
            ),
        )

        assessment = assess_terminal_progress(
            _intent(call_key="second"),
            second,
            (_record(first, call_key="first"),),
        )

        self.assertIs(assessment.kind, ProgressKind.NO_PROGRESS)
        self.assertFalse(assessment.novel)

    def test_successful_work_keeps_task_progress_semantics(self) -> None:
        assessment = assess_terminal_progress(
            _intent(),
            completed_result(return_code=0, stdout="done"),
            (),
        )

        self.assertIs(assessment.kind, ProgressKind.TASK_PROGRESS)
        self.assertIsNone(assessment.fingerprint)

    def test_completed_timeout_is_failure_evidence_not_task_progress(self) -> None:
        result = completed_result(return_code=0).model_copy(
            update={"timed_out": True}
        )

        assessment = assess_terminal_progress(_intent(), result, ())

        self.assertFalse(result.settled)
        self.assertFalse(result.succeeded)
        self.assertIs(assessment.kind, ProgressKind.RECOVERY_PROGRESS)
        self.assertEqual(terminal_failure_code(result), "tool.timeout")

        invocation = ToolInvocation(
            requirement_id=uuid4(),
            capability_id=TERMINAL_COMMAND_CAPABILITY,
            provider_id=TERMINAL_COMMAND_PROVIDER,
        )
        observation = terminal_tool_observation_from_result(
            invocation,
            result,
        )
        self.assertIs(observation.status, ToolExecutionStatus.TIMED_OUT)
        self.assertIs(
            observation.attempts[0].status,
            ToolAttemptStatus.TIMED_OUT,
        )

    def test_repeated_successful_inspection_is_no_progress(self) -> None:
        result = completed_result(return_code=0, stdout="same diagnostic")
        intent = _intent(role=TerminalCommandRole.INSPECT)

        assessment = assess_terminal_progress(
            intent,
            result,
            (_record(result, role=TerminalCommandRole.INSPECT),),
        )

        self.assertIs(assessment.kind, ProgressKind.NO_PROGRESS)
        self.assertFalse(assessment.novel)

    def test_failed_to_start_is_certain_and_has_distinct_code(self) -> None:
        result = _uncertain_result(state=TerminalExecutionState.FAILED_TO_START)

        self.assertIs(
            terminal_outcome_certainty(result),
            OutcomeCertainty.CERTAIN,
        )
        self.assertEqual(
            terminal_failure_code(result),
            "terminal.failed_to_start",
        )

    def test_command_timeout_preserves_in_doubt_certainty(self) -> None:
        result = _uncertain_result(
            state=TerminalExecutionState.IN_DOUBT,
            timed_out=True,
        )

        self.assertIs(
            terminal_outcome_certainty(result),
            OutcomeCertainty.IN_DOUBT,
        )
        self.assertEqual(terminal_failure_code(result), "tool.timeout")


class _RaisingEnvironment:
    def __init__(self, exception: BaseException) -> None:
        self.exception = exception

    async def exec(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        raise self.exception


class CommandTimeoutError(RuntimeError):
    pass


class TerminalExceptionSemanticsTests(unittest.IsolatedAsyncioTestCase):
    def test_timeout_classifier_follows_wrapped_cause(self) -> None:
        inner = CommandTimeoutError()
        outer = RuntimeError("terminal provider failed")
        outer.__cause__ = inner

        self.assertTrue(terminal_exception_indicates_timeout(outer))
        state, timed_out = terminal_exception_outcome(outer)
        self.assertIs(state, TerminalExecutionState.IN_DOUBT)
        self.assertTrue(timed_out)

    def test_suppressed_timeout_context_is_not_reported(self) -> None:
        try:
            try:
                raise TimeoutError("hidden timeout")
            except TimeoutError:
                raise RuntimeError("public provider failure") from None
        except RuntimeError as exc:
            suppressed = exc

        self.assertTrue(suppressed.__suppress_context__)
        self.assertFalse(terminal_exception_indicates_timeout(suppressed))
        state, timed_out = terminal_exception_outcome(suppressed)
        self.assertIs(state, TerminalExecutionState.IN_DOUBT)
        self.assertFalse(timed_out)

    async def test_known_not_started_never_claims_command_timeout(self) -> None:
        adapter = HarborTerminalEnvironmentAdapter(
            _RaisingEnvironment(
                TerminalExecutionError(
                    "Command timed out after 30 seconds",
                    command_started=False,
                    timed_out=True,
                )
            ),  # type: ignore[arg-type]
            max_output_characters=1_000,
        )

        result = await adapter.exec("long-command", timeout_sec=30)

        self.assertIs(
            result.execution_state,
            TerminalExecutionState.FAILED_TO_START,
        )
        self.assertFalse(result.timed_out)
        self.assertTrue(result.transport_failed)

    async def test_timeout_exception_class_without_message_is_recognized(
        self,
    ) -> None:
        adapter = HarborTerminalEnvironmentAdapter(
            _RaisingEnvironment(CommandTimeoutError()),  # type: ignore[arg-type]
            max_output_characters=1_000,
        )

        result = await adapter.exec("long-command", timeout_sec=30)

        self.assertIs(result.execution_state, TerminalExecutionState.IN_DOUBT)
        self.assertTrue(result.timed_out)
        self.assertTrue(result.transport_failed)

    async def test_provider_uses_shared_timeout_mapping(self) -> None:
        journal = JsonlTerminalTrialJournal("trial-provider-timeout")
        provider = TerminalCommandProvider(
            environment=_RaisingEnvironment(  # type: ignore[arg-type]
                RuntimeError("Command timed out after 30 seconds")
            ),
            journal=journal,
            policy=TerminalExecutionPolicy(),
        )
        invocation = ToolInvocation(
            requirement_id=uuid4(),
            capability_id=TERMINAL_COMMAND_CAPABILITY,
            provider_id=TERMINAL_COMMAND_PROVIDER,
            arguments={
                "trial_id": "trial-provider-timeout",
                "command": "long-command",
                "command_role": "work",
                "cwd": None,
                "env": {},
                "timeout_sec": 30,
                "process_reference": None,
                "verification": None,
            },
        )

        provider_result = await provider.invoke(invocation)
        recorded = journal.execution_for(invocation.invocation_id)

        self.assertFalse(provider_result.succeeded)
        self.assertIs(provider_result.outcome, ToolProviderOutcome.TIMED_OUT)
        self.assertIsNotNone(recorded)
        assert recorded is not None
        self.assertIs(recorded.execution_state, TerminalExecutionState.IN_DOUBT)
        self.assertTrue(recorded.timed_out)
        self.assertTrue(recorded.transport_failed)
        reconstructed = terminal_tool_observation_from_result(
            invocation,
            recorded,
        )
        self.assertIs(reconstructed.status, ToolExecutionStatus.TIMED_OUT)
        self.assertIs(
            reconstructed.attempts[0].status,
            ToolAttemptStatus.TIMED_OUT,
        )


if __name__ == "__main__":
    unittest.main()

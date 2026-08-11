from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from adaptive_agent_runtime.llm import InferenceUsage

from applications.terminal_bench.models import (
    TerminalCommandRole,
    TerminalExecResult,
    TerminalExecutionState,
    TerminalVerificationContract,
    TerminalTurnDraft,
    TerminalTurnProposal,
    TerminalTurnRequest,
    utc_now,
)


@dataclass(frozen=True)
class ExecCall:
    command: str
    cwd: str | None
    env: dict[str, str] | None
    timeout_sec: int | None


class FakeTerminalEnvironment:
    def __init__(self, *results: TerminalExecResult | BaseException) -> None:
        self._results = list(results)
        self.calls: list[ExecCall] = []

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> TerminalExecResult:
        self.calls.append(
            ExecCall(
                command=command,
                cwd=cwd,
                env=(dict(env) if env is not None else None),
                timeout_sec=timeout_sec,
            )
        )
        if not self._results:
            return completed_result(stdout=f"ran:{command}")
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class ScriptedTerminalTurnCapability:
    module_id = "test.terminal_turn.scripted"

    def __init__(
        self,
        *drafts: TerminalTurnDraft,
        usage: InferenceUsage | None = None,
    ) -> None:
        self._drafts = list(drafts)
        self._usage = usage or InferenceUsage()
        self.requests: list[TerminalTurnRequest] = []

    async def propose(self, request: TerminalTurnRequest) -> TerminalTurnProposal:
        self.requests.append(request)
        if not self._drafts:
            raise RuntimeError("scripted terminal drafts exhausted")
        return TerminalTurnProposal(
            draft=self._drafts.pop(0),
            usage=self._usage,
            model_id="test/model",
        )


def execute_draft(
    command: str,
    *,
    call_key: str = "command-1",
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout_sec: int | None = None,
    command_role: TerminalCommandRole = TerminalCommandRole.WORK,
    verification: TerminalVerificationContract | None = None,
) -> TerminalTurnDraft:
    return TerminalTurnDraft(
        decision="execute",
        call_key=call_key,
        command=command,
        command_role=command_role,
        verification=verification,
        cwd=cwd,
        env=env,
        timeout_sec=timeout_sec,
        rationale=f"run {call_key}",
    )


def verify_draft(
    command: str = "true",
    *,
    call_key: str = "verification-1",
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout_sec: int | None = None,
    verification: TerminalVerificationContract | None = None,
) -> TerminalTurnDraft:
    if verification is None:
        verification = TerminalVerificationContract(
            evidence_kind="official_tests",
            evidence_provenance="task_provided",
            evidence_sources=("/tests/scripted_official_test.py",),
            artifact_paths=(cwd or "/app",),
            requirement_coverage=("req-001",),
            coverage_dimensions=(
                "artifact",
                "format",
                "semantic",
                "end_to_end",
            ),
            validation_methods=("scripted end-to-end assertion",),
        )
    return execute_draft(
        command,
        call_key=call_key,
        cwd=cwd,
        env=env,
        timeout_sec=timeout_sec,
        command_role=TerminalCommandRole.VERIFY,
        verification=verification,
    )


def complete_draft(summary: str = "done") -> TerminalTurnDraft:
    return TerminalTurnDraft(
        decision="complete",
        summary=summary,
        rationale="task is complete",
    )


def completed_result(
    *,
    return_code: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> TerminalExecResult:
    now = utc_now()
    return TerminalExecResult(
        stdout=stdout,
        stderr=stderr,
        return_code=return_code,
        started_at=now,
        completed_at=now,
        duration_ms=0,
        execution_state=TerminalExecutionState.COMPLETED,
    )

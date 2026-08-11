from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re

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
from applications.terminal_bench.tools import _RUNTIME_EVIDENCE_MARKER


@dataclass(frozen=True)
class ExecCall:
    command: str
    cwd: str | None
    env: dict[str, str] | None
    timeout_sec: int | None


class FakeTerminalEnvironment:
    def __init__(
        self,
        *results: TerminalExecResult | BaseException,
        runtime_evidence_stdout: str | None = None,
    ) -> None:
        self._results = list(results)
        self._runtime_evidence_stdout = runtime_evidence_stdout
        self.calls: list[ExecCall] = []
        self.runtime_evidence_calls: list[ExecCall] = []

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> TerminalExecResult:
        call = ExecCall(
            command=command,
            cwd=cwd,
            env=(dict(env) if env is not None else None),
            timeout_sec=timeout_sec,
        )
        if _RUNTIME_EVIDENCE_MARKER in command:
            self.runtime_evidence_calls.append(call)
            stdout = self._runtime_evidence_stdout
            if stdout is None:
                source_indices = re.findall(r"printf 'S\\t(\d+)", command)
                artifact_indices = re.findall(r"printf 'A\\t(\d+)", command)
                lines = [_RUNTIME_EVIDENCE_MARKER]
                lines.extend(
                    f"S\t{index}\t/tests/scripted_official_test.py\t0\t{'a' * 64}"
                    for index in source_indices
                )
                lines.extend(
                    f"A\t{index}\t{'b' * 64}" for index in artifact_indices
                )
                stdout = "\n".join(lines)
            return completed_result(stdout=stdout)
        self.calls.append(call)
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

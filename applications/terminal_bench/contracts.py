"""Ports kept independent from Harbor and any concrete model provider."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable
from uuid import UUID

from adaptive_agent_runtime.core import Observation

from applications.terminal_bench.models import (
    TerminalCommandRecord,
    TerminalExecResult,
    TerminalExecutionState,
    TerminalPendingCommand,
    TerminalProposalRejection,
    TerminalRequirement,
    TerminalSessionSnapshot,
    TerminalTaskContract,
    TerminalTrialSummary,
    TerminalTurnProposal,
    TerminalTurnRequest,
)


class TerminalExecutionError(RuntimeError):
    """An adapter error with explicit knowledge about whether execution began."""

    def __init__(
        self,
        message: str,
        *,
        command_started: bool | None,
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.command_started = command_started
        self.timed_out = timed_out


_TIMEOUT_MARKERS = (
    "timed out",
    "time out",
    "timeout after",
    "timeout of",
    "deadline exceeded",
    "deadline expired",
)


def terminal_exception_indicates_timeout(exc: BaseException) -> bool:
    """Recognize structured and wrapped timeout errors at an adapter boundary."""

    for current in _exception_chain(exc):
        if bool(getattr(current, "timed_out", False)):
            return True
        if isinstance(current, TimeoutError):
            return True
        class_name = current.__class__.__name__.lower().replace("_", "")
        if "timeout" in class_name or "timedout" in class_name:
            return True
        detail = (str(current) or current.__class__.__name__).lower()
        if any(marker in detail for marker in _TIMEOUT_MARKERS):
            return True
    return False


def terminal_exception_outcome(
    exc: BaseException,
) -> tuple[TerminalExecutionState, bool]:
    """Return execution certainty and a non-contradictory timeout flag."""

    command_started: bool | None = None
    for current in _exception_chain(exc):
        candidate = getattr(current, "command_started", None)
        if isinstance(candidate, bool):
            command_started = candidate
            break
    state = (
        TerminalExecutionState.FAILED_TO_START
        if command_started is False
        else TerminalExecutionState.IN_DOUBT
    )
    timed_out = bool(
        state is TerminalExecutionState.IN_DOUBT
        and terminal_exception_indicates_timeout(exc)
    )
    return state, timed_out


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    pending = [exc]
    values: list[BaseException] = []
    seen: set[int] = set()
    while pending and len(values) < 16:
        current = pending.pop(0)
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        values.append(current)
        cause = current.__cause__
        context = current.__context__
        if cause is not None:
            pending.append(cause)
        elif context is not None and not current.__suppress_context__:
            pending.append(context)
        nested = getattr(current, "exceptions", ())
        if isinstance(nested, tuple):
            pending.extend(
                item for item in nested if isinstance(item, BaseException)
            )
    return tuple(values)


@runtime_checkable
class TerminalEnvironment(Protocol):
    """One independent, non-interactive shell execution per call."""

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> TerminalExecResult: ...


@runtime_checkable
class TerminalTurnProposalCapability(Protocol):
    @property
    def module_id(self) -> str: ...

    async def propose(self, request: TerminalTurnRequest) -> TerminalTurnProposal: ...


@runtime_checkable
class TerminalTrialJournal(Protocol):
    @property
    def trial_id(self) -> str: ...

    def snapshot(self) -> TerminalSessionSnapshot: ...

    def bind_task_contract(
        self,
        requirements: tuple[TerminalRequirement, ...],
        *,
        coverage_complete: bool = True,
        unmapped_fragments: tuple[str, ...] = (),
    ) -> None: ...

    def task_contract(self) -> TerminalTaskContract | None: ...

    def recent_records(self, limit: int) -> tuple[TerminalCommandRecord, ...]: ...

    def pending(self) -> TerminalPendingCommand | None: ...

    def save_pending(self, pending: TerminalPendingCommand) -> None: ...

    def abandon_pending(
        self,
        action_id: UUID,
        *,
        reason: str,
        phase: str,
    ) -> None: ...

    def record_execution(self, invocation_id: UUID, result: TerminalExecResult) -> None: ...

    def execution_for(self, invocation_id: UUID) -> TerminalExecResult | None: ...

    def commit_observation(self, observation: Observation) -> None: ...

    def record_usage(self, proposal: TerminalTurnProposal) -> None: ...

    def completion_gate_error(self) -> str | None: ...

    def record_completion_rejection(self, reason: str) -> None: ...

    def record_proposal_rejection(
        self,
        rejection: TerminalProposalRejection,
    ) -> None: ...

    def record_reconciliation_proposal_rejection(
        self,
        rejection: TerminalProposalRejection,
    ) -> None: ...

    def submission_gate_error(self) -> str | None: ...

    def mark_submitted_unverified(self, summary: str) -> None: ...

    def mark_complete(self, summary: str) -> None: ...

    def write_summary(self, summary: TerminalTrialSummary) -> None: ...

"""Ports kept independent from Harbor and any concrete model provider."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable
from uuid import UUID

from adaptive_agent_runtime.core import Observation

from applications.terminal_bench.models import (
    TerminalCommandRecord,
    TerminalExecResult,
    TerminalPendingCommand,
    TerminalSessionSnapshot,
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

    def recent_records(self, limit: int) -> tuple[TerminalCommandRecord, ...]: ...

    def pending(self) -> TerminalPendingCommand | None: ...

    def save_pending(self, pending: TerminalPendingCommand) -> None: ...

    def record_execution(self, invocation_id: UUID, result: TerminalExecResult) -> None: ...

    def execution_for(self, invocation_id: UUID) -> TerminalExecResult | None: ...

    def commit_observation(self, observation: Observation) -> None: ...

    def record_usage(self, proposal: TerminalTurnProposal) -> None: ...

    def mark_complete(self, summary: str) -> None: ...

    def write_summary(self, summary: TerminalTrialSummary) -> None: ...

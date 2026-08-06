"""Harbor-independent domain models for the sequential terminal profile."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from adaptive_agent_runtime.core import ActionRequest
from adaptive_agent_runtime.llm import InferenceUsage


AAR_TERMINAL_SEQUENTIAL_PROFILE = "AAR Terminal Sequential Profile"
TERMINAL_COMMAND_ACTION = "terminal.command"
TERMINAL_COMMAND_CAPABILITY = "terminal.command.execute"
TERMINAL_COMMAND_PROVIDER = "harbor.environment.exec"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def terminal_fingerprint(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TerminalModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class TerminalExecutionState(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED_TO_START = "FAILED_TO_START"
    IN_DOUBT = "IN_DOUBT"


class TerminalTurnDecision(StrEnum):
    EXECUTE = "execute"
    COMPLETE = "complete"


class TerminalProcessReference(TerminalModel):
    """Explicit evidence needed to inspect a detached background service."""

    reference_id: str = Field(min_length=1, max_length=128)
    pid_file: str = Field(min_length=1, max_length=4096)
    log_path: str = Field(min_length=1, max_length=4096)
    status_check_command: str = Field(min_length=1, max_length=20_000)
    stop_command: str | None = Field(default=None, min_length=1, max_length=20_000)


class TerminalExecResult(TerminalModel):
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    started_at: AwareDatetime
    completed_at: AwareDatetime
    duration_ms: int = Field(ge=0)
    execution_state: TerminalExecutionState
    timed_out: bool = False
    transport_failed: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    @model_validator(mode="after")
    def validate_execution_result(self) -> TerminalExecResult:
        if self.completed_at < self.started_at:
            raise ValueError("terminal completion cannot precede start")
        if self.execution_state is TerminalExecutionState.COMPLETED:
            if self.return_code is None:
                raise ValueError("completed terminal execution requires return_code")
            if self.transport_failed:
                raise ValueError("completed execution cannot be a transport failure")
        elif self.return_code is not None:
            raise ValueError("uncertain or unstarted execution cannot have return_code")
        if (
            self.execution_state is TerminalExecutionState.FAILED_TO_START
            and self.timed_out
        ):
            raise ValueError("a command known not to start cannot time out")
        return self

    @property
    def command_completed(self) -> bool:
        return self.execution_state is TerminalExecutionState.COMPLETED


class TerminalExecutionPolicy(TerminalModel):
    max_commands: int = Field(default=64, ge=1, le=512)
    default_timeout_sec: int = Field(default=120, ge=1)
    max_timeout_sec: int = Field(default=300, ge=1)
    provider_grace_sec: int = Field(default=10, ge=1, le=120)
    max_command_characters: int = Field(default=20_000, ge=1)
    max_output_characters: int = Field(default=64_000, ge=1)
    max_context_output_characters: int = Field(default=12_000, ge=1)
    max_context_records: int = Field(default=12, ge=1)
    max_environment_variables: int = Field(default=64, ge=0)
    max_environment_value_characters: int = Field(default=4096, ge=1)
    max_total_tokens: int | None = Field(default=200_000, ge=1)
    max_cost_usd: float | None = Field(default=None, ge=0.0)
    max_wall_clock_seconds: float | None = Field(default=None, gt=0.0)
    max_active_execution_seconds: float | None = Field(default=None, gt=0.0)
    external_job_deadline_seconds: float | None = Field(default=None, gt=0.0)
    cleanup_grace_seconds: float = Field(default=10.0, ge=0.0)
    repeated_invocation_limit: int | None = Field(default=3, ge=2)
    max_no_progress_steps: int | None = Field(default=5, ge=1)
    max_no_progress_seconds: float | None = Field(default=300.0, gt=0.0)

    @model_validator(mode="after")
    def validate_timeouts(self) -> TerminalExecutionPolicy:
        if self.default_timeout_sec > self.max_timeout_sec:
            raise ValueError("default timeout cannot exceed maximum timeout")
        return self


class TerminalSessionSnapshot(TerminalModel):
    trial_id: str = Field(min_length=1)
    current_cwd: str | None = Field(default=None, min_length=1, max_length=4096)
    environment: dict[str, str] = Field(default_factory=dict)
    process_references: tuple[TerminalProcessReference, ...] = ()
    committed_commands: int = Field(default=0, ge=0)
    denied_commands: int = Field(default=0, ge=0)
    timed_out_commands: int = Field(default=0, ge=0)
    in_doubt_commands: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class TerminalHistoryItem(TerminalModel):
    action_id: UUID
    call_key: str = Field(min_length=1)
    command: str = Field(min_length=1)
    cwd: str | None = None
    environment_keys: tuple[str, ...] = ()
    timeout_sec: int = Field(ge=1)
    execution_state: TerminalExecutionState
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    transport_failed: bool = False


class TerminalTurnRequest(TerminalModel):
    run_id: UUID
    task_id: UUID
    instruction: str = Field(min_length=1)
    profile: str = AAR_TERMINAL_SEQUENTIAL_PROFILE
    session: TerminalSessionSnapshot
    recent_history: tuple[TerminalHistoryItem, ...] = ()
    remaining_commands: int = Field(ge=0)
    remaining_tokens: int | None = Field(default=None, ge=0)
    remaining_cost_usd: float | None = Field(default=None, ge=0.0)
    execution_semantics: tuple[str, ...] = Field(min_length=1)


class TerminalTurnDraft(TerminalModel):
    """Authority-free model draft for one command or voluntary completion."""

    decision: TerminalTurnDecision
    call_key: str | None = Field(default=None, min_length=1, max_length=256)
    command: str | None = Field(default=None, min_length=1)
    cwd: str | None = Field(default=None, min_length=1, max_length=4096)
    env: dict[str, str] | None = None
    timeout_sec: int | None = Field(default=None, ge=1)
    process_reference: TerminalProcessReference | None = None
    summary: str | None = Field(default=None, min_length=1)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_turn(self) -> TerminalTurnDraft:
        execution_fields = (self.call_key, self.command)
        if self.decision is TerminalTurnDecision.EXECUTE:
            if any(item is None for item in execution_fields):
                raise ValueError("execute draft requires call_key and command")
            if self.summary is not None:
                raise ValueError("execute draft cannot contain completion summary")
        else:
            if any(item is not None for item in execution_fields):
                raise ValueError("complete draft cannot contain a command identity")
            if any(
                item is not None
                for item in (
                    self.cwd,
                    self.env,
                    self.timeout_sec,
                    self.process_reference,
                )
            ):
                raise ValueError("complete draft cannot contain execution settings")
            if self.summary is None:
                raise ValueError("complete draft requires summary")
        return self


class TerminalTurnProposal(TerminalModel):
    draft: TerminalTurnDraft
    usage: InferenceUsage = Field(default_factory=InferenceUsage)
    model_id: str | None = Field(default=None, min_length=1)


class TerminalCommandIntent(TerminalModel):
    """Runtime-resolved, fully explicit command carried by one Core Action."""

    trial_id: str = Field(min_length=1)
    call_key: str = Field(min_length=1, max_length=256)
    command: str = Field(min_length=1)
    cwd: str | None = Field(default=None, min_length=1, max_length=4096)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_sec: int = Field(ge=1)
    process_reference: TerminalProcessReference | None = None

    def tool_arguments(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "command": self.command,
            "cwd": self.cwd,
            "env": dict(self.env),
            "timeout_sec": self.timeout_sec,
            "process_reference": (
                self.process_reference.model_dump(mode="json")
                if self.process_reference is not None
                else None
            ),
        }

    @property
    def execution_fingerprint(self) -> str:
        return terminal_fingerprint(self.tool_arguments())


class TerminalPendingCommand(TerminalModel):
    action: ActionRequest
    intent: TerminalCommandIntent
    state_revision: int = Field(ge=0)
    proposal: TerminalTurnDraft
    created_at: AwareDatetime = Field(default_factory=utc_now)


class TerminalCommandRecord(TerminalModel):
    action_id: UUID
    invocation_id: UUID
    decision_request_id: UUID
    effect_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    intent: TerminalCommandIntent
    result: TerminalExecResult
    governance_status: str = Field(min_length=1)
    committed_at: AwareDatetime = Field(default_factory=utc_now)

    def history_item(self, *, output_limit: int) -> TerminalHistoryItem:
        def bounded(value: str) -> str:
            if len(value) <= output_limit:
                return value
            half = max(1, output_limit // 2)
            return value[:half] + "\n...[context truncated]...\n" + value[-half:]

        return TerminalHistoryItem(
            action_id=self.action_id,
            call_key=self.intent.call_key,
            command=self.intent.command,
            cwd=self.intent.cwd,
            environment_keys=tuple(sorted(self.intent.env)),
            timeout_sec=self.intent.timeout_sec,
            execution_state=self.result.execution_state,
            return_code=self.result.return_code,
            stdout=bounded(self.result.stdout),
            stderr=bounded(self.result.stderr),
            timed_out=self.result.timed_out,
            transport_failed=self.result.transport_failed,
        )


class TerminalTrialSummary(TerminalModel):
    profile: str = AAR_TERMINAL_SEQUENTIAL_PROFILE
    trial_id: str = Field(min_length=1)
    run_id: UUID
    task_id: UUID
    agent_complete: bool
    runtime_status: str = Field(min_length=1)
    final_output: Any = None
    command_count: int = Field(ge=0)
    denial_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    in_doubt_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    latency_ms: int = Field(ge=0)
    trace_consistent: bool
    started_at: AwareDatetime
    completed_at: AwareDatetime


class TerminalBenchmarkAnalysis(TerminalModel):
    trial_id: str = Field(min_length=1)
    verifier_rewards: dict[str, float] = Field(default_factory=dict)
    verifier_reward: float | None = None
    benchmark_pass: bool
    agent_complete: bool
    completion_matches_verifier: bool
    command_count: int = Field(ge=0)
    denial_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    in_doubt_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    latency_ms: int = Field(ge=0)
    trace_consistent: bool

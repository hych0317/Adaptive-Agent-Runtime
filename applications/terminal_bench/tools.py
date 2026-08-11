"""Terminal Provider that exposes only the current trial BaseEnvironment port."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import cast

from adaptive_agent_runtime.tool_ecosystem import (
    RetryStatus,
    ToolAttempt,
    ToolAttemptStatus,
    ToolExecutionStatus,
    ToolInvocation,
    ToolObservation,
    ToolProviderMetadata,
    ToolProviderOutcome,
    ToolProviderResult,
)
from adaptive_agent_runtime.tool_ecosystem.models import ImmutableJsonObject

from applications.terminal_bench.contracts import (
    TerminalEnvironment,
    TerminalExecutionError,
    TerminalTrialJournal,
    terminal_exception_outcome,
)
from applications.terminal_bench.models import (
    TERMINAL_COMMAND_CAPABILITY,
    TERMINAL_COMMAND_PROVIDER,
    TerminalCommandIntent,
    TerminalExecResult,
    TerminalExecutionPolicy,
    TerminalExecutionState,
    utc_now,
)


def terminal_provider_metadata(
    policy: TerminalExecutionPolicy,
) -> ToolProviderMetadata:
    process_reference_schema = {
        "type": ["object", "null"],
        "properties": {
            "reference_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "pid_file": {"type": "string", "minLength": 1, "maxLength": 4096},
            "log_path": {"type": "string", "minLength": 1, "maxLength": 4096},
            "status_check_command": {
                "type": "string",
                "minLength": 1,
                "maxLength": policy.max_command_characters,
            },
            "stop_command": {
                "type": ["string", "null"],
                "minLength": 1,
                "maxLength": policy.max_command_characters,
            },
        },
        "required": [
            "reference_id",
            "pid_file",
            "log_path",
            "status_check_command",
            "stop_command",
        ],
        "additionalProperties": False,
    }
    verification_schema = {
        "type": ["object", "null"],
        "properties": {
            "evidence_kind": {
                "type": "string",
                "enum": ["official_tests", "independent_check"],
            },
            "evidence_sources": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "artifact_paths": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "requirement_coverage": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "pattern": "^req-[0-9]{3,}$",
                },
            },
            "coverage_dimensions": {
                "type": "array",
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "enum": ["artifact", "format", "semantic", "end_to_end"],
                },
            },
            "evidence_provenance": {
                "type": "string",
                "enum": [
                    "task_provided",
                    "external_standard",
                    "runtime_observed",
                    "agent_generated",
                    "legacy_unspecified",
                ],
            },
            "artifact_fingerprints": {
                "type": "array",
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "pattern": "^[^=\\s]{1,4096}=sha256:[0-9a-f]{64}$",
                },
            },
            "validation_methods": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "independence_method": {
                "type": "string",
                "enum": [
                    "independent_oracle",
                    "cross_implementation",
                    "property_based",
                    "alternate_evidence",
                    "official_tests",
                ],
            },
            "process_isolation": {
                "type": "string",
                "enum": ["fresh_process", "ephemeral_fixture"],
            },
            "performance_protocol": {
                "type": "string",
                "enum": ["not_applicable", "cold_unique_inputs"],
            },
            "state_policy": {
                "type": "string",
                "enum": ["read_only"],
            },
        },
        "required": [
            "evidence_kind",
            "evidence_sources",
            "artifact_paths",
            "requirement_coverage",
            "coverage_dimensions",
            "evidence_provenance",
            "artifact_fingerprints",
            "validation_methods",
            "independence_method",
            "process_isolation",
            "performance_protocol",
            "state_policy",
        ],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "trial_id": {"type": "string", "minLength": 1},
            "command": {
                "type": "string",
                "minLength": 1,
                "maxLength": policy.max_command_characters,
            },
            "command_role": {
                "type": "string",
                "enum": ["inspect", "work", "verify"],
            },
            "cwd": {
                "type": ["string", "null"],
                "minLength": 1,
                "maxLength": 4096,
            },
            "env": {
                "type": "object",
                "maxProperties": policy.max_environment_variables,
                "additionalProperties": {
                    "type": "string",
                    "maxLength": policy.max_environment_value_characters,
                },
            },
            "timeout_sec": {
                "type": "integer",
                "minimum": 1,
                "maximum": policy.max_timeout_sec,
            },
            "process_reference": process_reference_schema,
            "verification": verification_schema,
        },
        "required": [
            "trial_id",
            "command",
            "command_role",
            "cwd",
            "env",
            "timeout_sec",
            "process_reference",
            "verification",
        ],
        "additionalProperties": False,
    }
    return ToolProviderMetadata(
        provider_id=TERMINAL_COMMAND_PROVIDER,
        name="Harbor trial command execution",
        capability_id=TERMINAL_COMMAND_CAPABILITY,
        description=(
            "Execute one independent non-interactive command in the current "
            "Harbor trial's main BaseEnvironment."
        ),
        input_schema=cast(ImmutableJsonObject, schema),
        tags=("terminal", "harbor", "trial_scoped"),
        selection_priority=100,
    )


class TerminalCommandProvider:
    module_id = "terminal_bench.provider.harbor_environment"
    provider_id = TERMINAL_COMMAND_PROVIDER

    def __init__(
        self,
        *,
        environment: TerminalEnvironment,
        journal: TerminalTrialJournal,
        policy: TerminalExecutionPolicy,
    ) -> None:
        self._environment = environment
        self._journal = journal
        self._policy = policy

    async def invoke(self, invocation: ToolInvocation) -> ToolProviderResult:
        try:
            intent = TerminalCommandIntent.model_validate(
                {
                    **dict(invocation.arguments),
                    "call_key": "runtime-bound",
                }
            )
        except Exception as exc:
            return ToolProviderResult.failed(
                error=f"invalid terminal invocation: {exc}",
                retryable=False,
            )
        if intent.trial_id != self._journal.trial_id:
            return ToolProviderResult.failed(
                error="terminal invocation escaped its trial scope",
                retryable=False,
            )
        started_at = utc_now()
        try:
            result = await self._environment.exec(
                intent.command,
                cwd=intent.cwd,
                env=dict(intent.env),
                timeout_sec=intent.timeout_sec,
            )
            if not isinstance(result, TerminalExecResult):
                raise TypeError("TerminalEnvironment must return TerminalExecResult")
        except TerminalExecutionError as exc:
            state, timed_out = terminal_exception_outcome(exc)
            result = self._failure_result(
                exc,
                started_at=started_at,
                state=state,
                timed_out=timed_out,
            )
        except TimeoutError as exc:
            state, timed_out = terminal_exception_outcome(exc)
            result = self._failure_result(
                exc,
                started_at=started_at,
                state=state,
                timed_out=timed_out,
            )
        except Exception as exc:
            state, timed_out = terminal_exception_outcome(exc)
            result = self._failure_result(
                exc,
                started_at=started_at,
                state=state,
                timed_out=timed_out,
            )
        result = self._bounded_output(result)
        self._journal.record_execution(invocation.invocation_id, result)
        if result.settled:
            return ToolProviderResult.ok(output=result.model_dump(mode="json"))
        error = _terminal_tool_error(result)
        if _terminal_provider_outcome(result) is ToolProviderOutcome.TIMED_OUT:
            return ToolProviderResult.timed_out_result(
                error=error,
                retryable=False,
            )
        return ToolProviderResult.failed(error=error, retryable=False)

    @staticmethod
    def _failure_result(
        exc: BaseException,
        *,
        started_at: datetime,
        state: TerminalExecutionState,
        timed_out: bool,
    ) -> TerminalExecResult:
        completed_at = utc_now()
        detail = str(exc) or exc.__class__.__name__
        return TerminalExecResult(
            stderr=f"{exc.__class__.__name__}: {detail}",
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=max(
                0,
                int((completed_at - started_at).total_seconds() * 1000),
            ),
            execution_state=state,
            timed_out=timed_out,
            transport_failed=True,
        )

    def _bounded_output(self, result: TerminalExecResult) -> TerminalExecResult:
        limit = self._policy.max_output_characters
        stdout, stdout_truncated = _truncate(result.stdout, limit)
        stderr, stderr_truncated = _truncate(result.stderr, limit)
        return result.model_copy(
            update={
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": result.stdout_truncated or stdout_truncated,
                "stderr_truncated": result.stderr_truncated or stderr_truncated,
            }
        )


def terminal_tool_observation_from_result(
    invocation: ToolInvocation,
    result: TerminalExecResult,
) -> ToolObservation:
    """Reconstruct an authoritative Tool result without replaying the command."""

    provider_outcome = _terminal_provider_outcome(result)
    succeeded = provider_outcome is ToolProviderOutcome.SUCCEEDED
    attempt_status = {
        ToolProviderOutcome.SUCCEEDED: ToolAttemptStatus.SUCCEEDED,
        ToolProviderOutcome.FAILED: ToolAttemptStatus.FAILED,
        ToolProviderOutcome.TIMED_OUT: ToolAttemptStatus.TIMED_OUT,
    }[provider_outcome]
    error = None if succeeded else _terminal_tool_error(result)
    attempt = ToolAttempt(
        attempt_number=1,
        status=attempt_status,
        started_at=result.started_at,
        completed_at=result.completed_at,
        output=result.model_dump(mode="json") if succeeded else None,
        error=error,
        retryable=False,
    )
    return ToolObservation(
        invocation_id=invocation.invocation_id,
        requirement_id=invocation.requirement_id,
        capability_id=invocation.capability_id,
        provider_id=invocation.provider_id,
        status=(
            {
                ToolProviderOutcome.SUCCEEDED: ToolExecutionStatus.SUCCEEDED,
                ToolProviderOutcome.FAILED: ToolExecutionStatus.FAILED,
                ToolProviderOutcome.TIMED_OUT: ToolExecutionStatus.TIMED_OUT,
            }[provider_outcome]
        ),
        retry_status=RetryStatus.NOT_RETRIED,
        output=result.model_dump(mode="json") if succeeded else None,
        error=error,
        attempts=(attempt,),
        correlation=invocation.correlation,
        started_at=result.started_at,
        completed_at=result.completed_at,
    )


def _terminal_provider_outcome(
    result: TerminalExecResult,
) -> ToolProviderOutcome:
    if result.settled:
        return ToolProviderOutcome.SUCCEEDED
    if result.timed_out:
        return ToolProviderOutcome.TIMED_OUT
    return ToolProviderOutcome.FAILED


def _terminal_tool_error(result: TerminalExecResult) -> str:
    label = "TIMED_OUT" if result.timed_out else result.execution_state.value
    return (
        f"terminal execution {label}: "
        f"{result.stderr or 'no final process status'}"
    )


def _truncate(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    half = max(1, limit // 2)
    return value[:half] + "\n...[output truncated]...\n" + value[-half:], True

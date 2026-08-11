"""Terminal Provider that exposes only the current trial BaseEnvironment port."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import shlex
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
    TerminalCommandRole,
    TerminalCommandIntent,
    TerminalEvidenceProvenance,
    TerminalExecResult,
    TerminalExecutionPolicy,
    TerminalExecutionState,
    TerminalRuntimeVerificationEvidence,
    TerminalVerificationContract,
    utc_now,
)
_RUNTIME_EVIDENCE_MARKER = "AAR_RUNTIME_EVIDENCE_V1"
_RUNTIME_EVIDENCE_TIMEOUT_SECONDS = 15
_MAX_RUNTIME_EVIDENCE_PATHS = 64
_TASK_EVIDENCE_ROOTS = ("/tests", "/test")
_STANDARD_EVIDENCE_ROOTS = ("/usr", "/bin", "/sbin", "/lib", "/lib64")


def _path_is_within(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(root + "/") for root in roots)


def _runtime_evidence_command(
    verification: TerminalVerificationContract,
) -> str | None:
    paths = (*verification.evidence_sources, *verification.artifact_paths)
    if len(paths) > _MAX_RUNTIME_EVIDENCE_PATHS or any(
        not path.startswith("/") or any(char.isspace() for char in path)
        for path in verification.artifact_paths
    ):
        return None
    lines = [
        "set -eu",
        "hash_path() {",
        "  if [ -f \"$1\" ]; then",
        "    sha256sum -- \"$1\" | awk '{print $1}'",
        "  elif [ -d \"$1\" ]; then",
        (
            "    (cd \"$1\" && find . -type f -print0 | LC_ALL=C "
            "sort -z | xargs -0 -r sha256sum) | sha256sum | awk '{print $1}'"
        ),
        "  else",
        "    return 1",
        "  fi",
        "}",
        f"printf '{_RUNTIME_EVIDENCE_MARKER}\\n'",
    ]
    for index, path in enumerate(verification.evidence_sources):
        if not path.startswith("/") or any(
            char.isspace() for char in path
        ):
            continue
        lines.extend(
            (
                f"p={shlex.quote(path)}",
                'r=$(readlink -f -- "$p")',
                'c=$(stat -c %Z -- "$r")',
                'h=$(hash_path "$r")',
                f"printf 'S\\t{index}\\t%s\\t%s\\t%s\\n' \"$r\" \"$c\" \"$h\"",
            )
        )
    for index, path in enumerate(verification.artifact_paths):
        lines.extend(
            (
                f"p={shlex.quote(path)}",
                'r=$(readlink -f -- "$p")',
                'h=$(hash_path "$r")',
                f"printf 'A\\t{index}\\t%s\\n' \"$h\"",
            )
        )
    return "\n".join(lines)


def _runtime_evidence_failure(
    verification: TerminalVerificationContract,
    reason: str,
) -> TerminalRuntimeVerificationEvidence:
    return TerminalRuntimeVerificationEvidence(
        requested_provenance=verification.evidence_provenance,
        observed_at=utc_now(),
        failure_reason=reason[:1024],
    )


def _parse_runtime_evidence(
    verification: TerminalVerificationContract,
    result: TerminalExecResult,
    *,
    trial_started_epoch: float,
) -> TerminalRuntimeVerificationEvidence:
    if not result.succeeded:
        return _runtime_evidence_failure(
            verification,
            "Runtime evidence probe did not complete successfully",
        )
    lines = result.stdout.splitlines()
    if not lines or lines[0] != _RUNTIME_EVIDENCE_MARKER:
        return _runtime_evidence_failure(
            verification,
            "Runtime evidence probe returned an invalid marker",
        )
    sources: dict[int, tuple[str, int, str]] = {}
    artifacts: dict[int, str] = {}
    try:
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) == 5 and parts[0] == "S":
                index = int(parts[1])
                path, changed_at, digest = parts[2], int(parts[3]), parts[4]
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("invalid source digest")
                sources[index] = (path, changed_at, digest)
            elif len(parts) == 3 and parts[0] == "A":
                index = int(parts[1])
                digest = parts[2]
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("invalid artifact digest")
                artifacts[index] = digest
            else:
                raise ValueError("invalid evidence line")
        if not set(sources).issubset(range(len(verification.evidence_sources))):
            raise ValueError("invalid evidence source index")
        if set(artifacts) != set(range(len(verification.artifact_paths))):
            raise ValueError("incomplete artifact fingerprints")
    except (TypeError, ValueError):
        return _runtime_evidence_failure(
            verification,
            "Runtime evidence probe output could not be validated",
        )
    source_fingerprints = tuple(
        f"{sources[index][0]}=sha256:{sources[index][2]}"
        for index in sorted(sources)
    )
    artifact_fingerprints = tuple(
        f"{path}=sha256:{artifacts[index]}"
        for index, path in enumerate(verification.artifact_paths)
    )
    roots = (
        _TASK_EVIDENCE_ROOTS
        if verification.evidence_provenance
        is TerminalEvidenceProvenance.TASK_PROVIDED
        else _STANDARD_EVIDENCE_ROOTS
    )
    provenance_verified = bool(
        verification.evidence_provenance
        in {
            TerminalEvidenceProvenance.TASK_PROVIDED,
            TerminalEvidenceProvenance.EXTERNAL_STANDARD,
        }
        and set(sources) == set(range(len(verification.evidence_sources)))
        and all(
            _path_is_within(path, roots)
            and changed_at <= trial_started_epoch + 1.0
            for path, changed_at, _digest in sources.values()
        )
    )
    return TerminalRuntimeVerificationEvidence(
        requested_provenance=verification.evidence_provenance,
        provenance_verified=provenance_verified,
        evidence_source_fingerprints=source_fingerprints,
        artifact_fingerprints=artifact_fingerprints,
        observed_at=utc_now(),
        failure_reason=(
            None
            if provenance_verified
            else "declared provenance was not verified by Runtime"
        ),
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
        self._trial_started_epoch = utc_now().timestamp()

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
        if (
            result.succeeded
            and intent.command_role is TerminalCommandRole.VERIFY
            and intent.verification is not None
        ):
            consumed_seconds = (result.duration_ms + 999) // 1000
            remaining_timeout_sec = max(
                0,
                intent.timeout_sec - consumed_seconds,
            )
            runtime_verification = await self._collect_runtime_evidence(
                intent,
                remaining_timeout_sec=remaining_timeout_sec,
            )
            result = result.model_copy(
                update={"runtime_verification": runtime_verification}
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

    async def _collect_runtime_evidence(
        self,
        intent: TerminalCommandIntent,
        *,
        remaining_timeout_sec: int,
    ) -> TerminalRuntimeVerificationEvidence:
        verification = intent.verification
        assert verification is not None
        command = _runtime_evidence_command(verification)
        if command is None:
            return _runtime_evidence_failure(
                verification,
                "evidence paths are not eligible for Runtime inspection",
            )
        if remaining_timeout_sec < 1:
            return _runtime_evidence_failure(
                verification,
                "VERIFY role timeout left no budget for Runtime evidence inspection",
            )
        try:
            result = await self._environment.exec(
                command,
                cwd=intent.cwd,
                env={"PATH": "/usr/bin:/bin"},
                timeout_sec=min(
                    _RUNTIME_EVIDENCE_TIMEOUT_SECONDS,
                    remaining_timeout_sec,
                ),
            )
            if not isinstance(result, TerminalExecResult):
                raise TypeError("TerminalEnvironment must return TerminalExecResult")
        except Exception as exc:
            return _runtime_evidence_failure(
                verification,
                "Runtime evidence probe failed: " + exc.__class__.__name__,
            )
        return _parse_runtime_evidence(
            verification,
            result,
            trial_started_epoch=self._trial_started_epoch,
        )

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

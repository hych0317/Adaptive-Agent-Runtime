"""Deterministic progress evidence for the sequential terminal profile."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re

from adaptive_agent_runtime.core import OutcomeCertainty, ProgressKind

from applications.terminal_bench.models import (
    TerminalCommandIntent,
    TerminalCommandRecord,
    TerminalCommandRole,
    TerminalExecResult,
    TerminalExecutionState,
    terminal_fingerprint,
)


_FAILURE_SIGNATURE_MARKER = re.compile(
    r"(?i)(?:\bFAIL(?:ED)?\b|\bERROR\b|AssertionError|AttributeError|"
    r"ImportError|ModuleNotFoundError|command not found|No such file or directory|"
    r"timed? out|IN_DOUBT|returned? non-zero|exit(?:ed)?[ =:]+[1-9])"
)
_SENSITIVE_FAILURE_VALUE = re.compile(
    r"(?i)(?:AKIA|ASIA)[A-Z0-9]{16}|"
    r"(?:gh[pousr]_|github_pat_|hf_)[A-Za-z0-9_]{16,}|"
    r"(?<=[=:\"'])[^\"'\s]{24,}(?=[\"'])"
)
_VOLATILE_FAILURE_VALUE = re.compile(
    r"(?i)(?:\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\b|"
    r"\b20[0-9]{2}-[0-9]{2}-[0-9]{2}[T ][0-9:.+-]+Z?\b|"
    r"/(?:tmp|var/tmp)/(?:tmp[A-Za-z0-9_.-]{6,}|"
    r"pytest[-_][A-Za-z0-9_.-]{4,}))"
)


@dataclass(frozen=True)
class TerminalProgressAssessment:
    """Trusted classification attached to one terminal observation."""

    kind: ProgressKind
    fingerprint: str | None
    basis: str
    novel: bool | None = None

    def metadata(self) -> dict[str, str | bool | None]:
        return {
            "kind": self.kind.value,
            "fingerprint": self.fingerprint,
            "basis": self.basis,
            "novel": self.novel,
        }


def terminal_failure_signatures(result: TerminalExecResult) -> tuple[str, ...]:
    """Return bounded, redacted failure families suitable for comparison."""

    values: list[str] = []
    combined = "\n".join((result.stdout, result.stderr))
    for raw_line in combined.splitlines():
        line = raw_line.strip()
        if not line or _FAILURE_SIGNATURE_MARKER.search(line) is None:
            continue
        line = _normalized_diagnostic(line, limit=300)
        if line not in values:
            values.append(line)
        if len(values) >= 12:
            break
    if not values and not result.settled:
        values.append(
            "execution_state="
            + result.execution_state.value
            + "; timed_out="
            + str(result.timed_out).lower()
            + "; transport_failed="
            + str(result.transport_failed).lower()
        )
    elif not values and result.return_code not in (None, 0):
        values.append(f"command returned non-zero status {result.return_code}")
    return tuple(values)


def terminal_failure_fingerprint(result: TerminalExecResult) -> str:
    """Fingerprint the observable failure cause without command identity."""

    return terminal_fingerprint(
        {
            "execution_state": result.execution_state.value,
            "return_code": result.return_code,
            "timed_out": result.timed_out,
            "transport_failed": result.transport_failed,
            "signatures": terminal_failure_signatures(result),
            "diagnostic_fingerprint": _diagnostic_fingerprint(result),
        }
    )


def assess_terminal_progress(
    intent: TerminalCommandIntent,
    result: TerminalExecResult,
    prior_records: Sequence[TerminalCommandRecord],
) -> TerminalProgressAssessment:
    """Classify progress from committed evidence, not model claims.

    A newly observed failure is recovery information. Repeating any failure
    already present in the bounded trial history is stagnation, including an
    A/B/A oscillation rather than only consecutive repetition.
    """

    successful = result.succeeded
    if successful and intent.command_role is not TerminalCommandRole.INSPECT:
        return TerminalProgressAssessment(
            kind=ProgressKind.TASK_PROGRESS,
            fingerprint=None,
            basis="successful_task_command",
        )
    if successful:
        fingerprint = _inspection_fingerprint(result)
        seen = any(
            record.intent.command_role is TerminalCommandRole.INSPECT
            and record.result.succeeded
            and _inspection_fingerprint(record.result) == fingerprint
            for record in prior_records
        )
        return TerminalProgressAssessment(
            kind=(
                ProgressKind.NO_PROGRESS
                if seen
                else ProgressKind.RECOVERY_PROGRESS
            ),
            fingerprint=fingerprint,
            basis=("repeated_inspection" if seen else "new_inspection_evidence"),
            novel=not seen,
        )

    fingerprint = terminal_failure_fingerprint(result)
    seen = any(
        _is_failure(record.result)
        and terminal_failure_fingerprint(record.result) == fingerprint
        for record in prior_records
    )
    return TerminalProgressAssessment(
        kind=(ProgressKind.NO_PROGRESS if seen else ProgressKind.RECOVERY_PROGRESS),
        fingerprint=fingerprint,
        basis=("repeated_failure" if seen else "new_failure_evidence"),
        novel=not seen,
    )


def terminal_outcome_certainty(result: TerminalExecResult) -> OutcomeCertainty:
    """Map process-state knowledge to Core's orthogonal certainty axis."""

    return (
        OutcomeCertainty.IN_DOUBT
        if result.execution_state is TerminalExecutionState.IN_DOUBT
        else OutcomeCertainty.CERTAIN
    )


def terminal_failure_code(result: TerminalExecResult) -> str:
    """Return a stable cause code without flattening execution certainty."""

    if result.timed_out:
        return "tool.timeout"
    if result.execution_state is TerminalExecutionState.FAILED_TO_START:
        return "terminal.failed_to_start"
    if result.transport_failed:
        return "terminal.transport_failed"
    return "terminal.execution_uncertain"


def _inspection_fingerprint(result: TerminalExecResult) -> str:
    return terminal_fingerprint(
        {
            "return_code": result.return_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )


def _diagnostic_fingerprint(result: TerminalExecResult) -> str | None:
    normalized = {
        "stdout": _normalized_diagnostic(result.stdout, limit=1_000),
        "stderr": _normalized_diagnostic(result.stderr, limit=1_000),
    }
    if not any(normalized.values()):
        return None
    return terminal_fingerprint(normalized)


def _normalized_diagnostic(value: str, *, limit: int) -> str:
    redacted = _SENSITIVE_FAILURE_VALUE.sub("<redacted>", value)
    stable = _VOLATILE_FAILURE_VALUE.sub("<volatile>", redacted)
    normalized = " ".join(stable.split())
    if len(normalized) <= limit:
        return normalized
    half = max(1, limit // 2)
    return normalized[:half] + " ...<bounded>... " + normalized[-half:]


def _is_failure(result: TerminalExecResult) -> bool:
    return not result.succeeded

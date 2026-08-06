"""Unified Tool execution with availability, timeout, retry, and trace."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Mapping, cast
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime.core.invocation import current_run_invocation_guard
from adaptive_agent_runtime.tool_ecosystem.contracts import (
    ToolExecutionGovernor,
    ToolProvider,
    ToolRegistry,
    ToolTraceSink,
)
from adaptive_agent_runtime.tool_ecosystem.errors import ProviderNotFoundError
from adaptive_agent_runtime.tool_ecosystem.governance import (
    OperationalToolGovernor,
    ToolExecutionPolicy,
)
from adaptive_agent_runtime.tool_ecosystem.models import (
    ProviderAvailability,
    RetryStatus,
    ToolAttempt,
    ToolAttemptStatus,
    ToolExecutionStatus,
    ToolInvocation,
    ToolObservation,
    ToolProviderResult,
    ToolTraceEntry,
    ToolTraceEvent,
    ToolTraceEventKind,
    utc_now,
)
from adaptive_agent_runtime.governance.contracts import CommitPermitValidation
from adaptive_agent_runtime.governance.models import (
    GovernanceTarget,
    RuntimeCommitPermit,
)


class InMemoryToolTraceSink:
    module_id = "tool.trace.in_memory"

    def __init__(self) -> None:
        self._entries: defaultdict[UUID, list[ToolTraceEntry]] = defaultdict(list)

    async def record(self, event: ToolTraceEvent) -> ToolTraceEntry:
        entries = self._entries[event.invocation_id]
        if entries:
            first = entries[0].event
            if (
                first.requirement_id != event.requirement_id
                or first.capability_id != event.capability_id
                or first.provider_id != event.provider_id
                or first.correlation != event.correlation
            ):
                raise ValueError("tool trace correlation fields cannot change")
        entry = ToolTraceEntry(sequence=len(entries) + 1, event=event)
        entries.append(entry)
        return entry

    def entries_for(self, invocation_id: UUID) -> tuple[ToolTraceEntry, ...]:
        return tuple(self._entries.get(invocation_id, ()))


class ManagedToolExecutor:
    module_id = "tool.executor.managed"

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        trace_sink: ToolTraceSink,
        governor: ToolExecutionGovernor | None = None,
    ) -> None:
        self._registry = registry
        self._trace_sink = trace_sink
        self._governor = governor or OperationalToolGovernor()

    async def execute(
        self,
        invocation: ToolInvocation,
        policy: ToolExecutionPolicy,
    ) -> ToolObservation:
        started_at = utc_now()
        await self._trace(
            invocation,
            ToolTraceEventKind.EXECUTION_STARTED,
            payload={"arguments": invocation.model_dump(mode="json")["arguments"]},
        )
        try:
            metadata = self._registry.metadata_for(invocation.provider_id)
            provider = self._registry.provider_for(invocation.provider_id)
        except ProviderNotFoundError as exc:
            return await self._unavailable(invocation, started_at, str(exc))
        if metadata.capability_id != invocation.capability_id:
            return await self._unavailable(
                invocation,
                started_at,
                "provider does not implement the requested capability",
            )
        if metadata.availability is ProviderAvailability.UNAVAILABLE:
            return await self._unavailable(
                invocation,
                started_at,
                f"provider '{invocation.provider_id}' is unavailable",
            )

        guard = current_run_invocation_guard()
        if guard is not None:
            admitted = await guard.admit(
                capability_id=invocation.capability_id,
                provider_id=invocation.provider_id,
                arguments=invocation.arguments,
                exempt="polling" in metadata.tags,
            )
            if not admitted:
                return await self._finish(
                    invocation,
                    started_at,
                    (),
                    ToolExecutionStatus.POLICY_REJECTED,
                    error=(
                        "Run repeated-invocation guard rejected this Tool call "
                        "before Provider execution"
                    ),
                )

        attempts: list[ToolAttempt] = []
        while True:
            try:
                current = self._registry.metadata_for(invocation.provider_id)
            except ProviderNotFoundError as exc:
                return await self._unavailable(
                    invocation,
                    started_at,
                    str(exc),
                    attempts=tuple(attempts),
                )
            if current.availability is ProviderAvailability.UNAVAILABLE:
                return await self._unavailable(
                    invocation,
                    started_at,
                    f"provider '{invocation.provider_id}' became unavailable",
                    attempts=tuple(attempts),
                )
            attempt_number = len(attempts) + 1
            attempt_started = utc_now()
            await self._trace(
                invocation,
                ToolTraceEventKind.ATTEMPT_STARTED,
                attempt_number=attempt_number,
            )
            attempt = await self._invoke_once(
                invocation,
                provider,
                policy,
                attempt_number,
                attempt_started,
            )
            attempts.append(attempt)
            await self._trace_attempt(invocation, attempt)
            if attempt.status is ToolAttemptStatus.SUCCEEDED:
                return await self._finish(
                    invocation,
                    started_at,
                    tuple(attempts),
                    ToolExecutionStatus.SUCCEEDED,
                    output=attempt.output,
                )
            if not self._governor.should_retry(attempt, policy):
                final_status = (
                    ToolExecutionStatus.TIMED_OUT
                    if attempt.status is ToolAttemptStatus.TIMED_OUT
                    else ToolExecutionStatus.FAILED
                )
                return await self._finish(
                    invocation,
                    started_at,
                    tuple(attempts),
                    final_status,
                    error=attempt.error,
                )
            await self._trace(
                invocation,
                ToolTraceEventKind.RETRY_SCHEDULED,
                attempt_number=attempt_number,
                payload={"next_attempt": attempt_number + 1},
            )
            delay = self._governor.retry_delay(policy)
            if delay:
                await asyncio.sleep(delay)

    async def _invoke_once(
        self,
        invocation: ToolInvocation,
        provider: ToolProvider,
        policy: ToolExecutionPolicy,
        attempt_number: int,
        started_at: datetime,
    ) -> ToolAttempt:
        try:
            result = await asyncio.wait_for(
                provider.invoke(invocation),
                timeout=policy.timeout_seconds,
            )
            if not isinstance(result, ToolProviderResult):
                raise TypeError("provider must return ToolProviderResult")
            if result.succeeded:
                return ToolAttempt(
                    attempt_number=attempt_number,
                    status=ToolAttemptStatus.SUCCEEDED,
                    started_at=started_at,
                    completed_at=utc_now(),
                    output=result.output,
                )
            return ToolAttempt(
                attempt_number=attempt_number,
                status=ToolAttemptStatus.FAILED,
                started_at=started_at,
                completed_at=utc_now(),
                error=result.error,
                retryable=result.retryable,
            )
        except TimeoutError:
            return ToolAttempt(
                attempt_number=attempt_number,
                status=ToolAttemptStatus.TIMED_OUT,
                started_at=started_at,
                completed_at=utc_now(),
                error=(
                    f"provider timed out after {policy.timeout_seconds:g} seconds"
                ),
                retryable=True,
            )
        except Exception as exc:
            detail = str(exc) or exc.__class__.__name__
            return ToolAttempt(
                attempt_number=attempt_number,
                status=ToolAttemptStatus.FAILED,
                started_at=started_at,
                completed_at=utc_now(),
                error=f"{exc.__class__.__name__}: {detail}",
                retryable=True,
            )

    async def _unavailable(
        self,
        invocation: ToolInvocation,
        started_at: datetime,
        error: str,
        *,
        attempts: tuple[ToolAttempt, ...] = (),
    ) -> ToolObservation:
        await self._trace(
            invocation,
            ToolTraceEventKind.PROVIDER_UNAVAILABLE,
            payload={"error": error},
        )
        return await self._finish(
            invocation,
            started_at,
            attempts,
            ToolExecutionStatus.PROVIDER_UNAVAILABLE,
            error=error,
        )

    async def _finish(
        self,
        invocation: ToolInvocation,
        started_at: datetime,
        attempts: tuple[ToolAttempt, ...],
        status: ToolExecutionStatus,
        *,
        output: JsonValue = None,
        error: str | None = None,
    ) -> ToolObservation:
        retry_status = RetryStatus.NOT_RETRIED
        if status is ToolExecutionStatus.PROVIDER_UNAVAILABLE and attempts:
            retry_status = RetryStatus.ABORTED
        elif len(attempts) > 1:
            retry_status = (
                RetryStatus.SUCCEEDED_AFTER_RETRY
                if status is ToolExecutionStatus.SUCCEEDED
                else (
                    RetryStatus.EXHAUSTED
                    if attempts[-1].retryable
                    else RetryStatus.FAILED_AFTER_RETRY
                )
            )
        completed_at = utc_now()
        observation = ToolObservation(
            invocation_id=invocation.invocation_id,
            requirement_id=invocation.requirement_id,
            capability_id=invocation.capability_id,
            provider_id=invocation.provider_id,
            status=status,
            retry_status=retry_status,
            output=output,
            error=error,
            attempts=attempts,
            correlation=invocation.correlation,
            started_at=started_at,
            completed_at=completed_at,
        )
        await self._trace(
            invocation,
            ToolTraceEventKind.EXECUTION_FINISHED,
            payload={
                "status": status.value,
                "retry_status": retry_status.value,
                "attempt_count": len(attempts),
            },
        )
        return observation
    async def _trace_attempt(
        self,
        invocation: ToolInvocation,
        attempt: ToolAttempt,
    ) -> None:
        kind = {
            ToolAttemptStatus.SUCCEEDED: ToolTraceEventKind.ATTEMPT_SUCCEEDED,
            ToolAttemptStatus.FAILED: ToolTraceEventKind.ATTEMPT_FAILED,
            ToolAttemptStatus.TIMED_OUT: ToolTraceEventKind.ATTEMPT_TIMED_OUT,
        }[attempt.status]
        payload = cast(
            Mapping[str, JsonValue],
            attempt.model_dump(mode="json"),
        )
        await self._trace(
            invocation,
            kind,
            attempt_number=attempt.attempt_number,
            payload=payload,
        )

    async def _trace(
        self,
        invocation: ToolInvocation,
        kind: ToolTraceEventKind,
        *,
        attempt_number: int | None = None,
        payload: Mapping[str, JsonValue] | None = None,
    ) -> None:
        event = ToolTraceEvent(
            invocation_id=invocation.invocation_id,
            requirement_id=invocation.requirement_id,
            capability_id=invocation.capability_id,
            provider_id=invocation.provider_id,
            kind=kind,
            attempt_number=attempt_number,
            correlation=invocation.correlation,
            payload=payload or {},
        )
        await self._trace_sink.record(event)


class PermitBoundToolExecutor:
    """Only execute a Tool while a Runtime authorization reservation is active."""

    module_id = "tool.executor.permit_bound"

    def __init__(
        self,
        *,
        delegate: ManagedToolExecutor,
        permit_verifier: CommitPermitValidation,
    ) -> None:
        self._delegate = delegate
        self._permit_verifier = permit_verifier

    async def execute(
        self,
        invocation: ToolInvocation,
        policy: ToolExecutionPolicy,
        *,
        permit: RuntimeCommitPermit,
        target: GovernanceTarget,
        subject_fingerprint: str,
    ) -> ToolObservation:
        await self._permit_verifier.verify(
            permit,
            operation="tool.call",
            target=target,
            subject_fingerprint=subject_fingerprint,
        )
        return await self._delegate.execute(invocation, policy)


def build_permit_bound_tool_executor(
    *,
    registry: ToolRegistry,
    trace_sink: ToolTraceSink,
    permit_verifier: CommitPermitValidation,
) -> PermitBoundToolExecutor:
    """Construct the raw executor inside the Tool kernel boundary."""

    return PermitBoundToolExecutor(
        delegate=ManagedToolExecutor(
            registry=registry,
            trace_sink=trace_sink,
        ),
        permit_verifier=permit_verifier,
    )

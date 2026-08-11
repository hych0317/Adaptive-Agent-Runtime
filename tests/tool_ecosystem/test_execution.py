from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone
from typing import cast
from uuid import uuid4

from pydantic import JsonValue, ValidationError

from adaptive_agent_runtime.core.invocation import (
    RunInvocationGuard,
    bind_run_invocation_guard,
)
from adaptive_agent_runtime.tool_ecosystem import (
    Capability,
    CapabilityRequirement,
    InMemoryCapabilityCatalog,
    InMemoryToolRegistry,
    InMemoryToolTraceSink,
    ManagedToolExecutor,
    ProviderAvailability,
    RetryPolicy,
    RetryStatus,
    ToolAttempt,
    ToolAttemptStatus,
    ToolExecutionPolicy,
    ToolExecutionStatus,
    ToolInvocation,
    ToolObservation,
    ToolProvider,
    ToolProviderMetadata,
    ToolProviderOutcome,
    ToolProviderResult,
    ToolTraceEventKind,
    ToolTraceEvent,
)


class ScriptedProvider:
    module_id = "test.provider.scripted"

    def __init__(
        self,
        provider_id: str,
        outcomes: list[ToolProviderResult | Exception],
    ) -> None:
        self.provider_id = provider_id
        self._outcomes = list(outcomes)
        self.calls = 0

    async def invoke(self, invocation: ToolInvocation) -> ToolProviderResult:
        del invocation
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class SlowProvider:
    module_id = "test.provider.slow"
    provider_id = "slow"

    def __init__(self) -> None:
        self.calls = 0
        self.cancelled = False

    async def invoke(self, invocation: ToolInvocation) -> ToolProviderResult:
        del invocation
        self.calls += 1
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return ToolProviderResult.ok(output="late")


class TimeoutThenSuccessProvider:
    module_id = "test.provider.timeout_then_success"
    provider_id = "timeout_then_success"

    def __init__(self) -> None:
        self.calls = 0
        self.cancelled_calls = 0

    async def invoke(self, invocation: ToolInvocation) -> ToolProviderResult:
        del invocation
        self.calls += 1
        if self.calls == 1:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.cancelled_calls += 1
                raise
        return ToolProviderResult.ok(output="recovered after timeout")


class AvailabilityFlippingProvider:
    module_id = "test.provider.availability_flipping"
    provider_id = "availability_flipping"

    def __init__(self) -> None:
        self.calls = 0
        self.registry: InMemoryToolRegistry | None = None

    async def invoke(self, invocation: ToolInvocation) -> ToolProviderResult:
        del invocation
        self.calls += 1
        if self.registry is None:
            raise RuntimeError("test provider has no registry")
        self.registry.set_availability(
            self.provider_id,
            ProviderAvailability.UNAVAILABLE,
        )
        return ToolProviderResult.failed(error="temporary failure")


def execution_fixture(
    provider: ToolProvider,
) -> tuple[
    InMemoryToolRegistry,
    InMemoryToolTraceSink,
    ManagedToolExecutor,
    ToolInvocation,
]:
    catalog = InMemoryCapabilityCatalog()
    catalog.register(
        Capability(
            capability_id="financial_information",
            name="Financial Information",
            description="Retrieve financial information.",
        )
    )
    registry = InMemoryToolRegistry(catalog)
    registry.register(
        ToolProviderMetadata(
            provider_id=provider.provider_id,
            name=provider.provider_id,
            capability_id="financial_information",
            description="Test provider.",
            input_schema={"type": "object"},
        ),
        provider,
    )
    trace = InMemoryToolTraceSink()
    executor = ManagedToolExecutor(registry=registry, trace_sink=trace)
    invocation = ToolInvocation(
        requirement_id=uuid4(),
        capability_id="financial_information",
        provider_id=provider.provider_id,
        arguments={"symbol": "ACME"},
    )
    return registry, trace, executor, invocation


class ToolExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_execution_generates_observation_and_trace(self) -> None:
        provider = ScriptedProvider(
            "primary",
            [ToolProviderResult.ok(output={"price": 42})],
        )
        _, trace, executor, invocation = execution_fixture(provider)

        result = await executor.execute(invocation, ToolExecutionPolicy())

        self.assertEqual(result.status, ToolExecutionStatus.SUCCEEDED)
        self.assertEqual(result.retry_status, RetryStatus.NOT_RETRIED)
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.attempts[0].status, ToolAttemptStatus.SUCCEEDED)
        self.assertEqual(provider.calls, 1)
        entries = trace.entries_for(invocation.invocation_id)
        self.assertEqual(
            tuple(entry.event.kind for entry in entries),
            (
                ToolTraceEventKind.EXECUTION_STARTED,
                ToolTraceEventKind.ATTEMPT_STARTED,
                ToolTraceEventKind.ATTEMPT_SUCCEEDED,
                ToolTraceEventKind.EXECUTION_FINISHED,
            ),
        )
        self.assertEqual(
            tuple(entry.sequence for entry in entries),
            (1, 2, 3, 4),
        )
        payload = cast(dict[str, JsonValue], entries[0].event.payload)
        arguments = cast(dict[str, JsonValue], payload["arguments"])
        with self.assertRaises(TypeError):
            arguments["symbol"] = "MUTATION"

    async def test_repeated_guard_blocks_provider_before_third_call(self) -> None:
        provider = ScriptedProvider(
            "primary",
            [
                ToolProviderResult.ok(output=1),
                ToolProviderResult.ok(output=2),
                ToolProviderResult.ok(output=3),
            ],
        )
        _, _, executor, invocation = execution_fixture(provider)
        guard = RunInvocationGuard(repeated_invocation_limit=3)

        with bind_run_invocation_guard(guard):
            first = await executor.execute(invocation, ToolExecutionPolicy())
            second = await executor.execute(
                invocation.model_copy(update={"invocation_id": uuid4()}),
                ToolExecutionPolicy(),
            )
            third = await executor.execute(
                invocation.model_copy(update={"invocation_id": uuid4()}),
                ToolExecutionPolicy(),
            )

        self.assertEqual(first.status, ToolExecutionStatus.SUCCEEDED)
        self.assertEqual(second.status, ToolExecutionStatus.SUCCEEDED)
        self.assertEqual(third.status, ToolExecutionStatus.POLICY_REJECTED)
        self.assertEqual(third.attempts, ())
        self.assertEqual(provider.calls, 2)

    async def test_explicit_failure_and_exception_are_normalized(self) -> None:
        failure = ScriptedProvider(
            "failure",
            [ToolProviderResult.failed(error="provider rejected", retryable=False)],
        )
        _, _, executor, invocation = execution_fixture(failure)
        failed = await executor.execute(
            invocation,
            ToolExecutionPolicy(retry=RetryPolicy(max_retries=2)),
        )

        self.assertEqual(failed.status, ToolExecutionStatus.FAILED)
        self.assertEqual(failed.error, "provider rejected")
        self.assertEqual(failure.calls, 1)

        raising = ScriptedProvider("raising", [RuntimeError("provider exploded")])
        _, _, executor, invocation = execution_fixture(raising)
        normalized = await executor.execute(invocation, ToolExecutionPolicy())
        self.assertEqual(normalized.status, ToolExecutionStatus.FAILED)
        self.assertIn("RuntimeError: provider exploded", normalized.error or "")

    async def test_timeout_cancels_provider_and_preserves_timeout_status(self) -> None:
        provider = SlowProvider()
        _, trace, executor, invocation = execution_fixture(provider)

        result = await executor.execute(
            invocation,
            ToolExecutionPolicy(timeout_seconds=0.001),
        )

        self.assertEqual(result.status, ToolExecutionStatus.TIMED_OUT)
        self.assertTrue(provider.cancelled)
        self.assertEqual(provider.calls, 1)
        self.assertIn(
            ToolTraceEventKind.ATTEMPT_TIMED_OUT,
            tuple(
                entry.event.kind
                for entry in trace.entries_for(invocation.invocation_id)
            ),
        )

    async def test_provider_declared_timeout_uses_explicit_outcome(self) -> None:
        provider = ScriptedProvider(
            "declared-timeout",
            [
                ToolProviderResult.timed_out_result(
                    error="remote command timed out",
                    retryable=False,
                )
            ],
        )
        _, trace, executor, invocation = execution_fixture(provider)

        result = await executor.execute(invocation, ToolExecutionPolicy())

        self.assertEqual(result.status, ToolExecutionStatus.TIMED_OUT)
        self.assertEqual(result.attempts[0].status, ToolAttemptStatus.TIMED_OUT)
        self.assertEqual(provider.calls, 1)
        declared = ToolProviderResult.timed_out_result(error="timed out")
        self.assertIs(declared.outcome, ToolProviderOutcome.TIMED_OUT)
        self.assertFalse(declared.succeeded)
        self.assertTrue(declared.timed_out)
        self.assertIn(
            ToolTraceEventKind.ATTEMPT_TIMED_OUT,
            tuple(
                entry.event.kind
                for entry in trace.entries_for(invocation.invocation_id)
            ),
        )

    async def test_failure_retries_then_succeeds(self) -> None:
        provider = ScriptedProvider(
            "retry",
            [
                ToolProviderResult.failed(error="temporary"),
                ToolProviderResult.ok(output="recovered"),
            ],
        )
        _, trace, executor, invocation = execution_fixture(provider)

        result = await executor.execute(
            invocation,
            ToolExecutionPolicy(retry=RetryPolicy(max_retries=2)),
        )

        self.assertEqual(result.status, ToolExecutionStatus.SUCCEEDED)
        self.assertEqual(result.retry_status, RetryStatus.SUCCEEDED_AFTER_RETRY)
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(provider.calls, 2)
        kinds = tuple(
            entry.event.kind
            for entry in trace.entries_for(invocation.invocation_id)
        )
        self.assertEqual(
            kinds[1:-1],
            (
                ToolTraceEventKind.ATTEMPT_STARTED,
                ToolTraceEventKind.ATTEMPT_FAILED,
                ToolTraceEventKind.RETRY_SCHEDULED,
                ToolTraceEventKind.ATTEMPT_STARTED,
                ToolTraceEventKind.ATTEMPT_SUCCEEDED,
            ),
        )

    async def test_retry_exhaustion_keeps_last_failure_status(self) -> None:
        provider = ScriptedProvider(
            "exhaust",
            [
                ToolProviderResult.failed(error="one"),
                ToolProviderResult.failed(error="two"),
                ToolProviderResult.failed(error="three"),
            ],
        )
        _, trace, executor, invocation = execution_fixture(provider)

        result = await executor.execute(
            invocation,
            ToolExecutionPolicy(retry=RetryPolicy(max_retries=2)),
        )

        self.assertEqual(result.status, ToolExecutionStatus.FAILED)
        self.assertEqual(result.retry_status, RetryStatus.EXHAUSTED)
        self.assertEqual(result.error, "three")
        self.assertEqual(provider.calls, 3)
        kinds = tuple(
            entry.event.kind
            for entry in trace.entries_for(invocation.invocation_id)
        )
        self.assertEqual(kinds.count(ToolTraceEventKind.RETRY_SCHEDULED), 2)
        self.assertEqual(kinds[-1], ToolTraceEventKind.EXECUTION_FINISHED)

    async def test_timeout_retries_then_succeeds(self) -> None:
        provider = TimeoutThenSuccessProvider()
        _, trace, executor, invocation = execution_fixture(provider)

        result = await executor.execute(
            invocation,
            ToolExecutionPolicy(
                timeout_seconds=0.001,
                retry=RetryPolicy(max_retries=1),
            ),
        )

        self.assertEqual(result.status, ToolExecutionStatus.SUCCEEDED)
        self.assertEqual(result.retry_status, RetryStatus.SUCCEEDED_AFTER_RETRY)
        self.assertEqual(
            tuple(attempt.status for attempt in result.attempts),
            (ToolAttemptStatus.TIMED_OUT, ToolAttemptStatus.SUCCEEDED),
        )
        self.assertEqual(provider.calls, 2)
        self.assertEqual(provider.cancelled_calls, 1)
        self.assertEqual(
            tuple(
                entry.event.kind
                for entry in trace.entries_for(invocation.invocation_id)
            )[1:-1],
            (
                ToolTraceEventKind.ATTEMPT_STARTED,
                ToolTraceEventKind.ATTEMPT_TIMED_OUT,
                ToolTraceEventKind.RETRY_SCHEDULED,
                ToolTraceEventKind.ATTEMPT_STARTED,
                ToolTraceEventKind.ATTEMPT_SUCCEEDED,
            ),
        )

    async def test_unavailable_between_retries_preserves_attempt_history(
        self,
    ) -> None:
        provider = AvailabilityFlippingProvider()
        registry, trace, executor, invocation = execution_fixture(provider)
        provider.registry = registry

        result = await executor.execute(
            invocation,
            ToolExecutionPolicy(retry=RetryPolicy(max_retries=2)),
        )

        self.assertEqual(
            result.status,
            ToolExecutionStatus.PROVIDER_UNAVAILABLE,
        )
        self.assertEqual(result.retry_status, RetryStatus.ABORTED)
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.attempts[0].status, ToolAttemptStatus.FAILED)
        self.assertEqual(provider.calls, 1)
        entries = trace.entries_for(invocation.invocation_id)
        finished = cast(dict[str, JsonValue], entries[-1].event.payload)
        self.assertEqual(finished["attempt_count"], len(result.attempts))
        self.assertEqual(finished["status"], result.status.value)
        self.assertEqual(finished["retry_status"], result.retry_status.value)
        self.assertEqual(
            tuple(entry.sequence for entry in entries),
            tuple(range(1, len(entries) + 1)),
        )

    async def test_unavailable_provider_is_not_invoked(self) -> None:
        provider = ScriptedProvider(
            "unavailable",
            [ToolProviderResult.ok(output="must not execute")],
        )
        registry, trace, executor, invocation = execution_fixture(provider)
        registry.set_availability(
            provider.provider_id,
            ProviderAvailability.UNAVAILABLE,
        )

        result = await executor.execute(invocation, ToolExecutionPolicy())

        self.assertEqual(
            result.status,
            ToolExecutionStatus.PROVIDER_UNAVAILABLE,
        )
        self.assertEqual(result.attempts, ())
        self.assertEqual(provider.calls, 0)
        self.assertEqual(
            tuple(
                entry.event.kind
                for entry in trace.entries_for(invocation.invocation_id)
            ),
            (
                ToolTraceEventKind.EXECUTION_STARTED,
                ToolTraceEventKind.PROVIDER_UNAVAILABLE,
                ToolTraceEventKind.EXECUTION_FINISHED,
            ),
        )

    async def test_runtime_cancellation_is_not_normalized_or_retried(self) -> None:
        provider = SlowProvider()
        _, trace, executor, invocation = execution_fixture(provider)
        task = asyncio.create_task(
            executor.execute(
                invocation,
                ToolExecutionPolicy(timeout_seconds=30),
            )
        )
        await asyncio.sleep(0)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(provider.calls, 1)
        kinds = tuple(
            entry.event.kind
            for entry in trace.entries_for(invocation.invocation_id)
        )
        self.assertNotIn(ToolTraceEventKind.RETRY_SCHEDULED, kinds)


class ToolContractTests(unittest.TestCase):
    def test_observation_rejects_inconsistent_retry_status(self) -> None:
        now = datetime.now(timezone.utc)
        attempt = ToolAttempt(
            attempt_number=1,
            status=ToolAttemptStatus.SUCCEEDED,
            started_at=now,
            completed_at=now,
            output="ok",
        )

        with self.assertRaises(ValidationError):
            ToolObservation(
                invocation_id=uuid4(),
                requirement_id=uuid4(),
                capability_id="financial_information",
                provider_id="primary",
                status=ToolExecutionStatus.SUCCEEDED,
                retry_status=RetryStatus.EXHAUSTED,
                output="ok",
                attempts=(attempt,),
                started_at=now,
                completed_at=now,
            )

    def test_trace_event_kind_requires_matching_attempt_number(self) -> None:
        invocation_id = uuid4()
        requirement_id = uuid4()
        with self.assertRaises(ValidationError):
            ToolTraceEvent(
                invocation_id=invocation_id,
                requirement_id=requirement_id,
                capability_id="financial_information",
                provider_id="primary",
                kind=ToolTraceEventKind.ATTEMPT_STARTED,
            )
        with self.assertRaises(ValidationError):
            ToolTraceEvent(
                invocation_id=invocation_id,
                requirement_id=requirement_id,
                capability_id="financial_information",
                provider_id="primary",
                kind=ToolTraceEventKind.EXECUTION_STARTED,
                attempt_number=1,
            )
        with self.assertRaises(ValidationError):
            ToolTraceEvent(
                invocation_id=invocation_id,
                requirement_id=requirement_id,
                capability_id="financial_information",
                provider_id="primary",
                kind=ToolTraceEventKind.RETRY_SCHEDULED,
                attempt_number=1,
                payload={"next_attempt": 3},
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from uuid import uuid4

from adaptive_agent_runtime import (
    RunBudgetExhaustedError,
    RunStopPolicy,
    RunUsageAccountingError,
)
from adaptive_agent_runtime.core.budget import (
    RunBudgetLedger,
    bind_run_budget_ledger,
)
from adaptive_agent_runtime.llm import (
    BackendAvailability,
    BackendKind,
    BackendMetering,
    BackendProtocolError,
    BackendProbeResult,
    BackendTransportFeatures,
    BackendUnavailableError,
    DeterministicInferenceRouter,
    FakeInferenceBackend,
    InferenceExecutionBudget,
    InferenceExecutionBudgetError,
    InferenceBackendError,
    InferenceCorrelation,
    InferenceFailureCode,
    InferenceGateway,
    InferenceGatewayPolicy,
    InferenceGatewayTraceEventKind,
    InferenceRequest,
    InferenceRequirements,
    InferenceRetryPolicy,
    InferenceResponseBudgetError,
    InferenceRoutingPolicy,
    InferenceTargetAlreadyRegisteredError,
    InferenceTargetProfile,
    InferenceUsage,
    InMemoryInferenceBackendRegistry,
    InMemoryInferenceGatewayTrace,
    ManagedInferenceGateway,
    ModelResponseKind,
    NoEligibleInferenceTargetError,
    NormalizedFinishReason,
    NormalizedModelResponse,
    ProviderNeutralResponseValidator,
    ResponseSchemaValidationError,
    StructuredOutputLevel,
)


def target_profile(
    target_id: str,
    *,
    structured_output: StructuredOutputLevel = StructuredOutputLevel.NONE,
    tags: tuple[str, ...] = (),
    reports_usage: bool = False,
    reports_cost: bool = False,
    supported_capabilities: tuple[str, ...] | None = None,
) -> InferenceTargetProfile:
    return InferenceTargetProfile(
        target_id=target_id,
        backend_id=target_id.split("/")[0],
        backend_kind=BackendKind.LOCAL,
        adapter_version="1",
        model_id=f"{target_id}-model",
        features=BackendTransportFeatures(
            structured_output=structured_output,
        ),
        metering=BackendMetering(
            reports_token_usage=reports_usage,
            reports_monetary_cost=reports_cost,
        ),
        tags=tags,
        supported_cognitive_capability_ids=supported_capabilities,
    )


def response(
    request: InferenceRequest,
    profile: InferenceTargetProfile,
    *,
    output: Any = "done",
    usage: InferenceUsage | None = None,
) -> NormalizedModelResponse:
    return NormalizedModelResponse(
        request_id=request.request_id,
        target_id=profile.target_id,
        model_id=profile.model_id,
        kind=ModelResponseKind.OUTPUT,
        output=output,
        finish_reason=NormalizedFinishReason.COMPLETED,
        usage=usage or InferenceUsage(),
    )


class GatewayHarness:
    def __init__(self) -> None:
        self.registry = InMemoryInferenceBackendRegistry()
        self.trace = InMemoryInferenceGatewayTrace()
        self.gateway = ManagedInferenceGateway(
            registry=self.registry,
            router=DeterministicInferenceRouter(),
            response_validator=ProviderNeutralResponseValidator(),
            trace_sink=self.trace,
        )


class SequenceInferenceBackend:
    module_id = "test.sequence_inference_backend"

    def __init__(
        self,
        profile: InferenceTargetProfile,
        outcomes: list[NormalizedModelResponse | Exception],
    ) -> None:
        self._profile = profile
        self._outcomes = outcomes
        self.probe_count = 0
        self.invoke_count = 0

    @property
    def target_id(self) -> str:
        return self._profile.target_id

    @property
    def profile(self) -> InferenceTargetProfile:
        return self._profile

    async def probe(self) -> BackendProbeResult:
        self.probe_count += 1
        return BackendProbeResult(
            target_id=self.target_id,
            availability=BackendAvailability.AVAILABLE,
        )

    async def invoke(
        self,
        request: InferenceRequest,
    ) -> NormalizedModelResponse:
        del request
        self.invoke_count += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class BlockingInferenceBackend:
    module_id = "test.blocking_inference_backend"

    def __init__(self, profile: InferenceTargetProfile) -> None:
        self._profile = profile
        self.invoke_count = 0
        self.cancelled = False

    @property
    def target_id(self) -> str:
        return self._profile.target_id

    @property
    def profile(self) -> InferenceTargetProfile:
        return self._profile

    async def probe(self) -> BackendProbeResult:
        return BackendProbeResult(
            target_id=self.target_id,
            availability=BackendAvailability.AVAILABLE,
        )

    async def invoke(
        self,
        request: InferenceRequest,
    ) -> NormalizedModelResponse:
        del request
        self.invoke_count += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("blocking backend unexpectedly resumed")


class InferenceGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_probe_is_cached_but_force_and_expiry_refresh(
        self,
    ) -> None:
        registry = InMemoryInferenceBackendRegistry()
        trace = InMemoryInferenceGatewayTrace()
        now = [10.0]
        gateway = ManagedInferenceGateway(
            registry=registry,
            router=DeterministicInferenceRouter(),
            response_validator=ProviderNeutralResponseValidator(),
            trace_sink=trace,
            probe_cache_ttl_seconds=30.0,
            clock=lambda: now[0],
        )
        profile = target_profile("cache/target")
        backend = SequenceInferenceBackend(profile, [])
        registry.register(backend)

        first = await gateway.probe_target(profile.target_id)
        second = await gateway.probe_target(profile.target_id)

        self.assertEqual(first, second)
        self.assertEqual(backend.probe_count, 1)
        await gateway.probe_target(profile.target_id, force=True)
        self.assertEqual(backend.probe_count, 2)
        now[0] = 41.0
        await gateway.probe_target(profile.target_id)
        self.assertEqual(backend.probe_count, 3)

    async def test_successful_execution_is_routed_validated_and_traced(self) -> None:
        harness = GatewayHarness()
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
        )
        profile = target_profile("fake/default")
        expected = response(request, profile)
        harness.registry.register(
            FakeInferenceBackend(profile, {request.request_id: expected})
        )

        actual = await harness.gateway.execute(
            request,
            InferenceGatewayPolicy(),
        )

        self.assertIsInstance(harness.gateway, InferenceGateway)
        self.assertEqual(actual, expected)
        entries = harness.trace.entries(request.request_id)
        self.assertEqual(
            tuple(entry.event.kind for entry in entries),
            (
                InferenceGatewayTraceEventKind.ROUTING_COMPLETED,
                InferenceGatewayTraceEventKind.ATTEMPT_STARTED,
                InferenceGatewayTraceEventKind.ATTEMPT_SUCCEEDED,
            ),
        )
        self.assertEqual(
            tuple(entry.sequence for entry in entries),
            (1, 2, 3),
        )

    async def test_trace_preserves_and_queries_runtime_correlation(self) -> None:
        harness = GatewayHarness()
        run_id = uuid4()
        task_id = uuid4()
        node_id = uuid4()
        action_id = uuid4()
        correlation = InferenceCorrelation(
            run_id=run_id,
            task_id=task_id,
            node_id=node_id,
            action_id=action_id,
        )
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
            correlation=correlation,
            trace_attributes={
                "application": "research_agent",
                "operation": "report_generation",
            },
        )
        profile = target_profile("fake/correlated")
        harness.registry.register(
            FakeInferenceBackend(
                profile,
                {request.request_id: response(request, profile)},
            )
        )

        await harness.gateway.execute(request, InferenceGatewayPolicy())

        entries = harness.trace.entries(run_id=run_id, node_id=node_id)
        self.assertEqual(len(entries), 3)
        self.assertTrue(
            all(entry.event.correlation == correlation for entry in entries)
        )
        self.assertTrue(
            all(
                entry.event.trace_attributes
                == {
                    "application": "research_agent",
                    "operation": "report_generation",
                }
                for entry in entries
            )
        )
        self.assertEqual(harness.trace.entries(run_id=uuid4()), ())

    async def test_auth_required_target_falls_back_within_attempt_budget(self) -> None:
        harness = GatewayHarness()
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
        )
        first = target_profile("a/auth-required")
        second = target_profile("b/available")
        harness.registry.register(
            FakeInferenceBackend(
                first,
                probe_result=BackendProbeResult(
                    target_id=first.target_id,
                    availability=BackendAvailability.AUTH_REQUIRED,
                ),
            )
        )
        expected = response(request, second)
        harness.registry.register(
            FakeInferenceBackend(second, {request.request_id: expected})
        )

        actual = await harness.gateway.execute(
            request,
            InferenceGatewayPolicy(
                budget=InferenceExecutionBudget(max_attempts=2)
            ),
        )

        self.assertEqual(actual.target_id, second.target_id)
        entries = harness.trace.entries(request.request_id)
        self.assertEqual(
            tuple(
                entry.event.target_id
                for entry in entries
                if entry.event.kind
                is InferenceGatewayTraceEventKind.ATTEMPT_STARTED
            ),
            (first.target_id, second.target_id),
        )
        self.assertEqual(
            entries[2].event.kind,
            InferenceGatewayTraceEventKind.ATTEMPT_FAILED,
        )

    async def test_router_honors_feature_policy_preference_and_tags(self) -> None:
        router = DeterministicInferenceRouter()
        request = InferenceRequest(
            cognitive_capability_id="judge",
            input="judge",
            response_schema={"type": "object"},
            requirements=InferenceRequirements(
                required_structured_output=StructuredOutputLevel.JSON_SCHEMA
            ),
        )
        unsupported = target_profile("a/unsupported")
        secondary = target_profile(
            "b/secondary",
            structured_output=StructuredOutputLevel.JSON_SCHEMA,
            tags=("trusted",),
        )
        preferred = target_profile(
            "c/preferred",
            structured_output=StructuredOutputLevel.JSON_SCHEMA,
            tags=("trusted",),
        )
        policy = InferenceGatewayPolicy(
            routing=InferenceRoutingPolicy(
                preferred_target_ids=(preferred.target_id,),
                required_target_tags=("trusted",),
            )
        )

        candidates = router.candidates(
            request,
            policy,
            (unsupported, secondary, preferred),
        )

        self.assertEqual(
            tuple(item.target_id for item in candidates),
            (preferred.target_id, secondary.target_id),
        )

    async def test_router_negotiates_requested_cognitive_capability(self) -> None:
        router = DeterministicInferenceRouter()
        reasoning_only = target_profile(
            "a/reasoning",
            supported_capabilities=("reasoning",),
        )
        generation = target_profile(
            "b/generation",
            supported_capabilities=("artifact_generation",),
        )
        request = InferenceRequest(
            cognitive_capability_id="artifact_generation",
            input="generate",
        )

        candidates = router.candidates(
            request,
            InferenceGatewayPolicy(),
            (reasoning_only, generation),
        )

        self.assertEqual(candidates, (generation,))

    async def test_request_target_binding_cannot_be_overridden_by_preference(
        self,
    ) -> None:
        router = DeterministicInferenceRouter()
        bound = target_profile("a/context-bound")
        preferred = target_profile("b/preferred")
        request = InferenceRequest(
            cognitive_capability_id="generation",
            required_target_id=bound.target_id,
            input="bounded context",
        )
        candidates = router.candidates(
            request,
            InferenceGatewayPolicy(
                routing=InferenceRoutingPolicy(
                    preferred_target_ids=(preferred.target_id,),
                )
            ),
            (preferred, bound),
        )

        self.assertEqual(candidates, (bound,))

    async def test_budget_requires_metering_and_rejects_excess_usage(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
        )
        budget = InferenceExecutionBudget(
            max_total_tokens=5,
            max_response_cost=1.0,
            currency="USD",
        )

        no_metering = GatewayHarness()
        unmetered_profile = target_profile("fake/unmetered")
        no_metering.registry.register(
            FakeInferenceBackend(
                unmetered_profile,
                {
                    request.request_id: response(
                        request,
                        unmetered_profile,
                    )
                },
            )
        )
        with self.assertRaises(NoEligibleInferenceTargetError):
            await no_metering.gateway.execute(
                request,
                InferenceGatewayPolicy(budget=budget),
            )

        harness = GatewayHarness()
        metered_profile = target_profile(
            "fake/metered",
            reports_usage=True,
            reports_cost=True,
        )
        over_budget = response(
            request,
            metered_profile,
            usage=InferenceUsage(
                input_tokens=4,
                output_tokens=3,
                total_tokens=7,
                monetary_cost=2.0,
                currency="USD",
            ),
        )
        harness.registry.register(
            FakeInferenceBackend(
                metered_profile,
                {request.request_id: over_budget},
            )
        )

        with self.assertRaises(InferenceResponseBudgetError):
            await harness.gateway.execute(
                request,
                InferenceGatewayPolicy(budget=budget),
            )
        self.assertEqual(
            harness.trace.entries(request.request_id)[-1].event.kind,
            InferenceGatewayTraceEventKind.BUDGET_REJECTED,
        )

    async def test_run_budget_accumulates_across_inference_requests(self) -> None:
        harness = GatewayHarness()
        run_id = uuid4()
        requests = tuple(
            InferenceRequest(
                cognitive_capability_id="generation",
                input=f"request-{index}",
                correlation=InferenceCorrelation(run_id=run_id),
            )
            for index in range(3)
        )
        profile = target_profile(
            "fake/run-metered",
            reports_usage=True,
            reports_cost=True,
        )
        harness.registry.register(
            FakeInferenceBackend(
                profile,
                {
                    requests[0].request_id: response(
                        requests[0],
                        profile,
                        usage=InferenceUsage(
                            input_tokens=2,
                            output_tokens=2,
                            total_tokens=4,
                            monetary_cost=0.4,
                            currency="USD",
                        ),
                    ),
                    requests[1].request_id: response(
                        requests[1],
                        profile,
                        usage=InferenceUsage(
                            input_tokens=3,
                            output_tokens=3,
                            total_tokens=6,
                            monetary_cost=0.6,
                            currency="USD",
                        ),
                    ),
                },
            )
        )
        ledger = RunBudgetLedger(
            run_id=run_id,
            policy=RunStopPolicy(
                max_total_tokens=10,
                max_monetary_cost=1.0,
                currency="USD",
            ),
        )

        with bind_run_budget_ledger(ledger):
            await harness.gateway.execute(requests[0], InferenceGatewayPolicy())
            await harness.gateway.execute(requests[1], InferenceGatewayPolicy())
            with self.assertRaises(RunBudgetExhaustedError) as captured:
                await harness.gateway.execute(
                    requests[2],
                    InferenceGatewayPolicy(),
                )

        usage = await ledger.snapshot()
        self.assertEqual(usage.total_tokens, 10)
        self.assertAlmostEqual(usage.monetary_cost, 1.0)
        self.assertEqual(captured.exception.resource, "tokens")

    async def test_budgeted_run_fails_closed_when_usage_is_missing(self) -> None:
        harness = GatewayHarness()
        run_id = uuid4()
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="missing usage",
            correlation=InferenceCorrelation(run_id=run_id),
        )
        profile = target_profile(
            "fake/missing-usage",
            reports_usage=True,
        )
        harness.registry.register(
            FakeInferenceBackend(
                profile,
                {request.request_id: response(request, profile)},
            )
        )
        ledger = RunBudgetLedger(
            run_id=run_id,
            policy=RunStopPolicy(max_total_tokens=10),
        )

        with bind_run_budget_ledger(ledger):
            with self.assertRaises(RunUsageAccountingError):
                await harness.gateway.execute(
                    request,
                    InferenceGatewayPolicy(),
                )

        self.assertEqual((await ledger.snapshot()).total_tokens, 0)

    async def test_registry_rejects_duplicate_target(self) -> None:
        registry = InMemoryInferenceBackendRegistry()
        profile = target_profile("fake/default")
        backend = FakeInferenceBackend(profile)
        registry.register(backend)

        with self.assertRaises(InferenceTargetAlreadyRegisteredError):
            registry.register(backend)

    async def test_retryable_failure_is_bounded_and_can_recover(self) -> None:
        harness = GatewayHarness()
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
        )
        profile = target_profile("fake/flaky")
        expected = response(request, profile)
        failure: InferenceBackendError = BackendUnavailableError(
            profile.target_id,
            reason="transient",
            retryable=True,
        )
        harness.registry.register(
            SequenceInferenceBackend(profile, [failure, expected])
        )

        actual = await harness.gateway.execute(
            request,
            InferenceGatewayPolicy(
                budget=InferenceExecutionBudget(max_attempts=2)
            ),
        )

        self.assertEqual(actual, expected)
        kinds = tuple(
            entry.event.kind
            for entry in harness.trace.entries(request.request_id)
        )
        self.assertEqual(
            kinds.count(InferenceGatewayTraceEventKind.ATTEMPT_STARTED),
            2,
        )
        self.assertIn(InferenceGatewayTraceEventKind.ATTEMPT_FAILED, kinds)

    async def test_schema_failure_falls_back_to_next_negotiated_target(
        self,
    ) -> None:
        harness = GatewayHarness()
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
            response_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            requirements=InferenceRequirements(
                required_structured_output=StructuredOutputLevel.JSON_SCHEMA,
            ),
        )
        first = target_profile(
            "a/schema-invalid",
            structured_output=StructuredOutputLevel.JSON_SCHEMA,
        )
        second = target_profile(
            "b/schema-valid",
            structured_output=StructuredOutputLevel.JSON_SCHEMA,
        )
        first_backend = SequenceInferenceBackend(
            first,
            [response(request, first, output={"unexpected": True})],
        )
        expected = response(request, second, output={"answer": "done"})
        second_backend = SequenceInferenceBackend(second, [expected])
        harness.registry.register(first_backend)
        harness.registry.register(second_backend)

        actual = await harness.gateway.execute(
            request,
            InferenceGatewayPolicy(
                budget=InferenceExecutionBudget(max_attempts=2),
            ),
        )

        self.assertEqual(actual, expected)
        self.assertEqual(first_backend.invoke_count, 1)
        self.assertEqual(second_backend.invoke_count, 1)
        failures = tuple(
            entry.event
            for entry in harness.trace.entries(request.request_id)
            if entry.event.kind
            is InferenceGatewayTraceEventKind.ATTEMPT_FAILED
        )
        self.assertEqual(len(failures), 1)
        self.assertEqual(
            failures[0].failure_code,
            InferenceFailureCode.SCHEMA_VIOLATION,
        )

    async def test_policy_can_disable_schema_failure_fallback(self) -> None:
        harness = GatewayHarness()
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
            response_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            requirements=InferenceRequirements(
                required_structured_output=StructuredOutputLevel.JSON_SCHEMA,
            ),
        )
        first = target_profile(
            "a/schema-invalid",
            structured_output=StructuredOutputLevel.JSON_SCHEMA,
        )
        second = target_profile(
            "b/not-used",
            structured_output=StructuredOutputLevel.JSON_SCHEMA,
        )
        first_backend = SequenceInferenceBackend(
            first,
            [response(request, first, output={"unexpected": True})],
        )
        second_backend = SequenceInferenceBackend(
            second,
            [response(request, second, output={"answer": "unused"})],
        )
        harness.registry.register(first_backend)
        harness.registry.register(second_backend)

        with self.assertRaises(ResponseSchemaValidationError):
            await harness.gateway.execute(
                request,
                InferenceGatewayPolicy(
                    budget=InferenceExecutionBudget(max_attempts=2),
                    retry=InferenceRetryPolicy(fallback_failure_codes=()),
                ),
            )

        self.assertEqual(first_backend.invoke_count, 1)
        self.assertEqual(second_backend.invoke_count, 0)

    async def test_elapsed_budget_cancels_attempt_and_stops_fallback(self) -> None:
        harness = GatewayHarness()
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
        )
        first = target_profile("a/blocking")
        second = target_profile("b/not-used")
        first_backend = BlockingInferenceBackend(first)
        second_backend = SequenceInferenceBackend(
            second,
            [response(request, second)],
        )
        harness.registry.register(first_backend)
        harness.registry.register(second_backend)

        with self.assertRaises(InferenceExecutionBudgetError):
            await harness.gateway.execute(
                request,
                InferenceGatewayPolicy(
                    budget=InferenceExecutionBudget(
                        max_attempts=3,
                        max_elapsed_seconds=0.02,
                    ),
                ),
            )

        self.assertEqual(first_backend.invoke_count, 1)
        self.assertTrue(first_backend.cancelled)
        self.assertEqual(second_backend.invoke_count, 0)
        self.assertEqual(
            tuple(
                entry.event.kind
                for entry in harness.trace.entries(request.request_id)
            ),
            (
                InferenceGatewayTraceEventKind.ROUTING_COMPLETED,
                InferenceGatewayTraceEventKind.ATTEMPT_STARTED,
                InferenceGatewayTraceEventKind.BUDGET_REJECTED,
            ),
        )

    async def test_unnormalized_backend_exception_is_contained(self) -> None:
        harness = GatewayHarness()
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
        )
        profile = target_profile("fake/broken")
        harness.registry.register(
            SequenceInferenceBackend(profile, [RuntimeError("secret detail")])
        )

        with self.assertRaisesRegex(
            BackendProtocolError,
            "unnormalized RuntimeError",
        ) as captured:
            await harness.gateway.execute(request, InferenceGatewayPolicy())
        self.assertNotIn("secret detail", str(captured.exception))


if __name__ == "__main__":
    unittest.main()

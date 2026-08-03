from __future__ import annotations

import asyncio
import unittest
from uuid import uuid4

from adaptive_agent_runtime.llm import (
    AuthenticationRequiredError,
    BackendAvailability,
    BackendKind,
    BackendProbeResult,
    BackendTransportFeatures,
    BackendUnavailableError,
    CapabilityTurnKind,
    CapabilityTurnResult,
    FakeInferenceBackend,
    FakeInferenceResponseNotFoundError,
    InferenceBackend,
    InferenceContractError,
    InferenceRequest,
    InferenceRequirements,
    InferenceTargetProfile,
    ModelResponseKind,
    NormalizedFinishReason,
    NormalizedModelResponse,
    ReasoningCapability,
    ReasoningContext,
    ReasoningResult,
    StructuredOutputLevel,
    ToolIntentDraft,
    ToolIntentMode,
    ToolSpecification,
    UnsupportedFeatureError,
)


def profile(
    *,
    structured_output: StructuredOutputLevel = StructuredOutputLevel.JSON_SCHEMA,
    tool_intent: bool = True,
) -> InferenceTargetProfile:
    return InferenceTargetProfile(
        target_id="fake/default",
        backend_id="fake",
        backend_kind=BackendKind.LOCAL,
        adapter_version="1",
        model_id="fake-model",
        features=BackendTransportFeatures(
            structured_output=structured_output,
            tool_intent=tool_intent,
        ),
    )


def response(request: InferenceRequest, value: str) -> NormalizedModelResponse:
    return NormalizedModelResponse(
        request_id=request.request_id,
        target_id="fake/default",
        model_id="fake-model",
        kind=ModelResponseKind.OUTPUT,
        output={"value": value},
        finish_reason=NormalizedFinishReason.COMPLETED,
    )


class FakeInferenceBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_backend_satisfies_protocol_and_records_immutable_requests(
        self,
    ) -> None:
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="hello",
        )
        expected = response(request, "world")
        backend = FakeInferenceBackend(
            profile(),
            {request.request_id: expected},
        )

        actual = await backend.invoke(request)

        self.assertIsInstance(backend, InferenceBackend)
        self.assertEqual(actual, expected)
        self.assertEqual(backend.recorded_requests, (request,))
        self.assertIsInstance(backend.recorded_requests, tuple)
        self.assertEqual((await backend.probe()).availability, BackendAvailability.AVAILABLE)

    async def test_request_addressing_is_stable_under_concurrency(self) -> None:
        first = InferenceRequest(
            cognitive_capability_id="reasoning",
            input="first",
        )
        second = InferenceRequest(
            cognitive_capability_id="reasoning",
            input="second",
        )
        backend = FakeInferenceBackend(
            profile(),
            {
                first.request_id: response(first, "one"),
                second.request_id: response(second, "two"),
            },
        )

        second_result, first_result = await asyncio.gather(
            backend.invoke(second),
            backend.invoke(first),
        )

        self.assertEqual(second_result.output, {"value": "two"})
        self.assertEqual(first_result.output, {"value": "one"})
        self.assertCountEqual(backend.recorded_requests, (first, second))

    async def test_response_mapping_is_copied_and_entries_are_replayable(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="hello",
        )
        expected = response(request, "world")
        responses = {request.request_id: expected}
        backend = FakeInferenceBackend(profile(), responses)
        responses.clear()

        first = await backend.invoke(request)
        second = await backend.invoke(request)

        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual(backend.recorded_requests, (request, request))

    async def test_set_response_does_not_overwrite_existing_outcome(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="hello",
        )
        backend = FakeInferenceBackend(profile())
        backend.set_response(request.request_id, response(request, "first"))

        with self.assertRaisesRegex(ValueError, "already has a fake response"):
            backend.set_response(request.request_id, response(request, "second"))

    async def test_missing_response_is_explicit(self) -> None:
        backend = FakeInferenceBackend(profile())
        request = InferenceRequest(
            cognitive_capability_id="judge",
            input="subject",
        )

        with self.assertRaises(FakeInferenceResponseNotFoundError):
            await backend.invoke(request)

        self.assertEqual(backend.recorded_requests, (request,))

    async def test_response_identity_mismatch_is_not_rewritten(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="judge",
            input="subject",
        )
        wrong_request = NormalizedModelResponse(
            request_id=uuid4(),
            target_id="fake/default",
            kind=ModelResponseKind.OUTPUT,
            output="wrong",
        )
        backend = FakeInferenceBackend(
            profile(),
            {request.request_id: wrong_request},
        )
        with self.assertRaisesRegex(
            InferenceContractError,
            "different request",
        ):
            await backend.invoke(request)

        wrong_target = NormalizedModelResponse(
            request_id=request.request_id,
            target_id="another/target",
            kind=ModelResponseKind.OUTPUT,
            output="wrong",
        )
        second_backend = FakeInferenceBackend(
            profile(),
            {request.request_id: wrong_target},
        )
        with self.assertRaisesRegex(
            InferenceContractError,
            "different target",
        ):
            await second_backend.invoke(request)

        wrong_model = NormalizedModelResponse(
            request_id=request.request_id,
            target_id="fake/default",
            model_id="another-model",
            kind=ModelResponseKind.OUTPUT,
            output="wrong",
        )
        third_backend = FakeInferenceBackend(
            profile(),
            {request.request_id: wrong_model},
        )
        with self.assertRaisesRegex(
            InferenceContractError,
            "different model",
        ):
            await third_backend.invoke(request)

        missing_model = NormalizedModelResponse(
            request_id=request.request_id,
            target_id="fake/default",
            kind=ModelResponseKind.OUTPUT,
            output="wrong",
        )
        fourth_backend = FakeInferenceBackend(
            profile(),
            {request.request_id: missing_model},
        )
        with self.assertRaisesRegex(
            InferenceContractError,
            "different model",
        ):
            await fourth_backend.invoke(request)

    async def test_probe_auth_state_blocks_invocation(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="hello",
        )
        backend = FakeInferenceBackend(
            profile(),
            {request.request_id: response(request, "unused")},
            probe_result=BackendProbeResult(
                target_id="fake/default",
                availability=BackendAvailability.AUTH_REQUIRED,
            ),
        )

        with self.assertRaises(AuthenticationRequiredError):
            await backend.invoke(request)

    async def test_unavailable_probe_blocks_invocation(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="hello",
        )
        backend = FakeInferenceBackend(
            profile(),
            {request.request_id: response(request, "unused")},
            probe_result=BackendProbeResult(
                target_id="fake/default",
                availability=BackendAvailability.UNAVAILABLE,
            ),
        )

        with self.assertRaises(BackendUnavailableError) as captured:
            await backend.invoke(request)
        self.assertFalse(captured.exception.retryable)

    async def test_required_transport_features_are_enforced(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="judge",
            input="subject",
            response_schema={"type": "object"},
            requirements=InferenceRequirements(
                required_structured_output=StructuredOutputLevel.JSON_SCHEMA
            ),
        )
        backend = FakeInferenceBackend(
            profile(structured_output=StructuredOutputLevel.NONE),
            {request.request_id: response(request, "unused")},
        )

        with self.assertRaises(UnsupportedFeatureError) as captured:
            await backend.invoke(request)
        self.assertEqual(captured.exception.feature, "structured_output")

    async def test_tool_intent_feature_must_be_supported(self) -> None:
        tool = ToolSpecification(
            capability_id="search",
            name="Search",
            description="Search evidence.",
        )
        request = InferenceRequest(
            cognitive_capability_id="reasoning",
            input="find evidence",
            requirements=InferenceRequirements(
                tool_intent=ToolIntentMode.ALLOWED
            ),
            eligible_tools=(tool,),
        )
        backend = FakeInferenceBackend(
            profile(tool_intent=False),
            {request.request_id: response(request, "unused")},
        )

        with self.assertRaises(UnsupportedFeatureError) as captured:
            await backend.invoke(request)
        self.assertEqual(captured.exception.feature, "tool_intent")

    async def test_output_token_limit_is_enforced(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
            requirements=InferenceRequirements(max_output_tokens=101),
        )
        limited_profile = profile().model_copy(
            update={"limits": {"max_output_tokens": 100}}
        )
        backend = FakeInferenceBackend(
            limited_profile,
            {request.request_id: response(request, "unused")},
        )

        with self.assertRaises(UnsupportedFeatureError) as captured:
            await backend.invoke(request)
        self.assertEqual(captured.exception.feature, "max_output_tokens")

    async def test_tool_intent_must_be_enabled_and_eligible(self) -> None:
        tool = ToolSpecification(
            capability_id="search",
            name="Search",
            description="Search evidence.",
        )
        request = InferenceRequest(
            cognitive_capability_id="reasoning",
            input="find evidence",
            requirements=InferenceRequirements(
                tool_intent=ToolIntentMode.ALLOWED
            ),
            eligible_tools=(tool,),
        )
        proposed = NormalizedModelResponse(
            request_id=request.request_id,
            target_id="fake/default",
            model_id="fake-model",
            kind=ModelResponseKind.TOOL_INTENT,
            tool_intents=(
                ToolIntentDraft(
                    call_key="call-1",
                    capability_id="unapproved",
                ),
            ),
        )
        backend = FakeInferenceBackend(
            profile(),
            {request.request_id: proposed},
        )

        with self.assertRaisesRegex(
            InferenceContractError,
            "ineligible tool capabilities",
        ):
            await backend.invoke(request)

    async def test_eligible_tool_intent_is_returned_as_a_proposal(self) -> None:
        tool = ToolSpecification(
            capability_id="search",
            name="Search",
            description="Search evidence.",
        )
        request = InferenceRequest(
            cognitive_capability_id="reasoning",
            input="find evidence",
            requirements=InferenceRequirements(
                tool_intent=ToolIntentMode.ALLOWED
            ),
            eligible_tools=(tool,),
        )
        proposed = NormalizedModelResponse(
            request_id=request.request_id,
            target_id="fake/default",
            model_id="fake-model",
            kind=ModelResponseKind.TOOL_INTENT,
            tool_intents=(
                ToolIntentDraft(
                    call_key="call-1",
                    capability_id="search",
                ),
            ),
        )
        backend = FakeInferenceBackend(
            profile(),
            {request.request_id: proposed},
        )

        self.assertEqual(await backend.invoke(request), proposed)

    async def test_response_kind_must_match_requested_tool_intent_mode(
        self,
    ) -> None:
        tool = ToolSpecification(
            capability_id="search",
            name="Search",
            description="Search evidence.",
        )
        disabled_request = InferenceRequest(
            cognitive_capability_id="reasoning",
            input="answer directly",
        )
        unexpected_intent = NormalizedModelResponse(
            request_id=disabled_request.request_id,
            target_id="fake/default",
            model_id="fake-model",
            kind=ModelResponseKind.TOOL_INTENT,
            tool_intents=(
                ToolIntentDraft(
                    call_key="call-1",
                    capability_id="search",
                ),
            ),
        )
        disabled_backend = FakeInferenceBackend(
            profile(),
            {disabled_request.request_id: unexpected_intent},
        )
        with self.assertRaisesRegex(
            InferenceContractError,
            "tool intent when it was disabled",
        ):
            await disabled_backend.invoke(disabled_request)

        required_request = InferenceRequest(
            cognitive_capability_id="reasoning",
            input="search first",
            requirements=InferenceRequirements(
                tool_intent=ToolIntentMode.REQUIRED
            ),
            eligible_tools=(tool,),
        )
        required_backend = FakeInferenceBackend(
            profile(),
            {required_request.request_id: response(required_request, "wrong")},
        )
        with self.assertRaisesRegex(
            InferenceContractError,
            "output when tool intent was required",
        ):
            await required_backend.invoke(required_request)

    async def test_configured_backend_error_is_propagated(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="hello",
        )
        failure = BackendUnavailableError(
            "fake/default",
            reason="scripted failure",
        )
        backend = FakeInferenceBackend(
            profile(),
            {request.request_id: failure},
        )

        with self.assertRaisesRegex(BackendUnavailableError, "scripted failure"):
            await backend.invoke(request)

    async def test_configured_backend_error_must_belong_to_target(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="hello",
        )
        backend = FakeInferenceBackend(
            profile(),
            {
                request.request_id: BackendUnavailableError(
                    "another/target",
                    reason="scripted failure",
                )
            },
        )

        with self.assertRaisesRegex(InferenceContractError, "different target"):
            await backend.invoke(request)

    async def test_arbitrary_exception_cannot_be_configured_as_an_outcome(
        self,
    ) -> None:
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="hello",
        )

        with self.assertRaisesRegex(TypeError, "fake outcomes must be"):
            FakeInferenceBackend(
                profile(),
                {request.request_id: RuntimeError("not normalized")},  # type: ignore[dict-item]
            )


class ReasoningStub:
    module_id = "test.reasoning"
    capability_id = "reasoning"

    async def analyze(
        self,
        context: ReasoningContext,
    ) -> CapabilityTurnResult[ReasoningResult]:
        return CapabilityTurnResult(
            kind=CapabilityTurnKind.COMPLETED,
            result=ReasoningResult(conclusions=(context.goal,)),
        )


class CognitiveContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_cognitive_protocol_is_structural_and_async(self) -> None:
        capability = ReasoningStub()

        self.assertIsInstance(capability, ReasoningCapability)
        turn = await capability.analyze(ReasoningContext(goal="Analyze"))
        self.assertIsNotNone(turn.result)
        assert turn.result is not None
        self.assertEqual(turn.result.conclusions, ("Analyze",))


if __name__ == "__main__":
    unittest.main()

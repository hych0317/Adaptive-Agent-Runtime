from __future__ import annotations

import os
import unittest
from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

from pydantic import ValidationError
from pydantic import SecretStr

from adaptive_agent_runtime.llm import (
    AsyncJSONTransport,
    AuthenticationRequiredError,
    BackendKind,
    BackendRequestRejectedError,
    BackendTransportFeatures,
    CapabilityDraftValidator,
    CapabilityInferenceSettings,
    ContextOverflowError,
    DeterministicInferenceRouter,
    GatewayReasoningCapability,
    HTTPJSONResponse,
    HTTPTransportTimeoutError,
    HTTPTransportUnavailableError,
    InferenceFailureCode,
    InferenceGatewayPolicy,
    InferenceRequest,
    InferenceRequirements,
    InferenceTargetProfile,
    InferenceTimeoutError,
    InMemoryInferenceBackendRegistry,
    InMemoryInferenceGatewayTrace,
    MalformedModelOutputError,
    ManagedInferenceGateway,
    ModelResponseKind,
    NormalizedFinishReason,
    OpenAICompatibleChatBackend,
    OpenAICompatibleChatConfig,
    OpenAICompatibleProbeMode,
    ProviderNeutralResponseValidator,
    RateLimitedError,
    ReasoningContext,
    StructuredOutputLevel,
    ToolIntentMode,
    ToolSpecification,
)


class RecordingJSONTransport:
    module_id = "test.http_transport.recording"

    def __init__(
        self,
        response: HTTPJSONResponse | None = None,
        error: Exception | None = None,
        *,
        probe_response: HTTPJSONResponse | None = None,
        probe_error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.probe_response = probe_response
        self.probe_error = probe_error
        self.requests: list[tuple[str, Mapping[str, str], Mapping[str, Any], float]] = []
        self.probe_requests: list[tuple[str, Mapping[str, str], float]] = []

    async def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        self.probe_requests.append((url, dict(headers), timeout_seconds))
        if self.probe_error is not None:
            raise self.probe_error
        if self.probe_response is None:
            raise AssertionError("test transport has no probe response")
        return self.probe_response

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        self.requests.append((url, dict(headers), dict(body), timeout_seconds))
        if self.error is not None:
            raise self.error
        if self.response is None:
            raise AssertionError("test transport has no response")
        return self.response


def profile(
    *,
    structured_output: StructuredOutputLevel = StructuredOutputLevel.JSON_SCHEMA,
    tool_intent: bool = True,
) -> InferenceTargetProfile:
    return InferenceTargetProfile(
        target_id="openai-compatible/test",
        backend_id="openai-compatible",
        backend_kind=BackendKind.API,
        adapter_version="1",
        model_id="test-model",
        features=BackendTransportFeatures(
            structured_output=structured_output,
            tool_intent=tool_intent,
        ),
    )


def config(*, requires_api_key: bool = True) -> OpenAICompatibleChatConfig:
    return OpenAICompatibleChatConfig(
        base_url="https://api.example.test/v1",
        api_key_env="AAR_TEST_API_KEY",
        requires_api_key=requires_api_key,
    )


def chat_response(
    *,
    content: str | None = '{"answer":"ok"}',
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, int] | None = None,
) -> HTTPJSONResponse:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    body: dict[str, Any] = {
        "id": "chatcmpl-test",
        "model": "provider-model-version",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }
    return HTTPJSONResponse(
        status_code=200,
        headers={"x-request-id": "request-header"},
        body=body,
    )


class OpenAICompatibleBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_and_invoke_require_environment_auth(self) -> None:
        transport = RecordingJSONTransport(chat_response())
        backend = OpenAICompatibleChatBackend(profile(), config(), transport)
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
            response_schema={"type": "object"},
            requirements=InferenceRequirements(
                required_structured_output=StructuredOutputLevel.JSON_SCHEMA
            ),
        )

        with patch.dict(os.environ, {}, clear=True):
            probe = await backend.probe()
            self.assertEqual(probe.availability.value, "auth_required")
            with self.assertRaises(AuthenticationRequiredError):
                await backend.invoke(request)
        self.assertEqual(transport.requests, [])

    async def test_probe_does_not_claim_remote_connectivity(self) -> None:
        transport = RecordingJSONTransport(chat_response())
        backend = OpenAICompatibleChatBackend(
            profile(), config(requires_api_key=False), transport
        )

        probe = await backend.probe()

        self.assertEqual(probe.availability.value, "available")
        self.assertEqual(probe.diagnostics, ("connectivity_unverified",))
        self.assertEqual(transport.requests, [])

    async def test_active_probe_verifies_credentials_connectivity_and_model(
        self,
    ) -> None:
        transport = RecordingJSONTransport(
            probe_response=HTTPJSONResponse(
                status_code=200,
                body={
                    "object": "list",
                    "data": [
                        {"id": "test-model", "object": "model"},
                        {"id": "text-embedding-3-large", "object": "model"},
                        {"id": "BAAI/bge-large-en-v1.5", "object": "model"},
                    ],
                },
            )
        )
        backend = OpenAICompatibleChatBackend(
            profile(),
            OpenAICompatibleChatConfig(
                base_url="https://api.example.test/v1",
                api_key_env="AAR_TEST_API_KEY",
                probe_mode=OpenAICompatibleProbeMode.MODELS_ENDPOINT,
                probe_timeout_seconds=4.0,
            ),
            transport,
        )

        with patch.dict(os.environ, {"AAR_TEST_API_KEY": "test-secret"}):
            probe = await backend.probe()

        self.assertEqual(probe.availability.value, "available")
        self.assertEqual(probe.available_model_ids, ("test-model",))
        self.assertEqual(
            probe.diagnostics, ("connectivity_and_model_verified",)
        )
        url, headers, timeout = transport.probe_requests[0]
        self.assertEqual(url, "https://api.example.test/v1/models")
        self.assertEqual(headers["Authorization"], "Bearer test-secret")
        self.assertEqual(timeout, 4.0)
        self.assertNotIn("test-secret", repr(probe))

    async def test_reasoning_effort_is_forwarded_to_compatible_api(self) -> None:
        from adaptive_agent_runtime.llm import ReasoningEffort

        transport = RecordingJSONTransport(chat_response(content="ok"))
        backend = OpenAICompatibleChatBackend(
            profile(structured_output=StructuredOutputLevel.NONE),
            OpenAICompatibleChatConfig(
                base_url="https://api.example.test/v1",
                requires_api_key=False,
                reasoning_effort=ReasoningEffort.HIGH,
            ),
            transport,
        )

        await backend.invoke(
            InferenceRequest(cognitive_capability_id="generation", input="write")
        )

        self.assertEqual(transport.requests[0][2]["reasoning_effort"], "high")

    async def test_private_config_key_is_masked_and_environment_can_override(
        self,
    ) -> None:
        transport = RecordingJSONTransport(
            probe_response=HTTPJSONResponse(
                status_code=200,
                body={"data": [{"id": "test-model"}]},
            )
        )
        private_key = "private-config-secret"
        backend = OpenAICompatibleChatBackend(
            profile(),
            OpenAICompatibleChatConfig(
                base_url="https://api.example.test/v1",
                api_key_env="AAR_TEST_API_KEY",
                api_key=SecretStr(private_key),
                probe_mode=OpenAICompatibleProbeMode.MODELS_ENDPOINT,
            ),
            transport,
        )

        with patch.dict(os.environ, {}, clear=True):
            configured = await backend.probe()
        self.assertEqual(configured.active_auth_method, "bearer_config")
        self.assertEqual(
            transport.probe_requests[-1][1]["Authorization"],
            f"Bearer {private_key}",
        )
        self.assertNotIn(private_key, repr(backend._config))

        with patch.dict(
            os.environ,
            {"AAR_TEST_API_KEY": "temporary-override"},
            clear=True,
        ):
            overridden = await backend.probe()
        self.assertEqual(overridden.active_auth_method, "bearer_env")
        self.assertEqual(
            transport.probe_requests[-1][1]["Authorization"],
            "Bearer temporary-override",
        )

    async def test_active_probe_normalizes_remote_failures(self) -> None:
        cases = (
            (
                RecordingJSONTransport(
                    probe_response=HTTPJSONResponse(status_code=401, body={})
                ),
                "auth_required",
                "credentials_rejected",
            ),
            (
                RecordingJSONTransport(
                    probe_response=HTTPJSONResponse(status_code=429, body={})
                ),
                "unavailable",
                "probe_rate_limited",
            ),
            (
                RecordingJSONTransport(
                    probe_response=HTTPJSONResponse(
                        status_code=200,
                        body={"data": [{"id": "other-model"}]},
                    )
                ),
                "unavailable",
                "model_not_available",
            ),
            (
                RecordingJSONTransport(
                    probe_response=HTTPJSONResponse(
                        status_code=200, body={"unexpected": True}
                    )
                ),
                "unavailable",
                "invalid_models_response",
            ),
            (
                RecordingJSONTransport(
                    probe_error=HTTPTransportTimeoutError()
                ),
                "unavailable",
                "probe_timed_out",
            ),
            (
                RecordingJSONTransport(
                    probe_error=HTTPTransportUnavailableError()
                ),
                "unavailable",
                "transport_unavailable",
            ),
        )
        for transport, availability, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                backend = OpenAICompatibleChatBackend(
                    profile(),
                    OpenAICompatibleChatConfig(
                        base_url="https://api.example.test/v1",
                        requires_api_key=False,
                        probe_mode=(
                            OpenAICompatibleProbeMode.MODELS_ENDPOINT
                        ),
                    ),
                    transport,
                )
                probe = await backend.probe()
                self.assertEqual(probe.availability.value, availability)
                self.assertEqual(probe.diagnostics, (diagnostic,))

    async def test_stale_selection_still_returns_discovered_models(self) -> None:
        transport = RecordingJSONTransport(
            probe_response=HTTPJSONResponse(
                status_code=200,
                body={"data": [{"id": "other-model"}, {"id": "next-model"}]},
            )
        )
        backend = OpenAICompatibleChatBackend(
            profile(),
            OpenAICompatibleChatConfig(
                base_url="https://api.example.test/v1",
                requires_api_key=False,
                probe_mode=OpenAICompatibleProbeMode.MODELS_ENDPOINT,
            ),
            transport,
        )

        result = await backend.probe()

        self.assertEqual(result.availability.value, "unavailable")
        self.assertEqual(result.diagnostics, ("model_not_available",))
        self.assertEqual(
            result.available_model_ids,
            ("other-model", "next-model"),
        )

    async def test_structured_request_and_usage_are_normalized(self) -> None:
        transport = RecordingJSONTransport(
            chat_response(
                content='{"answer":"ok"}',
                usage={
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            )
        )
        backend = OpenAICompatibleChatBackend(profile(), config(), transport)
        request = InferenceRequest(
            cognitive_capability_id="generation.report",
            input={"instruction": "write"},
            response_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            requirements=InferenceRequirements(
                required_structured_output=StructuredOutputLevel.JSON_SCHEMA,
                max_output_tokens=100,
            ),
            timeout_seconds=12.0,
        )

        with patch.dict(os.environ, {"AAR_TEST_API_KEY": "test-secret"}):
            response = await backend.invoke(request)

        self.assertEqual(response.output, {"answer": "ok"})
        self.assertEqual(response.usage.total_tokens, 5)
        self.assertEqual(response.remote_request_id, "chatcmpl-test")
        self.assertEqual(response.model_id, "test-model")
        url, headers, body, timeout = transport.requests[0]
        self.assertEqual(url, "https://api.example.test/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer test-secret")
        self.assertNotIn("test-secret", repr(response))
        self.assertEqual(timeout, 12.0)
        self.assertEqual(body["max_tokens"], 100)
        response_format = body["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertFalse(response_format["json_schema"]["strict"])

    async def test_tool_calls_remain_runtime_proposals(self) -> None:
        transport = RecordingJSONTransport(
            chat_response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "runtime_tool_0",
                            "arguments": '{"query":"evidence"}',
                        },
                    }
                ],
            )
        )
        backend = OpenAICompatibleChatBackend(
            profile(),
            config(requires_api_key=False),
            transport,
        )
        request = InferenceRequest(
            cognitive_capability_id="reasoning",
            input="research",
            requirements=InferenceRequirements(
                tool_intent=ToolIntentMode.ALLOWED
            ),
            eligible_tools=(
                ToolSpecification(
                    capability_id="web.search/company",
                    name="Search",
                    description="Search evidence",
                    input_schema={"type": "object"},
                ),
            ),
        )

        response = await backend.invoke(request)

        self.assertEqual(response.kind, ModelResponseKind.TOOL_INTENT)
        self.assertEqual(
            response.finish_reason,
            NormalizedFinishReason.TOOL_INTENT,
        )
        self.assertEqual(
            response.tool_intents[0].capability_id,
            "web.search/company",
        )
        outbound_tools = transport.requests[0][2]["tools"]
        self.assertEqual(
            outbound_tools[0]["function"]["name"],
            "runtime_tool_0",
        )

    async def test_http_failures_are_normalized_without_provider_body_leak(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
        )
        cases = (
            (401, AuthenticationRequiredError),
            (429, RateLimitedError),
            (422, BackendRequestRejectedError),
        )
        for status, error_type in cases:
            with self.subTest(status=status):
                transport = RecordingJSONTransport(
                    HTTPJSONResponse(
                        status_code=status,
                        body={"error": {"message": "sensitive echoed prompt"}},
                    )
                )
                backend = OpenAICompatibleChatBackend(
                    profile(structured_output=StructuredOutputLevel.NONE),
                    config(requires_api_key=False),
                    transport,
                )
                with self.assertRaises(error_type) as captured:
                    await backend.invoke(request)
                self.assertNotIn("sensitive echoed prompt", str(captured.exception))

        context_transport = RecordingJSONTransport(
            HTTPJSONResponse(
                status_code=400,
                body={"error": {"code": "context_length_exceeded"}},
            )
        )
        context_backend = OpenAICompatibleChatBackend(
            profile(structured_output=StructuredOutputLevel.NONE),
            config(requires_api_key=False),
            context_transport,
        )
        with self.assertRaises(ContextOverflowError):
            await context_backend.invoke(request)

    async def test_transport_timeout_and_malformed_json_are_normalized(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
            response_schema={"type": "object"},
            requirements=InferenceRequirements(
                required_structured_output=StructuredOutputLevel.JSON_SCHEMA
            ),
        )
        timeout_backend = OpenAICompatibleChatBackend(
            profile(),
            config(requires_api_key=False),
            RecordingJSONTransport(error=HTTPTransportTimeoutError()),
        )
        with self.assertRaises(InferenceTimeoutError) as timeout:
            await timeout_backend.invoke(request)
        self.assertEqual(timeout.exception.code, InferenceFailureCode.TIMEOUT)

        malformed_backend = OpenAICompatibleChatBackend(
            profile(),
            config(requires_api_key=False),
            RecordingJSONTransport(chat_response(content="not-json")),
        )
        with self.assertRaises(MalformedModelOutputError):
            await malformed_backend.invoke(request)

    async def test_gateway_capability_runs_through_compatible_backend(self) -> None:
        transport = RecordingJSONTransport(
            chat_response(content='{"conclusions":["Integrated"]}')
        )
        backend = OpenAICompatibleChatBackend(
            profile(),
            config(requires_api_key=False),
            transport,
        )
        registry = InMemoryInferenceBackendRegistry()
        registry.register(backend)
        gateway = ManagedInferenceGateway(
            registry=registry,
            router=DeterministicInferenceRouter(),
            response_validator=ProviderNeutralResponseValidator(),
            trace_sink=InMemoryInferenceGatewayTrace(),
        )
        capability = GatewayReasoningCapability(
            gateway=gateway,
            draft_validator=CapabilityDraftValidator(),
            settings=CapabilityInferenceSettings(
                gateway_policy=InferenceGatewayPolicy(),
            ),
        )

        turn = await capability.analyze(ReasoningContext(goal="Analyze"))

        self.assertIsNotNone(turn.result)
        assert turn.result is not None
        self.assertEqual(turn.result.conclusions, ("Integrated",))
        self.assertEqual(len(transport.requests), 1)

    async def test_json_object_provider_still_gets_local_schema_validation(
        self,
    ) -> None:
        transport = RecordingJSONTransport(
            chat_response(content='{"conclusions":["Compatible"]}')
        )
        backend = OpenAICompatibleChatBackend(
            profile(structured_output=StructuredOutputLevel.JSON_OBJECT),
            config(requires_api_key=False),
            transport,
        )
        registry = InMemoryInferenceBackendRegistry()
        registry.register(backend)
        capability = GatewayReasoningCapability(
            gateway=ManagedInferenceGateway(
                registry=registry,
                router=DeterministicInferenceRouter(),
                response_validator=ProviderNeutralResponseValidator(),
                trace_sink=InMemoryInferenceGatewayTrace(),
            ),
            draft_validator=CapabilityDraftValidator(),
            settings=CapabilityInferenceSettings(
                structured_output=StructuredOutputLevel.JSON_OBJECT,
            ),
        )

        turn = await capability.analyze(ReasoningContext(goal="Analyze"))

        self.assertIsNotNone(turn.result)
        self.assertEqual(
            transport.requests[0][2]["response_format"],
            {"type": "json_object"},
        )
        user_message = transport.requests[0][2]["messages"][1]["content"]
        self.assertIn('"response_schema"', user_message)
        self.assertIn('"conclusions"', user_message)

    def test_plain_http_is_limited_to_local_or_explicit_opt_in(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "non-local HTTP",
        ):
            OpenAICompatibleChatConfig(base_url="http://example.com/v1")
        local = OpenAICompatibleChatConfig(
            base_url="http://127.0.0.1:8000/v1",
            requires_api_key=False,
        )
        self.assertEqual(
            local.completions_url,
            "http://127.0.0.1:8000/v1/chat/completions",
        )
        with self.assertRaisesRegex(
            ValidationError,
            "environment variable name",
        ):
            OpenAICompatibleChatConfig(
                base_url="https://api.example.test/v1",
                api_key_env="invalid env name",
            )

    def test_transport_protocol_is_structural(self) -> None:
        self.assertIsInstance(
            RecordingJSONTransport(chat_response()),
            AsyncJSONTransport,
        )


if __name__ == "__main__":
    unittest.main()

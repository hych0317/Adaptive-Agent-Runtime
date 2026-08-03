from __future__ import annotations

import json
import os
import unittest
from collections.abc import Mapping
from typing import Any, cast
from unittest.mock import patch

from pydantic import SecretStr, ValidationError

from adaptive_agent_runtime.llm import (
    AnthropicAPITargetDefinition,
    AnthropicMessagesBackend,
    AnthropicMessagesConfig,
    AuthenticationRequiredError,
    BackendKind,
    BackendRequestRejectedError,
    BackendTransportFeatures,
    BackendUnavailableError,
    CapabilityDraftValidator,
    CapabilityInferenceSettings,
    ContextOverflowError,
    DeterministicInferenceRouter,
    GatewayReasoningCapability,
    HTTPJSONResponse,
    HTTPTransportTimeoutError,
    HTTPTransportUnavailableError,
    InferenceRequest,
    InferenceRequirements,
    InferenceTargetProfile,
    InferenceTimeoutError,
    InMemoryInferenceBackendRegistry,
    InMemoryInferenceGatewayTrace,
    ManagedInferenceGateway,
    MalformedModelOutputError,
    ModelResponseKind,
    NormalizedFinishReason,
    ProviderNeutralResponseValidator,
    RateLimitedError,
    ReasoningContext,
    StructuredOutputLevel,
    ToolIntentMode,
    ToolSpecification,
    UnsupportedFeatureError,
)


class RecordingAnthropicTransport:
    module_id = "test.http_transport.anthropic_recording"

    def __init__(
        self,
        *,
        probe_response: HTTPJSONResponse | None = None,
        response: HTTPJSONResponse | None = None,
        probe_error: Exception | None = None,
        error: Exception | None = None,
    ) -> None:
        self.probe_response = probe_response or model_response()
        self.response = response or message_response(
            [{"type": "text", "text": '{"conclusions":["ok"]}'}]
        )
        self.probe_error = probe_error
        self.error = error
        self.gets: list[tuple[str, Mapping[str, str], float]] = []
        self.posts: list[
            tuple[str, Mapping[str, str], Mapping[str, Any], float]
        ] = []

    async def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        self.gets.append((url, dict(headers), timeout_seconds))
        if self.probe_error is not None:
            raise self.probe_error
        return self.probe_response

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        self.posts.append((url, dict(headers), dict(body), timeout_seconds))
        if self.error is not None:
            raise self.error
        return self.response


def model_response(
    *,
    status: int = 200,
    structured: bool = True,
) -> HTTPJSONResponse:
    return HTTPJSONResponse(
        status_code=status,
        body=cast(
            Any,
            {
                "id": "claude-test-model",
                "type": "model",
                "display_name": "Claude Test",
                "capabilities": {
                    "structured_outputs": {"supported": structured}
                },
            },
        ),
    )


def message_response(
    content: list[dict[str, Any]],
    *,
    status: int = 200,
    stop_reason: str = "end_turn",
    usage: Mapping[str, Any] | None = None,
) -> HTTPJSONResponse:
    return HTTPJSONResponse(
        status_code=status,
        body=cast(
            Any,
            {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test-model-20260801",
                "content": content,
                "stop_reason": stop_reason,
                "usage": usage
                or {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 2,
                    "cache_read_input_tokens": 3,
                    "output_tokens": 5,
                },
            },
        ),
    )


def profile(
    *,
    structured: StructuredOutputLevel = StructuredOutputLevel.JSON_SCHEMA,
    tool_intent: bool = True,
) -> InferenceTargetProfile:
    return InferenceTargetProfile(
        target_id="anthropic/test",
        backend_id="anthropic.messages",
        backend_kind=BackendKind.API,
        adapter_version="1",
        model_id="claude-test-model",
        features=BackendTransportFeatures(
            structured_output=structured,
            tool_intent=tool_intent,
        ),
        supported_cognitive_capability_ids=("reasoning",),
    )


def config() -> AnthropicMessagesConfig:
    return AnthropicMessagesConfig(
        base_url="https://api.anthropic.example/v1",
        api_key_env="AAR_ANTHROPIC_TEST_KEY",
        default_max_tokens=512,
    )


def reasoning_request(
    *,
    tool_intent: ToolIntentMode = ToolIntentMode.DISABLED,
) -> InferenceRequest:
    tools = (
        (
            ToolSpecification(
                capability_id="research.retrieve",
                name="Retrieve",
                description="Retrieve bounded evidence",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        )
        if tool_intent is not ToolIntentMode.DISABLED
        else ()
    )
    return InferenceRequest(
        cognitive_capability_id="reasoning",
        input={"goal": "Analyze"},
        response_schema={
            "type": "object",
            "properties": {
                "conclusions": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["conclusions"],
        },
        requirements=InferenceRequirements(
            required_structured_output=StructuredOutputLevel.JSON_SCHEMA,
            tool_intent=tool_intent,
            max_output_tokens=256,
        ),
        eligible_tools=tools,
        timeout_seconds=12.0,
    )


class AnthropicMessagesBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_requires_key_and_verifies_exact_model(self) -> None:
        transport = RecordingAnthropicTransport()
        backend = AnthropicMessagesBackend(profile(), config(), transport)
        with patch.dict(os.environ, {}, clear=True):
            missing = await backend.probe()
        self.assertEqual(missing.availability.value, "auth_required")
        self.assertEqual(missing.diagnostics, ("api_key_missing",))
        self.assertEqual(transport.gets, [])

        with patch.dict(
            os.environ, {"AAR_ANTHROPIC_TEST_KEY": "test-secret"}
        ):
            available = await backend.probe()

        self.assertEqual(available.availability.value, "available")
        self.assertEqual(
            available.diagnostics, ("connectivity_and_model_verified",)
        )
        url, headers, timeout = transport.gets[0]
        self.assertEqual(
            url,
            "https://api.anthropic.example/v1/models/claude-test-model",
        )
        self.assertEqual(headers["x-api-key"], "test-secret")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(timeout, 10.0)
        self.assertNotIn("test-secret", repr(available))

    async def test_private_config_key_is_masked_and_environment_can_override(
        self,
    ) -> None:
        private_key = "private-anthropic-secret"
        configured = AnthropicMessagesConfig(
            base_url="https://api.anthropic.example/v1",
            api_key_env="AAR_ANTHROPIC_TEST_KEY",
            api_key=SecretStr(private_key),
        )
        transport = RecordingAnthropicTransport()
        backend = AnthropicMessagesBackend(profile(), configured, transport)

        with patch.dict(os.environ, {}, clear=True):
            private_probe = await backend.probe()
        self.assertEqual(private_probe.active_auth_method, "x_api_key_config")
        self.assertEqual(transport.gets[-1][1]["x-api-key"], private_key)
        self.assertNotIn(private_key, repr(configured))

        with patch.dict(
            os.environ,
            {"AAR_ANTHROPIC_TEST_KEY": "temporary-override"},
            clear=True,
        ):
            env_probe = await backend.probe()
        self.assertEqual(env_probe.active_auth_method, "x_api_key_env")
        self.assertEqual(
            transport.gets[-1][1]["x-api-key"],
            "temporary-override",
        )

    async def test_probe_normalizes_remote_and_capability_failures(self) -> None:
        cases = (
            (RecordingAnthropicTransport(probe_response=model_response(status=401)), "auth_required", "credentials_rejected"),
            (RecordingAnthropicTransport(probe_response=model_response(status=404)), "unavailable", "model_not_available"),
            (RecordingAnthropicTransport(probe_response=model_response(status=429)), "unavailable", "probe_rate_limited"),
            (RecordingAnthropicTransport(probe_response=model_response(structured=False)), "unavailable", "structured_output_not_supported"),
            (RecordingAnthropicTransport(probe_error=HTTPTransportTimeoutError()), "unavailable", "probe_timed_out"),
            (RecordingAnthropicTransport(probe_error=HTTPTransportUnavailableError()), "unavailable", "transport_unavailable"),
        )
        for transport, availability, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                backend = AnthropicMessagesBackend(profile(), config(), transport)
                with patch.dict(
                    os.environ, {"AAR_ANTHROPIC_TEST_KEY": "key"}
                ):
                    result = await backend.probe()
                self.assertEqual(result.availability.value, availability)
                self.assertEqual(result.diagnostics, (diagnostic,))

    async def test_structured_request_and_usage_are_normalized(self) -> None:
        transport = RecordingAnthropicTransport(
            response=message_response(
                [{"type": "text", "text": '{"conclusions":["Bounded"]}'}]
            )
        )
        backend = AnthropicMessagesBackend(profile(), config(), transport)
        with patch.dict(
            os.environ, {"AAR_ANTHROPIC_TEST_KEY": "test-secret"}
        ):
            response = await backend.invoke(reasoning_request())

        self.assertEqual(response.output, {"conclusions": ("Bounded",)})
        self.assertEqual(response.usage.input_tokens, 15)
        self.assertEqual(response.usage.total_tokens, 20)
        self.assertEqual(response.remote_request_id, "msg-test")
        url, headers, body, timeout = transport.posts[0]
        self.assertEqual(url, "https://api.anthropic.example/v1/messages")
        self.assertEqual(headers["x-api-key"], "test-secret")
        self.assertNotIn("Authorization", headers)
        self.assertEqual(timeout, 12.0)
        self.assertEqual(body["model"], "claude-test-model")
        self.assertEqual(body["max_tokens"], 256)
        self.assertIsInstance(body["system"], str)
        self.assertEqual(
            body["output_config"]["format"]["type"], "json_schema"
        )

    async def test_tool_use_is_only_a_runtime_proposal(self) -> None:
        transport = RecordingAnthropicTransport(
            response=message_response(
                [
                    {"type": "text", "text": "I will request evidence."},
                    {
                        "type": "tool_use",
                        "id": "toolu-runtime-1",
                        "name": "runtime_tool_0",
                        "input": {"query": "evidence"},
                    },
                ],
                stop_reason="tool_use",
            )
        )
        backend = AnthropicMessagesBackend(profile(), config(), transport)
        request = reasoning_request(tool_intent=ToolIntentMode.ALLOWED)
        with patch.dict(
            os.environ, {"AAR_ANTHROPIC_TEST_KEY": "test-secret"}
        ):
            response = await backend.invoke(request)

        self.assertIs(response.kind, ModelResponseKind.TOOL_INTENT)
        self.assertIs(
            response.finish_reason, NormalizedFinishReason.TOOL_INTENT
        )
        self.assertEqual(
            response.tool_intents[0].capability_id, "research.retrieve"
        )
        body = transport.posts[0][2]
        self.assertEqual(body["tool_choice"], {"type": "auto"})
        self.assertEqual(body["tools"][0]["name"], "runtime_tool_0")
        self.assertTrue(body["tools"][0]["strict"])

    async def test_gateway_capability_runs_through_anthropic_backend(self) -> None:
        transport = RecordingAnthropicTransport(
            response=message_response(
                [{"type": "text", "text": '{"conclusions":["API path"]}'}]
            )
        )
        registry = InMemoryInferenceBackendRegistry()
        registry.register(AnthropicMessagesBackend(profile(), config(), transport))
        capability = GatewayReasoningCapability(
            gateway=ManagedInferenceGateway(
                registry=registry,
                router=DeterministicInferenceRouter(),
                response_validator=ProviderNeutralResponseValidator(),
                trace_sink=InMemoryInferenceGatewayTrace(),
            ),
            draft_validator=CapabilityDraftValidator(),
            settings=CapabilityInferenceSettings(
                required_target_id="anthropic/test"
            ),
        )
        with patch.dict(
            os.environ, {"AAR_ANTHROPIC_TEST_KEY": "test-secret"}
        ):
            turn = await capability.analyze(ReasoningContext(goal="Analyze"))

        self.assertIsNotNone(turn.result)
        assert turn.result is not None
        self.assertEqual(turn.result.conclusions, ("API path",))
        self.assertEqual(len(transport.gets), 1)
        self.assertEqual(len(transport.posts), 1)

    async def test_http_and_protocol_failures_are_normalized(self) -> None:
        cases: tuple[tuple[RecordingAnthropicTransport, type[Exception]], ...] = (
            (
                RecordingAnthropicTransport(
                    response=HTTPJSONResponse(status_code=401, body={})
                ),
                AuthenticationRequiredError,
            ),
            (
                RecordingAnthropicTransport(
                    response=HTTPJSONResponse(status_code=429, body={})
                ),
                RateLimitedError,
            ),
            (
                RecordingAnthropicTransport(
                    response=HTTPJSONResponse(status_code=413, body={})
                ),
                ContextOverflowError,
            ),
            (
                RecordingAnthropicTransport(
                    response=HTTPJSONResponse(status_code=529, body={})
                ),
                BackendUnavailableError,
            ),
            (
                RecordingAnthropicTransport(
                    response=HTTPJSONResponse(status_code=422, body={})
                ),
                BackendRequestRejectedError,
            ),
            (
                RecordingAnthropicTransport(error=HTTPTransportTimeoutError()),
                InferenceTimeoutError,
            ),
            (
                RecordingAnthropicTransport(
                    error=HTTPTransportUnavailableError()
                ),
                BackendUnavailableError,
            ),
            (
                RecordingAnthropicTransport(
                    response=message_response(
                        [{"type": "server_tool_use", "id": "unsafe"}]
                    )
                ),
                MalformedModelOutputError,
            ),
        )
        with patch.dict(
            os.environ, {"AAR_ANTHROPIC_TEST_KEY": "test-secret"}
        ):
            for transport, error_type in cases:
                with self.subTest(error_type=error_type):
                    backend = AnthropicMessagesBackend(
                        profile(), config(), transport
                    )
                    with self.assertRaises(error_type):
                        await backend.invoke(reasoning_request())

    async def test_unsupported_features_fail_before_transport(self) -> None:
        transport = RecordingAnthropicTransport()
        backend = AnthropicMessagesBackend(
            profile(structured=StructuredOutputLevel.NONE, tool_intent=False),
            config(),
            transport,
        )
        with patch.dict(
            os.environ, {"AAR_ANTHROPIC_TEST_KEY": "test-secret"}
        ):
            with self.assertRaises(UnsupportedFeatureError):
                await backend.invoke(reasoning_request())
        self.assertEqual(transport.posts, [])

    def test_target_and_config_contracts_fail_closed(self) -> None:
        target = AnthropicAPITargetDefinition(
            target_id="anthropic/main",
            model_id="claude-test-model",
            features=BackendTransportFeatures(
                structured_output=StructuredOutputLevel.JSON_SCHEMA
            ),
            supported_cognitive_capability_ids=("reasoning",),
        )
        built_profile = target.build_profile()
        self.assertEqual(built_profile.backend_kind, BackendKind.API)
        self.assertEqual(
            built_profile.authentication.supported_methods,
            ("x_api_key_env",),
        )
        with self.assertRaisesRegex(ValidationError, "YYYY-MM-DD"):
            AnthropicMessagesConfig(anthropic_version="latest")
        with self.assertRaisesRegex(ValidationError, "non-local"):
            AnthropicMessagesConfig(base_url="http://anthropic.internal/v1")
        with self.assertRaisesRegex(ValidationError, "multimodal"):
            AnthropicAPITargetDefinition(
                target_id="anthropic/multimodal",
                model_id="claude-test-model",
                features=BackendTransportFeatures(multimodal=True),
                supported_cognitive_capability_ids=("reasoning",),
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import unittest
from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

from pydantic import ValidationError

from adaptive_agent_runtime.llm import (
    AsyncJSONTransport,
    BackendKind,
    BackendTransportFeatures,
    HTTPJSONResponse,
    InMemoryInferenceBackendRegistry,
    OpenAICompatibleService,
    OpenAICompatibleProbeMode,
    OpenAICompatibleTargetDefinition,
    StructuredOutputLevel,
)


class NeverJSONTransport:
    module_id = "test.http_transport.never"

    async def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        del url, headers, timeout_seconds
        raise AssertionError("catalog tests must not invoke the transport")

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        del url, headers, body, timeout_seconds
        raise AssertionError("catalog tests must not invoke the transport")


def definition(
    service: OpenAICompatibleService,
    target_id: str,
    *,
    base_url: str | None = None,
) -> OpenAICompatibleTargetDefinition:
    return OpenAICompatibleTargetDefinition(
        service=service,
        target_id=target_id,
        model_id=f"{service.value}-model",
        features=BackendTransportFeatures(
            structured_output=StructuredOutputLevel.JSON_OBJECT,
        ),
        supported_cognitive_capability_ids=("reasoning",),
        base_url=base_url,
    )


class ProviderCatalogTests(unittest.IsolatedAsyncioTestCase):
    def test_service_presets_have_explicit_endpoint_auth_and_kind(self) -> None:
        cases = (
            (
                OpenAICompatibleService.OPENAI,
                "https://api.openai.com/v1/chat/completions",
                "OPENAI_API_KEY",
                BackendKind.API,
                OpenAICompatibleProbeMode.MODELS_ENDPOINT,
            ),
            (
                OpenAICompatibleService.QWEN_CHINA,
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                "DASHSCOPE_API_KEY",
                BackendKind.API,
                OpenAICompatibleProbeMode.MODELS_ENDPOINT,
            ),
            (
                OpenAICompatibleService.QWEN_INTERNATIONAL,
                "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions",
                "DASHSCOPE_API_KEY",
                BackendKind.API,
                OpenAICompatibleProbeMode.MODELS_ENDPOINT,
            ),
            (
                OpenAICompatibleService.DEEPSEEK,
                "https://api.deepseek.com/chat/completions",
                "DEEPSEEK_API_KEY",
                BackendKind.API,
                OpenAICompatibleProbeMode.MODELS_ENDPOINT,
            ),
        )
        for service, url, key_env, kind, probe_mode in cases:
            with self.subTest(service=service):
                item = definition(service, f"target/{service.value}")
                self.assertEqual(item.build_config().completions_url, url)
                self.assertEqual(item.build_config().api_key_env, key_env)
                self.assertTrue(item.build_config().requires_api_key)
                self.assertEqual(item.build_profile().backend_kind, kind)
                self.assertEqual(item.build_config().probe_mode, probe_mode)
                self.assertEqual(
                    item.build_profile().authentication.supported_methods,
                    ("bearer_env",),
                )

    def test_local_target_requires_explicit_safe_endpoint(self) -> None:
        with self.assertRaisesRegex(ValidationError, "requires a base URL"):
            definition(OpenAICompatibleService.LOCAL, "local/missing")
        local = definition(
            OpenAICompatibleService.LOCAL,
            "local/ollama",
            base_url="http://127.0.0.1:11434/v1",
        )
        self.assertEqual(local.build_profile().backend_kind, BackendKind.LOCAL)
        self.assertFalse(local.build_config().requires_api_key)
        self.assertEqual(
            local.build_config().probe_mode,
            OpenAICompatibleProbeMode.MODELS_ENDPOINT,
        )
        self.assertEqual(
            local.build_config().completions_url,
            "http://127.0.0.1:11434/v1/chat/completions",
        )
        with self.assertRaisesRegex(ValidationError, "non-local HTTP"):
            definition(
                OpenAICompatibleService.LOCAL,
                "local/unsafe",
                base_url="http://model.internal/v1",
            )

    def test_model_features_are_declared_not_inferred_from_service(self) -> None:
        item = OpenAICompatibleTargetDefinition(
            service=OpenAICompatibleService.DEEPSEEK,
            target_id="deepseek/conservative",
            model_id="deployment-model",
            features=BackendTransportFeatures(),
            supported_cognitive_capability_ids=("reasoning",),
        )
        self.assertEqual(
            item.build_profile().features.structured_output,
            StructuredOutputLevel.NONE,
        )
        self.assertFalse(item.build_profile().features.tool_intent)

    def test_catalog_definitions_register_independent_targets(self) -> None:
        registry = InMemoryInferenceBackendRegistry()
        transport = NeverJSONTransport()
        definitions = (
            definition(OpenAICompatibleService.OPENAI, "openai/main"),
            definition(OpenAICompatibleService.QWEN_CHINA, "qwen/main"),
            definition(OpenAICompatibleService.DEEPSEEK, "deepseek/main"),
            definition(
                OpenAICompatibleService.LOCAL,
                "local/main",
                base_url="http://localhost:8000/v1",
            ),
        )
        for item in definitions:
            registry.register(item.build_backend(transport))

        self.assertEqual(
            tuple(profile.target_id for profile in registry.list_profiles()),
            ("deepseek/main", "local/main", "openai/main", "qwen/main"),
        )
        self.assertIsInstance(transport, AsyncJSONTransport)

    async def test_disabled_auth_never_sends_incidental_environment_key(
        self,
    ) -> None:
        class RecordingTransport:
            module_id = "test.http_transport.recording_auth"

            def __init__(self) -> None:
                self.headers: Mapping[str, str] = {}

            async def get_json(
                self,
                url: str,
                *,
                headers: Mapping[str, str],
                timeout_seconds: float,
            ) -> HTTPJSONResponse:
                del url, headers, timeout_seconds
                return HTTPJSONResponse(
                    status_code=200,
                    body={"data": [{"id": "local-model"}]},
                )

            async def post_json(
                self,
                url: str,
                *,
                headers: Mapping[str, str],
                body: Mapping[str, Any],
                timeout_seconds: float,
            ) -> HTTPJSONResponse:
                del url, body, timeout_seconds
                self.headers = dict(headers)
                return HTTPJSONResponse(
                    status_code=200,
                    body={
                        "id": "local-response",
                        "choices": [
                            {
                                "message": {"content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                    },
                )

        transport = RecordingTransport()
        local = definition(
            OpenAICompatibleService.LOCAL,
            "local/no-auth",
            base_url="http://localhost:8000/v1",
        ).build_backend(transport)
        from adaptive_agent_runtime.llm import InferenceRequest

        with patch.dict(os.environ, {"UNUSED_API_KEY": "must-not-leak"}):
            await local.invoke(
                InferenceRequest(cognitive_capability_id="generation", input="x")
            )
        self.assertNotIn("Authorization", transport.headers)


if __name__ == "__main__":
    unittest.main()

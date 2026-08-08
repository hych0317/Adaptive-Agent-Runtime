from __future__ import annotations

import json
import stat
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from adaptive_agent_runtime.llm import (
    HTTPJSONResponse,
    InferenceGatewayPolicy,
    InferenceRequest,
    ModelResponseKind,
    NormalizedFinishReason,
    NormalizedModelResponse,
    ReasoningEffort,
    StructuredOutputLevel,
)
from applications.terminal_bench.composition import (
    _CapturingJSONTransport,
    TerminalModelConfig,
    build_terminal_model_capability,
)
from applications.terminal_bench.models import (
    TerminalExecutionPolicy,
    TerminalSessionSnapshot,
    TerminalTurnRequest,
)
from applications.terminal_bench.planner import (
    GatewayTerminalTurnProposalCapability,
    _resolve_terminal_cwd,
    _terminal_turn_response_schema,
)


class _RecordingGateway:
    module_id = "test.terminal_bench.recording_gateway"

    def __init__(self, output: Any) -> None:
        self.output = output
        self.request: InferenceRequest | None = None

    async def execute(
        self,
        request: InferenceRequest,
        policy: InferenceGatewayPolicy,
    ) -> NormalizedModelResponse:
        del policy
        self.request = request
        return NormalizedModelResponse(
            request_id=request.request_id,
            target_id="terminal-bench:deepseek:test-model",
            model_id="test-model",
            kind=ModelResponseKind.OUTPUT,
            output=self.output,
            finish_reason=NormalizedFinishReason.COMPLETED,
        )


class _StaticJSONTransport:
    module_id = "test.terminal_bench.static_json_transport"

    def __init__(self, response: HTTPJSONResponse) -> None:
        self.response = response
        self.requests: list[tuple[str, Mapping[str, Any]]] = []

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
            body={"data": [{"id": "test-model"}]},
        )

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        del headers, timeout_seconds
        self.requests.append((url, dict(body)))
        return self.response


def _turn_request() -> TerminalTurnRequest:
    return TerminalTurnRequest(
        run_id=UUID("00000000-0000-0000-0000-000000000001"),
        task_id=UUID("00000000-0000-0000-0000-000000000002"),
        instruction="Solve the task",
        session=TerminalSessionSnapshot(trial_id="trial-json-object"),
        remaining_commands=1,
        execution_semantics=("Each command is independent.",),
    )


class TerminalModelCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    def test_remote_evaluation_uses_expanded_model_budget(self) -> None:
        config = TerminalModelConfig(model_name="deepseek/test-model")

        self.assertEqual(
            config.max_output_tokens,
            32768,
        )
        self.assertEqual(config.inference_timeout_sec, 300.0)
        self.assertEqual(
            config.deepseek_reasoning_effort,
            ReasoningEffort.HIGH,
        )
        self.assertEqual(config.deepseek_thinking, "enabled")

    def test_terminal_context_is_bounded_within_total_token_budget(self) -> None:
        policy = TerminalExecutionPolicy()

        self.assertEqual(policy.max_total_tokens, 200_000)
        self.assertEqual(policy.max_context_output_characters, 6_000)
        self.assertEqual(policy.max_context_records, 8)

    def test_deepseek_uses_json_object_without_changing_other_providers(self) -> None:
        deepseek_capability, deepseek_inference = build_terminal_model_capability(
            TerminalModelConfig(model_name="deepseek/test-model")
        )
        openai_capability, openai_inference = build_terminal_model_capability(
            TerminalModelConfig(model_name="openai/test-model")
        )

        self.assertEqual(
            deepseek_inference.registry.list_profiles()[0].features.structured_output,
            StructuredOutputLevel.JSON_OBJECT,
        )
        self.assertEqual(
            openai_inference.registry.list_profiles()[0].features.structured_output,
            StructuredOutputLevel.JSON_SCHEMA,
        )
        self.assertEqual(
            deepseek_capability._required_structured_output,
            StructuredOutputLevel.JSON_OBJECT,
        )
        self.assertEqual(
            openai_capability._required_structured_output,
            StructuredOutputLevel.JSON_SCHEMA,
        )

    def test_codex_cli_uses_inference_only_json_schema_backend(self) -> None:
        capability, inference = build_terminal_model_capability(
            TerminalModelConfig(
                model_name="codex-cli/gpt-5.6-terra",
                codex_executable="codex",
                codex_reasoning_effort=ReasoningEffort.HIGH,
            )
        )

        profile = inference.registry.list_profiles()[0]
        self.assertEqual(profile.backend_kind.value, "cli")
        self.assertEqual(
            profile.target_id,
            "terminal-bench:codex-cli:gpt-5.6-terra",
        )
        self.assertEqual(
            profile.features.structured_output,
            StructuredOutputLevel.JSON_SCHEMA,
        )
        self.assertFalse(profile.features.tool_intent)
        self.assertEqual(
            capability._required_structured_output,
            StructuredOutputLevel.JSON_SCHEMA,
        )
        self.assertIsNone(capability._max_output_tokens)
        self.assertEqual(profile.limits.default_timeout_seconds, 300.0)
        self.assertTrue(capability._strict_json_schema)

    def test_codex_cli_schema_requires_all_nullable_fields(self) -> None:
        schema = _terminal_turn_response_schema(strict=True)

        self.assertNotIn("allOf", schema)
        properties = schema["properties"]
        self.assertEqual(set(schema["required"]), set(properties))
        self.assertNotIn("rationale", properties)
        process_reference = schema["$defs"]["TerminalProcessReference"]
        self.assertEqual(
            set(process_reference["required"]),
            set(process_reference["properties"]),
        )
        env_object = properties["env"]["anyOf"][0]
        self.assertEqual(env_object["properties"], {})
        self.assertEqual(env_object["required"], [])
        self.assertFalse(env_object["additionalProperties"])

    async def test_raw_response_capture_excludes_request_secrets(self) -> None:
        malformed_content = '```json\n{"decision":"complete"}\n```'
        response = HTTPJSONResponse(
            status_code=200,
            headers={"authorization": "Bearer response-secret"},
            body={
                "choices": [
                    {"message": {"content": malformed_content}},
                ]
            },
        )
        delegate = _StaticJSONTransport(response)

        with tempfile.TemporaryDirectory() as temp_dir:
            capture_path = Path(temp_dir) / "aar-model-responses.jsonl"
            transport = _CapturingJSONTransport(delegate, capture_path)

            returned = await transport.post_json(
                "https://provider.invalid/chat/completions",
                headers={"Authorization": "Bearer request-secret"},
                body={"request_secret": "must-not-be-captured"},
                timeout_seconds=1.0,
            )

            self.assertEqual(returned, response)
            records = [
                json.loads(line)
                for line in capture_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                records,
                [
                    {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {"message": {"content": malformed_content}}
                            ]
                        },
                    }
                ],
            )
            saved = capture_path.read_text(encoding="utf-8")
            self.assertNotIn("request-secret", saved)
            self.assertNotIn("response-secret", saved)
            self.assertNotIn("must-not-be-captured", saved)
            self.assertEqual(stat.S_IMODE(capture_path.stat().st_mode), 0o600)

    async def test_deepseek_high_reasoning_effort_reaches_request_body(self) -> None:
        response = HTTPJSONResponse(
            status_code=200,
            body={
                "id": "chatcmpl-test",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "decision": "complete",
                                    "summary": "Task complete",
                                }
                            )
                            + '"}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            },
        )
        transport = _StaticJSONTransport(response)
        capability, _ = build_terminal_model_capability(
            TerminalModelConfig(
                model_name="deepseek/test-model",
                api_key="test-secret",
                base_url="https://provider.invalid/v1",
            ),
            transport=transport,
        )

        proposal = await capability.propose(_turn_request())

        self.assertEqual(proposal.draft.decision.value, "complete")
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(transport.requests[0][1]["reasoning_effort"], "high")
        self.assertEqual(
            transport.requests[0][1]["thinking"],
            {"type": "enabled"},
        )

    async def test_json_object_turn_is_strictly_validated_locally(self) -> None:
        valid_gateway = _RecordingGateway(
            {
                "decision": "complete",
                "summary": "Task complete",
            }
        )
        capability = GatewayTerminalTurnProposalCapability(
            gateway=valid_gateway,
            gateway_policy=InferenceGatewayPolicy(),
            target_id="terminal-bench:deepseek:test-model",
            required_structured_output=StructuredOutputLevel.JSON_OBJECT,
        )

        proposal = await capability.propose(_turn_request())

        self.assertEqual(proposal.draft.decision.value, "complete")
        assert valid_gateway.request is not None
        self.assertEqual(
            valid_gateway.request.requirements.required_structured_output,
            StructuredOutputLevel.JSON_OBJECT,
        )
        self.assertIsNotNone(valid_gateway.request.response_schema)
        schema = valid_gateway.request.response_schema
        assert schema is not None
        self.assertNotIn("rationale", schema["properties"])
        self.assertNotIn("rationale", schema["required"])
        self.assertEqual(proposal.draft.rationale, "Model-proposed terminal turn.")
        self.assertEqual(len(schema["allOf"]), 2)
        execute_contract = schema["allOf"][0]["then"]
        self.assertEqual(
            execute_contract["required"],
            ("call_key", "command"),
        )
        instruction = valid_gateway.request.input["instruction"]
        self.assertIn("unique across payload.recent_history", instruction)
        self.assertIn("never '.' or another relative path", instruction)
        self.assertIn("not a copy of the production algorithm", instruction)
        self.assertIn("complete reported set", instruction)
        self.assertIn("Never return rationale", instruction)

    def test_terminal_cwd_is_normalized_before_docker_execution(self) -> None:
        self.assertIsNone(_resolve_terminal_cwd(".", None))
        self.assertIsNone(_resolve_terminal_cwd(None, "."))
        self.assertEqual(_resolve_terminal_cwd(".", "/app"), "/app")
        self.assertEqual(
            _resolve_terminal_cwd("generated", "/app"),
            "/app/generated",
        )
        self.assertEqual(
            _resolve_terminal_cwd("/app/../tmp", "/app"),
            "/tmp",
        )
        with self.assertRaisesRegex(ValueError, "absolute POSIX path"):
            _resolve_terminal_cwd("generated", None)

    async def test_invalid_json_object_turn_is_rejected_locally(self) -> None:
        invalid_gateway = _RecordingGateway(
            {
                "decision": "complete",
                "summary": "Task complete",
                "rationale": "The requested artifact is present.",
                "unexpected": True,
            }
        )
        invalid_capability = GatewayTerminalTurnProposalCapability(
            gateway=invalid_gateway,
            gateway_policy=InferenceGatewayPolicy(),
            target_id="terminal-bench:deepseek:test-model",
            required_structured_output=StructuredOutputLevel.JSON_OBJECT,
        )

        with self.assertRaises(ValidationError):
            await invalid_capability.propose(_turn_request())


if __name__ == "__main__":
    unittest.main()

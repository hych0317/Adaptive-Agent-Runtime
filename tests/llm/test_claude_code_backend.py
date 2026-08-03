from __future__ import annotations

import json
import os
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

from pydantic import ValidationError

from adaptive_agent_runtime.llm import (
    BackendKind,
    BackendProcessFailedError,
    BackendProtocolError,
    BackendTransportFeatures,
    CapabilityDraftValidator,
    CapabilityInferenceSettings,
    ClaudeCodeInferenceBackend,
    ClaudeCodeInferenceConfig,
    ClaudeCodeInferenceTargetDefinition,
    DeterministicInferenceRouter,
    GatewayReasoningCapability,
    InferenceRequest,
    InferenceRequirements,
    InferenceTargetProfile,
    InferenceTimeoutError,
    InMemoryInferenceBackendRegistry,
    InMemoryInferenceGatewayTrace,
    ManagedInferenceGateway,
    MalformedModelOutputError,
    ProcessResult,
    ProcessTransportTimeoutError,
    ProviderNeutralResponseValidator,
    ReasoningContext,
    StructuredOutputLevel,
    ToolIntentMode,
    UnsupportedFeatureError,
)


class RecordingClaudeTransport:
    module_id = "test.process_transport.claude_recording"

    def __init__(
        self,
        *,
        resolved: str | None = "C:/tools/claude.exe",
        version: ProcessResult | None = None,
        auth: ProcessResult | None = None,
        execution: ProcessResult | None = None,
        execution_error: Exception | None = None,
    ) -> None:
        self.resolved = resolved
        self.version = version or ProcessResult(
            exit_code=0,
            stdout="2.1.205 (Claude Code)",
        )
        self.auth = auth or ProcessResult(
            exit_code=0,
            stdout=json.dumps(
                {"loggedIn": True, "authMethod": "claude.ai"}
            ),
        )
        self.execution = execution or claude_result({"answer": "ok"})
        self.execution_error = execution_error
        self.calls: list[
            tuple[tuple[str, ...], str | None, str, Mapping[str, str], float]
        ] = []
        self.schema_during_execution: Any = None

    def resolve(self, executable: str) -> str | None:
        self.last_resolved_name = executable
        return self.resolved

    async def run(
        self,
        argv: Sequence[str],
        *,
        stdin: str | None,
        cwd: str,
        environment: Mapping[str, str],
        timeout_seconds: float,
    ) -> ProcessResult:
        arguments = tuple(argv)
        self.calls.append(
            (arguments, stdin, cwd, dict(environment), timeout_seconds)
        )
        if "--version" in arguments:
            return self.version
        if "auth" in arguments and "status" in arguments:
            return self.auth
        if self.execution_error is not None:
            raise self.execution_error
        schema_index = arguments.index("--json-schema")
        self.schema_during_execution = json.loads(arguments[schema_index + 1])
        return self.execution


def claude_result(output: Mapping[str, Any]) -> ProcessResult:
    return ProcessResult(
        exit_code=0,
        stdout=json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "structured_output": output,
                "session_id": "claude-session",
                "total_cost_usd": 0.01,
                "usage": {"input_tokens": 11, "output_tokens": 7},
            }
        ),
    )


def claude_profile() -> InferenceTargetProfile:
    return InferenceTargetProfile(
        target_id="claude-code/test",
        backend_id="claude-code.inference",
        backend_kind=BackendKind.CLI,
        adapter_version="1",
        model_id="claude-test-model",
        features=BackendTransportFeatures(
            structured_output=StructuredOutputLevel.JSON_SCHEMA,
        ),
        supported_cognitive_capability_ids=("reasoning",),
    )


def reasoning_request() -> InferenceRequest:
    return InferenceRequest(
        cognitive_capability_id="reasoning",
        input={"goal": "Analyze"},
        response_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {
                "conclusion": {"type": "string", "minLength": 1}
            },
            "type": "object",
            "properties": {
                "conclusions": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/conclusion"},
                }
            },
            "required": ["conclusions"],
        },
        requirements=InferenceRequirements(
            required_structured_output=StructuredOutputLevel.JSON_SCHEMA
        ),
    )


class ClaudeCodeInferenceBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_reports_version_and_cached_authentication(self) -> None:
        transport = RecordingClaudeTransport()
        backend = ClaudeCodeInferenceBackend(
            claude_profile(), ClaudeCodeInferenceConfig(), transport
        )

        result = await backend.probe()

        self.assertEqual(result.availability.value, "available")
        self.assertEqual(result.runtime_version, "2.1.205")
        self.assertEqual(result.active_auth_method, "claude.ai")
        self.assertEqual(result.protocol_version, "claude-code-json-v1")
        self.assertEqual(len(transport.calls), 2)

    async def test_probe_fails_closed_for_missing_old_or_signed_out_cli(
        self,
    ) -> None:
        missing_transport = RecordingClaudeTransport(resolved=None)
        missing = ClaudeCodeInferenceBackend(
            claude_profile(), ClaudeCodeInferenceConfig(), missing_transport
        )
        missing_result = await missing.probe()
        self.assertEqual(missing_result.availability.value, "unavailable")
        self.assertEqual(
            missing_result.diagnostics, ("executable_not_found",)
        )

        old = ClaudeCodeInferenceBackend(
            claude_profile(),
            ClaudeCodeInferenceConfig(),
            RecordingClaudeTransport(
                version=ProcessResult(exit_code=0, stdout="2.1.204")
            ),
        )
        old_result = await old.probe()
        self.assertEqual(old_result.availability.value, "unavailable")
        self.assertEqual(
            old_result.diagnostics, ("unsupported_runtime_version",)
        )

        signed_out = ClaudeCodeInferenceBackend(
            claude_profile(),
            ClaudeCodeInferenceConfig(),
            RecordingClaudeTransport(
                auth=ProcessResult(exit_code=1, stderr="private detail")
            ),
        )
        signed_out_result = await signed_out.probe()
        self.assertEqual(signed_out_result.availability.value, "auth_required")
        self.assertNotIn("private detail", repr(signed_out_result))

    async def test_invoke_is_inference_only_and_normalized(self) -> None:
        transport = RecordingClaudeTransport(
            execution=claude_result({"conclusions": ["Bounded"]})
        )
        backend = ClaudeCodeInferenceBackend(
            claude_profile(),
            ClaudeCodeInferenceConfig(
                inherited_environment_variables=("AAR_SAFE_TEST",)
            ),
            transport,
        )
        with patch.dict(
            os.environ,
            {"AAR_SAFE_TEST": "allowed", "AAR_SECRET_TEST": "secret"},
            clear=True,
        ):
            response = await backend.invoke(reasoning_request())

        self.assertEqual(response.output, {"conclusions": ("Bounded",)})
        self.assertEqual(response.usage.total_tokens, 18)
        self.assertEqual(response.usage.monetary_cost, 0.01)
        self.assertEqual(response.remote_request_id, "claude-session")
        argv, prompt, workspace, environment, _ = transport.calls[-1]
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertEqual(argv[argv.index("--disallowedTools") + 1], "*")
        self.assertEqual(argv[argv.index("--max-turns") + 1], "1")
        self.assertEqual(
            argv[argv.index("--permission-mode") + 1], "dontAsk"
        )
        for flag in (
            "--strict-mcp-config",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--no-chrome",
            "--bare",
            "--safe-mode",
        ):
            self.assertIn(flag, argv)
        self.assertEqual(environment, {"AAR_SAFE_TEST": "allowed"})
        self.assertNotIn("secret", prompt or "")
        self.assertFalse(Path(workspace).exists())
        schema = transport.schema_during_execution
        self.assertNotIn("$defs", schema)
        self.assertIn("definitions", schema)
        self.assertEqual(
            schema["properties"]["conclusions"]["items"]["$ref"],
            "#/definitions/conclusion",
        )

    async def test_gateway_capability_uses_claude_inference_backend(self) -> None:
        transport = RecordingClaudeTransport(
            execution=claude_result({"conclusions": ["OAuth path"]})
        )
        registry = InMemoryInferenceBackendRegistry()
        registry.register(
            ClaudeCodeInferenceBackend(
                claude_profile(), ClaudeCodeInferenceConfig(), transport
            )
        )
        capability = GatewayReasoningCapability(
            gateway=ManagedInferenceGateway(
                registry=registry,
                router=DeterministicInferenceRouter(),
                response_validator=ProviderNeutralResponseValidator(),
                trace_sink=InMemoryInferenceGatewayTrace(),
            ),
            draft_validator=CapabilityDraftValidator(),
            settings=CapabilityInferenceSettings(
                required_target_id="claude-code/test"
            ),
        )

        turn = await capability.analyze(ReasoningContext(goal="Analyze"))

        self.assertIsNotNone(turn.result)
        assert turn.result is not None
        self.assertEqual(turn.result.conclusions, ("OAuth path",))
        self.assertEqual(len(transport.calls), 3)

    async def test_unsupported_requests_and_schema_fail_before_execution(
        self,
    ) -> None:
        transport = RecordingClaudeTransport()
        backend = ClaudeCodeInferenceBackend(
            claude_profile(), ClaudeCodeInferenceConfig(), transport
        )
        for requirements in (
            InferenceRequirements(tool_intent=ToolIntentMode.ALLOWED),
            InferenceRequirements(max_output_tokens=100),
        ):
            request = reasoning_request().model_copy(
                update={"requirements": requirements}
            )
            with self.assertRaises(UnsupportedFeatureError):
                await backend.invoke(request)

        unsupported_schema = reasoning_request().model_copy(
            update={
                "response_schema": {
                    "type": "array",
                    "prefixItems": [{"type": "string"}],
                }
            }
        )
        with self.assertRaisesRegex(
            UnsupportedFeatureError, "schema_keyword:prefixItems"
        ):
            await backend.invoke(unsupported_schema)
        self.assertEqual(transport.calls, [])

    async def test_process_and_malformed_outputs_are_normalized(self) -> None:
        request = reasoning_request()
        timeout = ClaudeCodeInferenceBackend(
            claude_profile(),
            ClaudeCodeInferenceConfig(),
            RecordingClaudeTransport(
                execution_error=ProcessTransportTimeoutError()
            ),
        )
        with self.assertRaises(InferenceTimeoutError):
            await timeout.invoke(request)

        failed = ClaudeCodeInferenceBackend(
            claude_profile(),
            ClaudeCodeInferenceConfig(),
            RecordingClaudeTransport(
                execution=ProcessResult(
                    exit_code=9, stderr="sensitive provider failure"
                )
            ),
        )
        with self.assertRaises(BackendProcessFailedError) as failure:
            await failed.invoke(request)
        self.assertNotIn("sensitive provider failure", str(failure.exception))

        for output, expected_error in (
            (ProcessResult(exit_code=0, stdout="not-json"), MalformedModelOutputError),
            (
                ProcessResult(
                    exit_code=0,
                    stdout=json.dumps(
                        {"type": "result", "subtype": "error"}
                    ),
                ),
                BackendProtocolError,
            ),
            (
                ProcessResult(
                    exit_code=0,
                    stdout=json.dumps(
                        {
                            "type": "result",
                            "subtype": "success",
                            "usage": {"input_tokens": "eleven"},
                            "structured_output": {"conclusions": ["x"]},
                        }
                    ),
                ),
                MalformedModelOutputError,
            ),
        ):
            malformed = ClaudeCodeInferenceBackend(
                claude_profile(),
                ClaudeCodeInferenceConfig(),
                RecordingClaudeTransport(execution=output),
            )
            with self.assertRaises(expected_error):
                await malformed.invoke(request)

    def test_definition_and_configuration_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValidationError, "cannot expose ToolIntent"):
            ClaudeCodeInferenceTargetDefinition(
                target_id="claude/test",
                model_id="claude-test",
                features=BackendTransportFeatures(
                    structured_output=StructuredOutputLevel.JSON_SCHEMA,
                    tool_intent=True,
                ),
                supported_cognitive_capability_ids=("reasoning",),
            )
        with self.assertRaisesRegex(ValidationError, "require JSON Schema"):
            ClaudeCodeInferenceTargetDefinition(
                target_id="claude/test",
                model_id="claude-test",
                features=BackendTransportFeatures(
                    structured_output=StructuredOutputLevel.JSON_OBJECT,
                ),
                supported_cognitive_capability_ids=("reasoning",),
            )
        with self.assertRaisesRegex(ValidationError, "must be unique"):
            ClaudeCodeInferenceConfig(
                inherited_environment_variables=("PATH", "PATH")
            )


if __name__ == "__main__":
    unittest.main()

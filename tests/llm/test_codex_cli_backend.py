from __future__ import annotations

import json
import os
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from pydantic import ValidationError

from adaptive_agent_runtime.llm import (
    AsyncProcessTransport,
    AuthenticationRequiredError,
    BackendKind,
    BackendProcessFailedError,
    BackendProtocolError,
    BackendTransportFeatures,
    CapabilityDraftValidator,
    CapabilityInferenceSettings,
    CodexCLIAuthProbeMode,
    CodexCLIInferenceBackend,
    CodexCLIInferenceConfig,
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
    SubprocessTransport,
    ToolIntentMode,
    UnsupportedFeatureError,
)
from adaptive_agent_runtime.llm.providers.cli_integration import (
    resolve_codex_native_binary,
)


class RecordingProcessTransport:
    module_id = "test.process_transport.recording"

    def __init__(
        self,
        *,
        resolved: str | None = "C:/tools/codex.exe",
        version: ProcessResult | None = None,
        login: ProcessResult | None = None,
        models: ProcessResult | None = None,
        execution: ProcessResult | None = None,
        version_error: Exception | None = None,
        login_error: Exception | None = None,
        execution_error: Exception | None = None,
    ) -> None:
        self.resolved = resolved
        self.version = version or ProcessResult(
            exit_code=0,
            stdout="codex-cli 1.2.3",
        )
        self.login = login or ProcessResult(
            exit_code=0,
            stdout="Logged in using ChatGPT",
        )
        self.models = models or ProcessResult(
            exit_code=0,
            stdout=json.dumps(
                {
                    "models": [
                        {
                            "slug": "gpt-test",
                            "display_name": "GPT Test",
                            "visibility": "list",
                        }
                    ]
                }
            ),
        )
        self.execution = execution or ProcessResult(
            exit_code=0,
            stdout=codex_jsonl('{"answer":"ok"}'),
        )
        self.version_error = version_error
        self.login_error = login_error
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
        snapshot = (tuple(argv), stdin, cwd, dict(environment), timeout_seconds)
        self.calls.append(snapshot)
        if "--version" in argv:
            if self.version_error is not None:
                raise self.version_error
            return self.version
        if "login" in argv and "status" in argv:
            if self.login_error is not None:
                raise self.login_error
            return self.login
        if "debug" in argv and "models" in argv:
            return self.models
        if self.execution_error is not None:
            raise self.execution_error
        if "--output-schema" in argv:
            index = argv.index("--output-schema")
            self.schema_during_execution = json.loads(
                Path(argv[index + 1]).read_text(encoding="utf-8")
            )
        return self.execution


def codex_jsonl(
    final_text: str,
    *,
    item_type: str = "agent_message",
) -> str:
    events = (
        {"type": "thread.started", "thread_id": "thread-test"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {
                "id": "item-1",
                "type": item_type,
                "text": final_text,
            },
        },
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 11, "output_tokens": 7},
        },
    )
    return "\n".join(json.dumps(event) for event in events)


def cli_profile(
    *,
    tool_intent: bool = False,
) -> InferenceTargetProfile:
    return InferenceTargetProfile(
        target_id="codex-cli/test",
        backend_id="codex-cli",
        backend_kind=BackendKind.CLI,
        adapter_version="1",
        model_id="test-model",
        features=BackendTransportFeatures(
            structured_output=StructuredOutputLevel.JSON_SCHEMA,
            tool_intent=tool_intent,
        ),
    )


def reasoning_request() -> InferenceRequest:
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
            required_structured_output=StructuredOutputLevel.JSON_SCHEMA
        ),
    )


class CodexCLIInferenceBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_reports_executable_auth_and_version(self) -> None:
        transport = RecordingProcessTransport()
        backend = CodexCLIInferenceBackend(
            cli_profile(),
            CodexCLIInferenceConfig(),
            transport,
        )

        result = await backend.probe()

        self.assertEqual(result.availability.value, "available")
        self.assertEqual(result.runtime_version, "1.2.3")
        self.assertEqual(result.active_auth_method, "chatgpt_oauth")
        self.assertEqual(result.protocol_version, "codex-exec-jsonl-v1")
        self.assertEqual(result.available_model_ids, ("default", "gpt-test"))
        self.assertEqual(len(transport.calls), 3)

    async def test_open_design_version_probe_failure_semantics_are_preserved(
        self,
    ) -> None:
        generic_failure = RecordingProcessTransport(
            version=ProcessResult(exit_code=1, stderr="unsupported flag"),
        )
        generic_result = await CodexCLIInferenceBackend(
            cli_profile(),
            CodexCLIInferenceConfig(),
            generic_failure,
        ).probe()

        self.assertEqual(generic_result.availability.value, "available")
        self.assertIsNone(generic_result.runtime_version)
        self.assertIn("version_unverified", generic_result.diagnostics)
        self.assertEqual(len(generic_failure.calls), 3)

        timed_out = RecordingProcessTransport(
            version_error=ProcessTransportTimeoutError(),
        )
        timeout_result = await CodexCLIInferenceBackend(
            cli_profile(),
            CodexCLIInferenceConfig(),
            timed_out,
        ).probe()

        self.assertEqual(timeout_result.availability.value, "available")
        self.assertIn("version_unverified", timeout_result.diagnostics)
        self.assertEqual(len(timed_out.calls), 3)

        stale_wrapper = RecordingProcessTransport(
            version=ProcessResult(exit_code=127, stderr="target missing"),
        )
        stale_result = await CodexCLIInferenceBackend(
            cli_profile(),
            CodexCLIInferenceConfig(),
            stale_wrapper,
        ).probe()
        self.assertEqual(stale_result.availability.value, "unavailable")
        self.assertEqual(len(stale_wrapper.calls), 1)

    async def test_probe_separates_binary_availability_from_auth_confidence(
        self,
    ) -> None:
        missing = RecordingProcessTransport(resolved=None)
        missing_backend = CodexCLIInferenceBackend(
            cli_profile(),
            CodexCLIInferenceConfig(),
            missing,
        )
        missing_result = await missing_backend.probe()
        self.assertEqual(missing_result.availability.value, "unavailable")
        self.assertEqual(missing_result.diagnostics, ("executable_not_found",))
        self.assertEqual(missing.calls, [])

        signed_out = RecordingProcessTransport(
            login=ProcessResult(exit_code=1, stderr="private account detail")
        )
        signed_out_backend = CodexCLIInferenceBackend(
            cli_profile(),
            CodexCLIInferenceConfig(),
            signed_out,
        )
        signed_out_result = await signed_out_backend.probe()
        self.assertEqual(signed_out_result.availability.value, "available")
        self.assertEqual(
            signed_out_result.diagnostics,
            ("authentication_unverified",),
        )
        self.assertNotIn("private account detail", repr(signed_out_result))

        explicitly_missing = RecordingProcessTransport(
            login=ProcessResult(exit_code=1, stderr="Not logged in")
        )
        explicitly_missing_result = await CodexCLIInferenceBackend(
            cli_profile(),
            CodexCLIInferenceConfig(),
            explicitly_missing,
        ).probe()
        self.assertEqual(
            explicitly_missing_result.availability.value,
            "available",
        )
        self.assertEqual(
            explicitly_missing_result.diagnostics,
            ("authentication_missing",),
        )

        strict_backend = CodexCLIInferenceBackend(
            cli_profile(),
            CodexCLIInferenceConfig(
                auth_probe_mode=CodexCLIAuthProbeMode.REQUIRED,
            ),
            signed_out,
        )
        strict_result = await strict_backend.probe()
        self.assertEqual(strict_result.availability.value, "auth_required")
        self.assertEqual(strict_result.diagnostics, ("login_required",))

    async def test_invoke_is_hardened_audited_and_normalized(self) -> None:
        transport = RecordingProcessTransport(
            execution=ProcessResult(
                exit_code=0,
                stdout=codex_jsonl('{"conclusions":["Bounded"]}'),
            )
        )
        backend = CodexCLIInferenceBackend(
            cli_profile(),
            CodexCLIInferenceConfig(
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
        self.assertEqual(response.remote_request_id, "thread-test")
        argv, prompt, workspace, environment, _ = transport.calls[-1]
        self.assertIn("--ephemeral", argv)
        self.assertIn("--ignore-user-config", argv)
        self.assertIn("--ignore-rules", argv)
        self.assertEqual(argv[argv.index("--sandbox") + 1], "read-only")
        self.assertEqual(argv[argv.index("--ask-for-approval") + 1], "never")
        self.assertEqual(environment["AAR_SAFE_TEST"], "allowed")
        self.assertIn("PATH", environment)
        self.assertIn("C:/tools", environment["PATH"].replace("\\", "/"))
        self.assertNotIn("secret", prompt or "")
        self.assertFalse(Path(workspace).exists())
        self.assertEqual(
            transport.schema_during_execution["required"],
            ["conclusions"],
        )
        self.assertNotEqual(argv[-1], "-")

    async def test_codex_api_key_short_circuits_login_status_probe(self) -> None:
        transport = RecordingProcessTransport(
            login=ProcessResult(exit_code=1, stderr="Not logged in"),
        )
        backend = CodexCLIInferenceBackend(
            cli_profile(),
            CodexCLIInferenceConfig(),
            transport,
        )

        with patch.dict(os.environ, {"CODEX_API_KEY": "test-key"}, clear=True):
            result = await backend.probe()

        self.assertEqual(result.availability.value, "available")
        self.assertEqual(result.active_auth_method, "api_key")
        self.assertEqual(len(transport.calls), 2)

    def test_codex_wrapper_is_upgraded_to_packaged_native_binary(self) -> None:
        with TemporaryDirectory(prefix="aar-codex-native-test-") as temporary:
            root = Path(temporary)
            wrapper = root / "bin" / "codex.cmd"
            wrapper.parent.mkdir(parents=True)
            wrapper.write_text("@echo off\r\n", encoding="utf-8")
            native = (
                root
                / "node_modules"
                / "@openai"
                / "codex"
                / "node_modules"
                / "@openai"
                / "codex-win32-x64"
                / "vendor"
                / "x86_64-pc-windows-msvc"
                / "bin"
                / "codex.exe"
            )
            native.parent.mkdir(parents=True)
            native.write_bytes(b"fixture")

            resolved = resolve_codex_native_binary(
                str(wrapper),
                platform_name="win32",
                machine="AMD64",
            )

        self.assertEqual(resolved, str(native))

    async def test_internal_action_event_is_rejected(self) -> None:
        transport = RecordingProcessTransport(
            execution=ProcessResult(
                exit_code=0,
                stdout=codex_jsonl("ignored", item_type="command_execution"),
            )
        )
        backend = CodexCLIInferenceBackend(
            cli_profile(),
            CodexCLIInferenceConfig(),
            transport,
        )

        with self.assertRaisesRegex(
            BackendProtocolError,
            "attempted an internal action",
        ):
            await backend.invoke(reasoning_request())

    async def test_timeout_process_failure_and_malformed_output_are_normalized(
        self,
    ) -> None:
        request = reasoning_request()
        timeout_backend = CodexCLIInferenceBackend(
            cli_profile(),
            CodexCLIInferenceConfig(),
            RecordingProcessTransport(
                execution_error=ProcessTransportTimeoutError()
            ),
        )
        with self.assertRaises(InferenceTimeoutError):
            await timeout_backend.invoke(request)

        failed_backend = CodexCLIInferenceBackend(
            cli_profile(),
            CodexCLIInferenceConfig(),
            RecordingProcessTransport(
                execution=ProcessResult(
                    exit_code=9,
                    stderr="sensitive provider failure",
                )
            ),
        )
        with self.assertRaises(BackendProcessFailedError) as failed:
            await failed_backend.invoke(request)
        self.assertNotIn("sensitive provider failure", str(failed.exception))

        unauthenticated_backend = CodexCLIInferenceBackend(
            cli_profile(),
            CodexCLIInferenceConfig(),
            RecordingProcessTransport(
                execution=ProcessResult(
                    exit_code=1,
                    stderr="Not logged in; please log in",
                )
            ),
        )
        with self.assertRaises(AuthenticationRequiredError):
            await unauthenticated_backend.invoke(request)

        malformed_backend = CodexCLIInferenceBackend(
            cli_profile(),
            CodexCLIInferenceConfig(),
            RecordingProcessTransport(
                execution=ProcessResult(exit_code=0, stdout="not-jsonl")
            ),
        )
        with self.assertRaises(MalformedModelOutputError):
            await malformed_backend.invoke(request)

    async def test_gateway_capability_runs_through_codex_cli_backend(self) -> None:
        transport = RecordingProcessTransport(
            execution=ProcessResult(
                exit_code=0,
                stdout=codex_jsonl('{"conclusions":["OAuth path"]}'),
            )
        )
        registry = InMemoryInferenceBackendRegistry()
        registry.register(
            CodexCLIInferenceBackend(
                cli_profile(),
                CodexCLIInferenceConfig(),
                transport,
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
            settings=CapabilityInferenceSettings(),
        )

        turn = await capability.analyze(ReasoningContext(goal="Analyze"))

        self.assertIsNotNone(turn.result)
        assert turn.result is not None
        self.assertEqual(turn.result.conclusions, ("OAuth path",))
        self.assertEqual(len(transport.calls), 4)

    async def test_gateway_can_invoke_when_auth_probe_is_inconclusive(self) -> None:
        transport = RecordingProcessTransport(
            login=ProcessResult(exit_code=1, stderr="Not logged in"),
            execution=ProcessResult(
                exit_code=0,
                stdout=codex_jsonl('{"conclusions":["Cached OAuth path"]}'),
            ),
        )
        registry = InMemoryInferenceBackendRegistry()
        registry.register(
            CodexCLIInferenceBackend(
                cli_profile(),
                CodexCLIInferenceConfig(),
                transport,
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
        )

        turn = await capability.analyze(ReasoningContext(goal="Analyze"))

        self.assertIsNotNone(turn.result)
        assert turn.result is not None
        self.assertEqual(turn.result.conclusions, ("Cached OAuth path",))
        self.assertEqual(len(transport.calls), 4)

    async def test_unsupported_tool_and_output_token_requests_fail_before_run(
        self,
    ) -> None:
        transport = RecordingProcessTransport()
        backend = CodexCLIInferenceBackend(
            cli_profile(),
            CodexCLIInferenceConfig(),
            transport,
        )
        tool_request = InferenceRequest(
            cognitive_capability_id="reasoning",
            input="analyze",
            requirements=InferenceRequirements(
                tool_intent=ToolIntentMode.ALLOWED
            ),
        )
        with self.assertRaises(UnsupportedFeatureError):
            await backend.invoke(tool_request)

        token_request = InferenceRequest(
            cognitive_capability_id="reasoning",
            input="analyze",
            requirements=InferenceRequirements(max_output_tokens=100),
        )
        with self.assertRaises(UnsupportedFeatureError):
            await backend.invoke(token_request)
        self.assertEqual(transport.calls, [])

    def test_profile_config_and_transport_contracts_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "CLI kind"):
            CodexCLIInferenceBackend(
                cli_profile().model_copy(update={"backend_kind": BackendKind.API}),
                CodexCLIInferenceConfig(),
                RecordingProcessTransport(),
            )
        with self.assertRaisesRegex(ValueError, "cannot expose tool intent"):
            CodexCLIInferenceBackend(
                cli_profile(tool_intent=True),
                CodexCLIInferenceConfig(),
                RecordingProcessTransport(),
            )
        with self.assertRaisesRegex(ValidationError, "must be unique"):
            CodexCLIInferenceConfig(
                inherited_environment_variables=("PATH", "PATH")
            )
        self.assertIsInstance(RecordingProcessTransport(), AsyncProcessTransport)
        self.assertIsInstance(SubprocessTransport(), AsyncProcessTransport)


if __name__ == "__main__":
    unittest.main()

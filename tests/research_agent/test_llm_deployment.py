from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from typing import Any
from unittest.mock import patch

from adaptive_agent_runtime import RunStatus
from adaptive_agent_runtime.llm import (
    AnthropicAPITargetDefinition,
    BackendKind,
    BackendTransportFeatures,
    ClaudeCodeInferenceTargetDefinition,
    ContextSensitivity,
    CodexCLIInferenceTargetDefinition,
    HTTPJSONResponse,
    OpenAICompatibleService,
    OpenAICompatibleProbeMode,
    OpenAICompatibleTargetDefinition,
    ProcessResult,
    StructuredOutputLevel,
)

from applications.research_agent import (
    ResearchAgent,
    ResearchLLMCapability,
    ResearchLLMDeploymentConfig,
    build_research_llm_deployment,
    managed_capability_ids,
)
from applications.research_agent.cli import (
    build_parser,
    llm_config_from_args,
    load_llm_config_file,
)
from applications.research_agent.report import ReportSection, ResearchReport


class RecordingReportTransport:
    module_id = "test.http_transport.research_deployment"

    def __init__(self) -> None:
        self.requests: list[tuple[str, Mapping[str, str], Mapping[str, Any]]] = []
        self.probes: list[tuple[str, Mapping[str, str]]] = []

    async def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        del timeout_seconds
        self.probes.append((url, dict(headers)))
        return HTTPJSONResponse(
            status_code=200,
            body={"data": [{"id": "fixture-model"}]},
        )

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        del timeout_seconds
        self.requests.append((url, headers, body))
        report = ResearchReport(
            company="Tesla",
            executive_summary="Gateway-composed report.",
            sections=(
                ReportSection(
                    title="Evidence",
                    findings=("Runtime evidence remained authoritative.",),
                ),
            ),
            risk_factors=("Provider output remains subject to validation.",),
            investment_view="Balanced",
            markdown="# Tesla Gateway Report",
        )
        artifact = {
            "media_type": "application/json",
            "content": report.model_dump(mode="json"),
            "evidence_reference_ids": [],
            "warnings": [],
        }
        return HTTPJSONResponse(
            status_code=200,
            body={
                "id": "fake-chat-1",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(artifact),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
        )


class RecordingAnthropicReportTransport:
    module_id = "test.http_transport.research_anthropic_deployment"

    def __init__(self) -> None:
        self.requests: list[
            tuple[str, Mapping[str, str], Mapping[str, Any]]
        ] = []
        self.probes: list[tuple[str, Mapping[str, str]]] = []

    async def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        del timeout_seconds
        self.probes.append((url, dict(headers)))
        return HTTPJSONResponse(
            status_code=200,
            body={
                "data": [
                    {
                        "id": "claude-test-model",
                        "type": "model",
                        "capabilities": {
                            "structured_outputs": {"supported": True}
                        },
                    }
                ],
                "has_more": False,
            },
        )

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        del timeout_seconds
        self.requests.append((url, dict(headers), dict(body)))
        report = ResearchReport(
            company="Tesla",
            executive_summary="Anthropic-native report.",
            sections=(
                ReportSection(
                    title="Evidence",
                    findings=("Runtime context remained authoritative.",),
                ),
            ),
            risk_factors=("Provider output remains validated.",),
            investment_view="Balanced",
            markdown="# Tesla Anthropic Report",
        )
        artifact = {
            "media_type": "application/json",
            "content": report.model_dump(mode="json"),
            "evidence_reference_ids": [],
            "warnings": [],
        }
        return HTTPJSONResponse(
            status_code=200,
            body={
                "id": "msg-research-anthropic",
                "type": "message",
                "role": "assistant",
                "model": "claude-test-model",
                "content": [
                    {"type": "text", "text": json.dumps(artifact)}
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        )

class RecordingCodexTransport:
    module_id = "test.process_transport.research_deployment"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str | None]] = []

    def resolve(self, executable: str) -> str | None:
        return f"C:/tools/{executable}.exe"

    async def run(
        self,
        argv: Sequence[str],
        *,
        stdin: str | None,
        cwd: str,
        environment: Mapping[str, str],
        timeout_seconds: float,
    ) -> ProcessResult:
        del cwd, environment, timeout_seconds
        arguments = tuple(argv)
        self.calls.append((arguments, stdin))
        if "--version" in arguments:
            return ProcessResult(exit_code=0, stdout="codex-cli 1.2.3")
        if "login" in arguments and "status" in arguments:
            return ProcessResult(exit_code=0, stdout="Logged in using ChatGPT")
        report = ResearchReport(
            company="Tesla",
            executive_summary="OAuth CLI report.",
            sections=(
                ReportSection(
                    title="Evidence",
                    findings=("CLI inference stayed behind Runtime policy.",),
                ),
            ),
            risk_factors=("CLI output remains untrusted until validated.",),
            investment_view="Balanced",
            markdown="# Tesla OAuth CLI Report",
        )
        artifact = json.dumps(
            {
                "media_type": "application/json",
                "content": report.model_dump(mode="json"),
                "evidence_reference_ids": [],
                "warnings": [],
            }
        )
        events = (
            {"type": "thread.started", "thread_id": "oauth-thread"},
            {
                "type": "item.completed",
                "item": {
                    "id": "message-1",
                    "type": "agent_message",
                    "text": artifact,
                },
            },
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        )
        return ProcessResult(
            exit_code=0,
            stdout="\n".join(json.dumps(event) for event in events),
        )


class RecordingClaudeTransport:
    module_id = "test.process_transport.research_claude_deployment"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str | None]] = []

    def resolve(self, executable: str) -> str | None:
        return f"C:/tools/{executable}.exe"

    async def run(
        self,
        argv: Sequence[str],
        *,
        stdin: str | None,
        cwd: str,
        environment: Mapping[str, str],
        timeout_seconds: float,
    ) -> ProcessResult:
        del cwd, environment, timeout_seconds
        arguments = tuple(argv)
        self.calls.append((arguments, stdin))
        if "--version" in arguments:
            return ProcessResult(exit_code=0, stdout="2.1.205 (Claude Code)")
        if "auth" in arguments and "status" in arguments:
            return ProcessResult(
                exit_code=0,
                stdout=json.dumps(
                    {"loggedIn": True, "authMethod": "claude.ai"}
                ),
            )
        report = ResearchReport(
            company="Tesla",
            executive_summary="Claude Code CLI report.",
            sections=(
                ReportSection(
                    title="Evidence",
                    findings=("Claude inference stayed Runtime-governed.",),
                ),
            ),
            risk_factors=("Provider output is validated before use.",),
            investment_view="Balanced",
            markdown="# Tesla Claude Code Report",
        )
        artifact = {
            "media_type": "application/json",
            "content": report.model_dump(mode="json"),
            "evidence_reference_ids": [],
            "warnings": [],
        }
        return ProcessResult(
            exit_code=0,
            stdout=json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "structured_output": artifact,
                    "session_id": "claude-oauth-session",
                    "total_cost_usd": 0.02,
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                }
            ),
        )


class RecordingToolIntentTransport:
    module_id = "test.http_transport.research_tool_intent"

    def __init__(self) -> None:
        self.requests: list[Mapping[str, Any]] = []
        self.probes: list[str] = []

    async def get_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        del headers, timeout_seconds
        self.probes.append(url)
        return HTTPJSONResponse(
            status_code=200,
            body={"data": [{"id": "fixture-tool-model"}]},
        )

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        del url, headers, timeout_seconds
        self.requests.append(body)
        if len(self.requests) == 1:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "retrieve-industry-http",
                        "type": "function",
                        "function": {
                            "name": "runtime_tool_0",
                            "arguments": json.dumps(
                                {
                                    "company": "Tesla",
                                    "scope": "industry",
                                    "query": "industry evidence",
                                }
                            ),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            message = {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "conclusions": ["Runtime-governed inference conclusion."],
                        "evidence_reference_ids": [],
                    }
                ),
            }
            finish_reason = "stop"
        return HTTPJSONResponse(
            status_code=200,
            body={
                "id": f"reasoning-{len(self.requests)}",
                "choices": [
                    {
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 10,
                    "total_tokens": 60,
                },
            },
        )


def local_config() -> ResearchLLMDeploymentConfig:
    target = OpenAICompatibleTargetDefinition(
        service=OpenAICompatibleService.LOCAL,
        target_id="research/local-test",
        model_id="fixture-model",
        base_url="http://127.0.0.1:11434/v1",
        features=BackendTransportFeatures(
            structured_output=StructuredOutputLevel.JSON_SCHEMA,
        ),
        supported_cognitive_capability_ids=("artifact_generation",),
    )
    return ResearchLLMDeploymentConfig(
        target=target,
        enabled_capabilities=(ResearchLLMCapability.GENERATION,),
        allowed_context_sensitivities=(ContextSensitivity.INTERNAL,),
    )


class ResearchLLMDeploymentTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_provider_composes_through_gateway_and_application(self) -> None:
        transport = RecordingReportTransport()
        deployment = build_research_llm_deployment(
            local_config(),
            transport=transport,
        )

        result = await ResearchAgent(
            cognitive_capabilities=deployment.cognitive_capabilities,
        ).run("分析 Tesla 投资价值")

        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(result.report.markdown, "# Tesla Gateway Report")
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(len(transport.probes), 1)
        url, headers, body = transport.requests[0]
        self.assertEqual(url, "http://127.0.0.1:11434/v1/chat/completions")
        self.assertNotIn("Authorization", headers)
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(len(result.llm_context_packages), 1)
        self.assertEqual(
            result.llm_context_packages[0].target_id,
            deployment.target_id,
        )
        run_id = result.runtime_result.final_state.run_id
        task_id = result.runtime_result.final_state.task.task_id
        trace = deployment.inference.trace.entries(run_id=run_id)
        self.assertTrue(trace)
        self.assertTrue(
            all(
                entry.event.target_id in {None, deployment.target_id}
                for entry in trace
            )
        )
        self.assertTrue(
            all(entry.event.correlation.task_id == task_id for entry in trace)
        )
        self.assertTrue(
            all(entry.event.correlation.node_id is not None for entry in trace)
        )
        action_ids = {
            entry.event.correlation.action_id for entry in trace
        }
        self.assertEqual(len(action_ids), 1)
        self.assertNotIn(None, action_ids)
        self.assertTrue(
            all(
                entry.event.trace_attributes
                == {
                    "application": "research_agent",
                    "operation": "report.generate",
                }
                for entry in trace
            )
        )

    async def test_anthropic_api_composes_through_gateway_and_application(
        self,
    ) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--llm-service",
                "anthropic",
                "--llm-model",
                "claude-test-model",
                "--llm-structured-output",
                "json_schema",
                "--llm-base-url",
                "https://api.anthropic.example/v1",
                "--llm-api-key-env",
                "AAR_ANTHROPIC_RESEARCH_TEST_KEY",
                "--llm-max-output-tokens",
                "1000",
            ]
        )
        config = llm_config_from_args(args)
        assert config is not None
        self.assertIsInstance(config.target, AnthropicAPITargetDefinition)
        transport = RecordingAnthropicReportTransport()
        deployment = build_research_llm_deployment(
            config,
            transport=transport,
        )

        with patch.dict(
            os.environ,
            {"AAR_ANTHROPIC_RESEARCH_TEST_KEY": "test-secret"},
        ):
            probe = await deployment.probe()
            result = await ResearchAgent(
                cognitive_capabilities=deployment.cognitive_capabilities,
            ).run("分析 Tesla 投资价值")

        self.assertEqual(probe.availability.value, "available")
        self.assertEqual(
            result.runtime_result.final_state.status, RunStatus.COMPLETED
        )
        self.assertEqual(result.report.markdown, "# Tesla Anthropic Report")
        self.assertEqual(len(transport.probes), 1)
        self.assertEqual(len(transport.requests), 1)
        probe_url, probe_headers = transport.probes[0]
        self.assertEqual(
            probe_url,
            "https://api.anthropic.example/v1/models?limit=1000",
        )
        self.assertEqual(probe_headers["x-api-key"], "test-secret")
        url, headers, body = transport.requests[0]
        self.assertEqual(url, "https://api.anthropic.example/v1/messages")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertNotIn("Authorization", headers)
        self.assertEqual(body["max_tokens"], 1000)
        self.assertEqual(
            body["output_config"]["format"]["type"], "json_schema"
        )
        self.assertNotIn("tools", body)
        self.assertEqual(len(result.llm_context_packages), 1)
        profile = config.target.build_profile()
        self.assertEqual(profile.backend_kind, BackendKind.API)
        self.assertIn(
            "x_api_key_env", profile.authentication.supported_methods
        )

    async def test_codex_oauth_cli_composes_as_inference_only_target(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--llm-service",
                "codex_cli",
                "--llm-model",
                "gpt-test",
                "--llm-structured-output",
                "json_schema",
                "--llm-executable",
                "codex-test",
            ]
        )
        config = llm_config_from_args(args)
        assert config is not None
        self.assertIsInstance(config.target, CodexCLIInferenceTargetDefinition)
        transport = RecordingCodexTransport()
        deployment = build_research_llm_deployment(
            config,
            transport=transport,
        )

        probe = await deployment.probe()
        self.assertEqual(probe.availability.value, "available")
        self.assertEqual(probe.runtime_version, "1.2.3")
        self.assertEqual(probe.active_auth_method, "chatgpt_oauth")

        result = await ResearchAgent(
            cognitive_capabilities=deployment.cognitive_capabilities,
        ).run("分析 Tesla 投资价值")

        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(result.report.markdown, "# Tesla OAuth CLI Report")
        self.assertEqual(len(transport.calls), 4)
        execution_arguments, prompt = transport.calls[-1]
        self.assertIn("--ephemeral", execution_arguments)
        self.assertEqual(
            execution_arguments[execution_arguments.index("--sandbox") + 1],
            "read-only",
        )
        self.assertIn("bounded inference engine", prompt or "")
        self.assertEqual(len(result.llm_context_packages), 1)
        profile = config.target.build_profile()
        self.assertEqual(profile.backend_kind, BackendKind.CLI)
        self.assertIn("chatgpt_oauth", profile.authentication.supported_methods)

    async def test_claude_code_session_composes_as_inference_only_target(
        self,
    ) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--llm-service",
                "claude_code",
                "--llm-model",
                "claude-test",
                "--llm-structured-output",
                "json_schema",
                "--llm-executable",
                "claude-test",
            ]
        )
        config = llm_config_from_args(args)
        assert config is not None
        self.assertIsInstance(
            config.target, ClaudeCodeInferenceTargetDefinition
        )
        transport = RecordingClaudeTransport()
        deployment = build_research_llm_deployment(
            config,
            transport=transport,
        )

        probe = await deployment.probe()
        self.assertEqual(probe.availability.value, "available")
        self.assertEqual(probe.runtime_version, "2.1.205")
        self.assertEqual(probe.active_auth_method, "claude.ai")

        result = await ResearchAgent(
            cognitive_capabilities=deployment.cognitive_capabilities,
        ).run("分析 Tesla 投资价值")

        self.assertEqual(
            result.runtime_result.final_state.status, RunStatus.COMPLETED
        )
        self.assertEqual(
            result.report.markdown, "# Tesla Claude Code Report"
        )
        self.assertEqual(len(transport.calls), 3)
        execution_arguments, prompt = transport.calls[-1]
        self.assertEqual(
            execution_arguments[execution_arguments.index("--tools") + 1],
            "",
        )
        self.assertIn("--strict-mcp-config", execution_arguments)
        self.assertIn("--no-session-persistence", execution_arguments)
        self.assertIn("bounded inference engine", prompt or "")
        self.assertEqual(len(result.llm_context_packages), 1)
        profile = config.target.build_profile()
        self.assertEqual(profile.backend_kind, BackendKind.CLI)
        self.assertIn(
            "claude_ai_oauth", profile.authentication.supported_methods
        )

    async def test_openai_compatible_tool_intent_round_trip_is_runtime_owned(
        self,
    ) -> None:
        target = OpenAICompatibleTargetDefinition(
            service=OpenAICompatibleService.LOCAL,
            target_id="research/tool-intent-test",
            model_id="fixture-tool-model",
            base_url="http://127.0.0.1:11434/v1",
            features=BackendTransportFeatures(
                structured_output=StructuredOutputLevel.JSON_SCHEMA,
                tool_intent=True,
            ),
            supported_cognitive_capability_ids=("reasoning",),
        )
        config = ResearchLLMDeploymentConfig(
            target=target,
            enabled_capabilities=(ResearchLLMCapability.REASONING,),
            reasoner_tool_intent_limit=1,
        )
        transport = RecordingToolIntentTransport()
        deployment = build_research_llm_deployment(config, transport=transport)

        result = await ResearchAgent(
            cognitive_capabilities=deployment.cognitive_capabilities,
        ).run("分析 Tesla 投资价值")

        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(len(result.llm_tool_intents), 1)
        self.assertTrue(result.llm_tool_intents[0].observation.succeeded)
        self.assertEqual(len(transport.requests), 7)
        self.assertEqual(len(transport.probes), 1)
        self.assertEqual(transport.requests[0]["tool_choice"], "auto")
        self.assertEqual(len(transport.requests[0]["tools"]), 1)
        self.assertTrue(
            all(len(body["tools"]) == 1 for body in transport.requests)
        )

    def test_cli_requires_explicit_model_features_and_never_accepts_key_value(
        self,
    ) -> None:
        parser = build_parser()
        missing_features = parser.parse_args(
            [
                "--llm-service",
                "local",
                "--llm-model",
                "fixture-model",
                "--llm-base-url",
                "http://127.0.0.1:11434/v1",
            ]
        )
        with self.assertRaisesRegex(ValueError, "not inferred"):
            llm_config_from_args(missing_features)

        args = parser.parse_args(
            [
                "--llm-service",
                "local",
                "--llm-model",
                "fixture-model",
                "--llm-structured-output",
                "json_schema",
                "--llm-base-url",
                "http://127.0.0.1:11434/v1",
                "--llm-allow-confidential-context",
                "--llm-capability",
                "generation",
                "--llm-api-probe",
                "credentials_only",
                "--llm-max-attempts",
                "2",
                "--llm-max-elapsed-seconds",
                "12.5",
            ]
        )

        config = llm_config_from_args(args)

        self.assertIsNotNone(config)
        assert config is not None
        self.assertIsInstance(config.target, OpenAICompatibleTargetDefinition)
        assert isinstance(config.target, OpenAICompatibleTargetDefinition)
        self.assertEqual(config.target.build_profile().backend_kind, BackendKind.LOCAL)
        self.assertIn(
            ContextSensitivity.CONFIDENTIAL,
            config.allowed_context_sensitivities,
        )
        self.assertFalse(config.target.build_config().requires_api_key)
        self.assertEqual(
            config.target.build_config().probe_mode,
            OpenAICompatibleProbeMode.CREDENTIALS_ONLY,
        )
        self.assertEqual(config.max_attempts, 2)
        self.assertEqual(config.max_elapsed_seconds, 12.5)

    def test_unified_llm_config_loads_active_deepseek_target(self) -> None:
        config_path = Path(__file__).parents[2] / "config" / "llm.toml"

        config = load_llm_config_file(config_path)

        self.assertIsInstance(config.target, OpenAICompatibleTargetDefinition)
        assert isinstance(config.target, OpenAICompatibleTargetDefinition)
        self.assertEqual(config.target.service, OpenAICompatibleService.DEEPSEEK)
        self.assertEqual(config.target.model_id, "deepseek-v4-flash")
        self.assertEqual(
            config.target.features.structured_output,
            StructuredOutputLevel.JSON_OBJECT,
        )
        backend_config = config.target.build_config()
        self.assertEqual(backend_config.base_url, "https://api.deepseek.com")
        self.assertEqual(backend_config.api_key_env, "DEEPSEEK_API_KEY")
        assert backend_config.reasoning_effort is not None
        self.assertEqual(backend_config.reasoning_effort.value, "high")
        self.assertNotIn("api_key =", config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            config.enabled_capabilities,
            (
                ResearchLLMCapability.GENERATION,
                ResearchLLMCapability.REASONING,
            ),
        )
        self.assertEqual(config.reasoner_tool_intent_limit, 1)

    def test_private_provider_config_is_merged_and_masked(self) -> None:
        config_path = Path(__file__).parents[2] / "config" / "llm.toml"
        with tempfile.TemporaryDirectory() as temp_dir:
            private_path = Path(temp_dir) / "llm.local.toml"
            private_path.write_text(
                """
schema_version = 1
[providers.deepseek]
api_key = "private-deepseek-secret"
""".strip(),
                encoding="utf-8",
            )

            config = load_llm_config_file(
                config_path,
                private_path=private_path,
            )

        self.assertIsInstance(config.target, OpenAICompatibleTargetDefinition)
        assert isinstance(config.target, OpenAICompatibleTargetDefinition)
        assert config.target.api_key is not None
        self.assertEqual(
            config.target.api_key.get_secret_value(),
            "private-deepseek-secret",
        )
        self.assertNotIn("private-deepseek-secret", repr(config))

    def test_private_target_selection_overrides_model_and_effort(self) -> None:
        config_path = Path(__file__).parents[2] / "config" / "llm.toml"
        with tempfile.TemporaryDirectory() as temp_dir:
            private_path = Path(temp_dir) / "llm.local.toml"
            private_path.write_text(
                """
schema_version = 1
[selections.deepseek-research]
model = "deepseek-reasoner"
reasoning_effort = "max"
""".strip(),
                encoding="utf-8",
            )

            config = load_llm_config_file(
                config_path,
                private_path=private_path,
            )

        self.assertIsInstance(config.target, OpenAICompatibleTargetDefinition)
        assert isinstance(config.target, OpenAICompatibleTargetDefinition)
        self.assertEqual(config.target.model_id, "deepseek-reasoner")
        effort = config.target.build_config().reasoning_effort
        assert effort is not None
        self.assertEqual(effort.value, "max")
        self.assertEqual(
            config.target.target_id,
            "research/deepseek/deepseek-v4-flash",
        )

    def test_private_provider_config_rejects_unknown_fields(self) -> None:
        config_path = Path(__file__).parents[2] / "config" / "llm.toml"
        with tempfile.TemporaryDirectory() as temp_dir:
            private_path = Path(temp_dir) / "llm.local.toml"
            private_path.write_text(
                """
schema_version = 1
[providers.deepseek]
api_key = "fixture-secret"
base_url = "https://unexpected.example"
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unknown.*base_url"):
                load_llm_config_file(
                    config_path,
                    private_path=private_path,
                )

    def test_unified_llm_config_selects_named_codex_target(self) -> None:
        config_path = Path(__file__).parents[2] / "config" / "llm.toml"

        config = load_llm_config_file(
            config_path,
            target_name="codex-research",
        )

        self.assertIsInstance(config.target, CodexCLIInferenceTargetDefinition)
        assert isinstance(config.target, CodexCLIInferenceTargetDefinition)
        self.assertEqual(config.target.model_id, "gpt-5.6-terra")
        self.assertEqual(config.target.config.executable, "codex")
        assert config.target.config.reasoning_effort is not None
        self.assertEqual(config.target.config.reasoning_effort.value, "high")

    def test_llm_target_selects_slot_from_unified_config(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--llm-config",
                "config/llm.toml",
                "--llm-target",
                "codex-research",
            ]
        )

        config = llm_config_from_args(args)

        self.assertIsNotNone(config)
        assert config is not None
        self.assertIsInstance(config.target, CodexCLIInferenceTargetDefinition)

    def test_llm_target_requires_config_and_must_exist(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--llm-target", "codex-research"])
        with self.assertRaisesRegex(ValueError, "requires --llm-config"):
            llm_config_from_args(args)

        config_path = Path(__file__).parents[2] / "config" / "llm.toml"
        with self.assertRaisesRegex(ValueError, "is not configured"):
            load_llm_config_file(config_path, target_name="missing-target")

    def test_unified_config_rejects_plaintext_key_in_any_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "llm.toml"
            config_path.write_text(
                """
schema_version = 1
[llm]
active_target = "clean"
[llm.targets.clean]
[llm.targets.unselected]
api_key = "fixture-secret"
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must not contain api_key"):
                load_llm_config_file(config_path)

    def test_llm_config_cannot_be_mixed_with_inline_llm_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--llm-config",
                "config/llm.toml",
                "--llm-model",
                "another-model",
            ]
        )

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            llm_config_from_args(args)

    def test_anthropic_cli_rejects_inapplicable_provider_switches(self) -> None:
        parser = build_parser()
        common = [
            "--llm-service",
            "anthropic",
            "--llm-model",
            "claude-test-model",
            "--llm-structured-output",
            "json_schema",
        ]
        with self.assertRaisesRegex(ValueError, "exact model"):
            llm_config_from_args(
                parser.parse_args(
                    [*common, "--llm-api-probe", "models_endpoint"]
                )
            )
        with self.assertRaisesRegex(ValueError, "native strict"):
            llm_config_from_args(
                parser.parse_args([*common, "--llm-strict-json-schema"])
            )
        object_args = common.copy()
        object_args[-1] = "json_object"
        with self.assertRaisesRegex(ValueError, "require JSON Schema"):
            llm_config_from_args(parser.parse_args(object_args))

    def test_planning_negotiates_all_runtime_owned_proposal_contracts(
        self,
    ) -> None:
        self.assertEqual(
            managed_capability_ids((ResearchLLMCapability.PLANNING,)),
            (
                "task_graph_proposal",
                "action_proposal",
                "graph_mutation_proposal",
                "recovery_proposal",
            ),
        )
        parser = build_parser()
        args = parser.parse_args(
            [
                "--llm-service",
                "local",
                "--llm-model",
                "fixture-model",
                "--llm-structured-output",
                "json_schema",
                "--llm-base-url",
                "http://127.0.0.1:11434/v1",
                "--llm-capability",
                "planning",
            ]
        )

        config = llm_config_from_args(args)

        assert config is not None
        self.assertEqual(
            config.target.build_profile().supported_cognitive_capability_ids,
            (
                "task_graph_proposal",
                "action_proposal",
                "graph_mutation_proposal",
                "recovery_proposal",
            ),
        )
        deployment = build_research_llm_deployment(
            config,
            transport=RecordingReportTransport(),
        )
        cognition = deployment.cognitive_capabilities
        self.assertIsNotNone(cognition.task_planner)
        self.assertIsNotNone(cognition.action_planner)
        self.assertIsNotNone(cognition.mutation_planner)
        self.assertIsNotNone(cognition.recovery_planner)
        self.assertIsNotNone(cognition.mutation_context)

    def test_root_cause_is_a_distinct_negotiated_capability(self) -> None:
        self.assertEqual(
            managed_capability_ids((ResearchLLMCapability.ROOT_CAUSE,)),
            ("root_cause_analysis",),
        )

    def test_tool_selection_is_a_distinct_negotiated_capability(self) -> None:
        self.assertEqual(
            managed_capability_ids((ResearchLLMCapability.TOOL_SELECTION,)),
            ("tool_selection_proposal",),
        )


if __name__ == "__main__":
    unittest.main()

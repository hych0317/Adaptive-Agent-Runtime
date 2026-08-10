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
    InferenceExecutionBudgetError,
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
    TerminalRequirement,
    TerminalSessionSnapshot,
    TerminalTurnRequest,
)
from applications.terminal_bench.planner import (
    GatewayTerminalTurnProposalCapability,
    _identifies_official_test_source,
    _known_verification_mutation,
    _resolve_terminal_cwd,
    _task_requirements,
    _terminal_turn_response_schema,
)


class _RecordingGateway:
    module_id = "test.terminal_bench.recording_gateway"

    def __init__(self, output: Any) -> None:
        self.output = output
        self.request: InferenceRequest | None = None
        self.policy: InferenceGatewayPolicy | None = None

    async def execute(
        self,
        request: InferenceRequest,
        policy: InferenceGatewayPolicy,
    ) -> NormalizedModelResponse:
        self.policy = policy
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
        requirements=(
            TerminalRequirement(
                requirement_id="req-001",
                description="Solve the task",
            ),
        ),
        session=TerminalSessionSnapshot(trial_id="trial-json-object"),
        remaining_commands=1,
        execution_semantics=("Each command is independent.",),
    )


class TerminalModelCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_instruction_scopes_repairs_and_verification(self) -> None:
        gateway = _RecordingGateway(
            {"decision": "complete", "summary": "Task complete"}
        )
        capability = GatewayTerminalTurnProposalCapability(
            gateway=gateway,
            gateway_policy=InferenceGatewayPolicy(),
            target_id="terminal-bench:deepseek:test-model",
            required_structured_output=StructuredOutputLevel.JSON_OBJECT,
        )

        await capability.propose(_turn_request())

        assert gateway.request is not None
        instruction = gateway.request.input["instruction"]
        self.assertIsInstance(instruction, str)
        assert isinstance(instruction, str)
        self.assertIn("inspect only that artifact's bounded", instruction)
        self.assertIn(
            "Gate verification only on explicit task requirements",
            instruction,
        )
        self.assertIn("unrelated tracked files remain unchanged", instruction)
        self.assertIn("prefer a standard-library implementation", instruction)
        self.assertIn("timeout_admission_margin_seconds", instruction)

    def test_verification_mutation_detection_preserves_read_only_git_checks(
        self,
    ) -> None:
        for command in (
            "git branch --list",
            "git show-ref --verify refs/heads/main",
        ):
            with self.subTest(command=command):
                self.assertIsNone(_known_verification_mutation(command))

        for command in (
            "git branch verifier-ref",
            "git checkout -b verifier-ref",
            "git switch -c verifier-ref",
            "git update-ref refs/heads/verifier-ref HEAD",
        ):
            with self.subTest(command=command):
                self.assertIsNotNone(_known_verification_mutation(command))

    def test_verification_allows_disposable_git_mutation_with_cleanup(self) -> None:
        command = "\n".join(
            (
                'tmp="$(mktemp -d)"',
                'trap \'rm -rf "$tmp"\' EXIT',
                'git clone /git/server "$tmp/clone"',
                'git -C "$tmp/clone" commit -m verifier',
            )
        )
        self.assertIsNone(_known_verification_mutation(command))
        self.assertIsNotNone(
            _known_verification_mutation(
                command.replace('trap \'rm -rf "$tmp"\' EXIT\n', "")
            )
        )

    def test_official_test_source_accepts_exact_task_test_path(self) -> None:
        self.assertTrue(_identifies_official_test_source("/app/test_outputs.py"))
        self.assertTrue(_identifies_official_test_source("python -m pytest -q"))
        self.assertFalse(_identifies_official_test_source("task-local check"))

    def test_remote_evaluation_uses_expanded_model_budget(self) -> None:
        config = TerminalModelConfig(model_name="deepseek/test-model")

        self.assertEqual(
            config.max_output_tokens,
            32768,
        )
        self.assertEqual(config.compact_max_output_tokens, 8192)
        self.assertEqual(config.emergency_max_output_tokens, 4096)
        self.assertEqual(config.inference_timeout_sec, 300.0)
        self.assertEqual(config.delivery_inference_timeout_sec, 180.0)
        self.assertEqual(config.emergency_inference_timeout_sec, 120.0)
        self.assertEqual(config.minimum_inference_timeout_sec, 120.0)
        self.assertEqual(
            config.minimum_delivery_inference_timeout_sec,
            60.0,
        )
        self.assertEqual(
            config.deepseek_reasoning_effort,
            ReasoningEffort.HIGH,
        )
        self.assertEqual(config.deepseek_thinking, "enabled")
        self.assertEqual(
            config.codex_reasoning_effort,
            ReasoningEffort.HIGH,
        )

    def test_terminal_context_is_bounded_within_total_token_budget(self) -> None:
        policy = TerminalExecutionPolicy()

        self.assertEqual(policy.max_total_tokens, 600_000)
        self.assertEqual(policy.max_context_output_characters, 3_000)
        self.assertEqual(policy.max_context_records, 4)
        self.assertEqual(policy.max_delivery_context_output_characters, 1_000)
        self.assertEqual(policy.max_delivery_context_records, 2)
        self.assertEqual(policy.deadline_reserve_seconds, 60.0)
        self.assertEqual(policy.delivery_mode_fraction, 0.40)
        self.assertEqual(policy.timeout_admission_margin_seconds, 8.0)

        self.assertEqual(policy.max_no_progress_seconds, 480.0)
        self.assertEqual(policy.max_artifact_first_inspections, 1)
        self.assertEqual(policy.max_consecutive_inspections, 3)
        self.assertEqual(policy.max_total_inspections, 5)

    def test_task_requirements_are_stable_and_atomic(self) -> None:
        requirements = _task_requirements(
            "Configure the Git service. Serve hello.html over HTTP.\n"
            "- Verify a fresh clone succeeds"
        )

        self.assertEqual(
            tuple(item.requirement_id for item in requirements),
            ("req-001", "req-002", "req-003"),
        )
        self.assertIn("hello.html", requirements[1].description)

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
        schema = _terminal_turn_response_schema(
            strict=True,
            max_timeout_sec=300,
        )

        def assert_ref_nodes_have_no_default(node: object) -> None:
            if isinstance(node, list):
                for item in node:
                    assert_ref_nodes_have_no_default(item)
                return
            if not isinstance(node, dict):
                return
            if "$ref" in node:
                self.assertNotIn("default", node)
            for value in node.values():
                assert_ref_nodes_have_no_default(value)

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
        timeout_integer = properties["timeout_sec"]["anyOf"][0]
        self.assertEqual(timeout_integer["maximum"], 300)
        verification = schema["$defs"]["TerminalVerificationContract"]
        self.assertEqual(
            set(verification["required"]),
            set(verification["properties"]),
        )
        assert_ref_nodes_have_no_default(schema)

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
        self.assertEqual(len(schema["allOf"]), 4)
        execute_contract = schema["allOf"][0]["then"]
        self.assertEqual(
            execute_contract["required"],
            ("call_key", "command", "command_role"),
        )
        timeout_integer = schema["properties"]["timeout_sec"]["anyOf"][0]
        self.assertEqual(timeout_integer["maximum"], 300)
        instruction = valid_gateway.request.input["instruction"]
        self.assertIn("absent from payload.used_call_keys", instruction)
        self.assertIn("never '.' or another relative path", instruction)
        self.assertIn("not a copy of the production algorithm", instruction)
        self.assertIn("complete reported set", instruction)
        self.assertIn("canonical end-to-end install or build command", instruction)
        self.assertIn("Do not make speculative compatibility patches", instruction)
        self.assertIn("coherent batch", instruction)
        self.assertIn("successful compile is not a successful install", instruction)
        self.assertIn("prepare the complete toolchain", instruction)
        self.assertIn("disabling build isolation when justified", instruction)
        self.assertIn("Never accept a pure-Python install", instruction)
        self.assertIn("most recent failed command", instruction)
        self.assertIn("targeted portable POSIX tools", instruction)
        self.assertIn("continue through all checks", instruction)
        self.assertIn("combine targeted inspection and repair", instruction)
        self.assertIn("all visible compatibility fixes", instruction)
        self.assertIn("write verbose logs to task-local files", instruction)
        self.assertIn("root-cause summary for every failed check", instruction)
        self.assertIn("Do not repeat an unchanged failed verification", instruction)
        self.assertIn("deprecated or removed API family", instruction)
        self.assertIn("Never mask a diagnostic failure", instruction)
        self.assertIn("apply_patch is explicitly unavailable", instruction)
        self.assertIn("payload.execution_limits", instruction)
        self.assertIn("state_policy=read_only", instruction)
        self.assertIn("complete immediately", instruction)
        self.assertIn("Never return rationale", instruction)
        self.assertIn("payload.artifact_first_mode", instruction)
        self.assertIn("payload.repair_mode", instruction)
        self.assertIn("payload.verification_due", instruction)

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

    async def test_delivery_mode_preserves_action_time(self) -> None:
        gateway = _RecordingGateway(
            {"decision": "complete", "summary": "Task complete"}
        )
        gateway_policy = InferenceGatewayPolicy()
        gateway_policy = gateway_policy.model_copy(
            update={
                "budget": gateway_policy.budget.model_copy(
                    update={"max_elapsed_seconds": 360.0}
                )
            }
        )
        capability = GatewayTerminalTurnProposalCapability(
            gateway=gateway,
            gateway_policy=gateway_policy,
            target_id="terminal-bench:deepseek:test-model",
            required_structured_output=StructuredOutputLevel.JSON_OBJECT,
            delivery_timeout_seconds=180.0,
        )
        request = _turn_request().model_copy(
            update={
                "remaining_wall_clock_seconds": 480.0,
                "delivery_mode": True,
            }
        )

        await capability.propose(request)

        assert gateway.policy is not None
        self.assertEqual(gateway.policy.budget.max_elapsed_seconds, 180.0)

        assert gateway.request is not None
        payload = gateway.request.input["payload"]
        assert isinstance(payload, Mapping)
        self.assertTrue(payload["delivery_mode"])

    async def test_emergency_mode_uses_compact_inference_timeout(self) -> None:
        gateway = _RecordingGateway(
            {"decision": "complete", "summary": "Task complete"}
        )
        capability = GatewayTerminalTurnProposalCapability(
            gateway=gateway,
            gateway_policy=InferenceGatewayPolicy(),
            target_id="terminal-bench:deepseek:test-model",
            required_structured_output=StructuredOutputLevel.JSON_OBJECT,
            delivery_timeout_seconds=180.0,
            emergency_timeout_seconds=120.0,
        )
        request = _turn_request().model_copy(
            update={
                "remaining_wall_clock_seconds": 480.0,
                "delivery_mode": True,
                "emergency_mode": True,
            }
        )

        await capability.propose(request)

        assert gateway.policy is not None
        self.assertEqual(gateway.policy.budget.max_elapsed_seconds, 120.0)

    async def test_endgame_modes_reduce_supported_output_budget(self) -> None:
        for update, expected in (
            ({"delivery_mode": True}, 8192),
            ({"repair_mode": True}, 8192),
            ({"emergency_mode": True}, 4096),
        ):
            with self.subTest(update=update):
                gateway = _RecordingGateway(
                    {"decision": "complete", "summary": "Task complete"}
                )
                capability = GatewayTerminalTurnProposalCapability(
                    gateway=gateway,
                    gateway_policy=InferenceGatewayPolicy(),
                    target_id="terminal-bench:deepseek:test-model",
                    max_output_tokens=32768,
                )

                await capability.propose(
                    _turn_request().model_copy(update=update)
                )

                assert gateway.request is not None
                self.assertEqual(
                    gateway.request.requirements.max_output_tokens,
                    expected,
                )

    async def test_normal_inference_preserves_future_correction_slot(
        self,
    ) -> None:
        for remaining, expected in ((840.0, 300.0), (480.0, 160.0)):
            with self.subTest(remaining=remaining):
                gateway = _RecordingGateway(
                    {"decision": "complete", "summary": "Task complete"}
                )
                gateway_policy = InferenceGatewayPolicy()
                gateway_policy = gateway_policy.model_copy(
                    update={
                        "budget": gateway_policy.budget.model_copy(
                            update={"max_elapsed_seconds": 300.0}
                        )
                    }
                )
                capability = GatewayTerminalTurnProposalCapability(
                    gateway=gateway,
                    gateway_policy=gateway_policy,
                    target_id="terminal-bench:deepseek:test-model",
                    required_structured_output=(
                        StructuredOutputLevel.JSON_OBJECT
                    ),
                    delivery_timeout_seconds=180.0,
                    minimum_delivery_timeout_seconds=60.0,
                )
                request = _turn_request().model_copy(
                    update={"remaining_wall_clock_seconds": remaining}
                )

                await capability.propose(request)

                assert gateway.policy is not None
                self.assertEqual(
                    gateway.policy.budget.max_elapsed_seconds,
                    expected,
                )

    async def test_repair_mode_without_wall_clock_uses_delivery_timeout(
        self,
    ) -> None:
        gateway = _RecordingGateway(
            {"decision": "complete", "summary": "Task complete"}
        )
        gateway_policy = InferenceGatewayPolicy()
        gateway_policy = gateway_policy.model_copy(
            update={
                "budget": gateway_policy.budget.model_copy(
                    update={"max_elapsed_seconds": 300.0}
                )
            }
        )
        capability = GatewayTerminalTurnProposalCapability(
            gateway=gateway,
            gateway_policy=gateway_policy,
            target_id="terminal-bench:deepseek:test-model",
            required_structured_output=StructuredOutputLevel.JSON_OBJECT,
            delivery_timeout_seconds=180.0,
        )
        request = _turn_request().model_copy(update={"repair_mode": True})

        await capability.propose(request)

        assert gateway.policy is not None
        self.assertEqual(
            gateway.policy.budget.max_elapsed_seconds,
            180.0,
        )

    async def test_inference_is_not_started_without_minimum_viable_window(
        self,
    ) -> None:
        gateway = _RecordingGateway(
            {"decision": "complete", "summary": "Task complete"}
        )
        capability = GatewayTerminalTurnProposalCapability(
            gateway=gateway,
            gateway_policy=InferenceGatewayPolicy(),
            target_id="terminal-bench:deepseek:test-model",
            required_structured_output=StructuredOutputLevel.JSON_OBJECT,
            delivery_timeout_seconds=300.0,
            minimum_timeout_seconds=120.0,
        )
        request = _turn_request().model_copy(
            update={
                "remaining_wall_clock_seconds": 180.0,
                "delivery_mode": True,
            }
        )

        with self.assertRaisesRegex(
            InferenceExecutionBudgetError,
            "insufficient wall-clock capacity",
        ):
            await capability.propose(request)
        self.assertIsNone(gateway.request)

    async def test_zero_inference_window_is_a_budget_error(self) -> None:
        gateway = _RecordingGateway(
            {"decision": "complete", "summary": "Task complete"}
        )
        capability = GatewayTerminalTurnProposalCapability(
            gateway=gateway,
            gateway_policy=InferenceGatewayPolicy(),
            target_id="terminal-bench:deepseek:test-model",
            required_structured_output=StructuredOutputLevel.JSON_OBJECT,
            delivery_timeout_seconds=300.0,
            minimum_timeout_seconds=120.0,
            minimum_delivery_timeout_seconds=60.0,
        )
        request = _turn_request().model_copy(
            update={
                "remaining_wall_clock_seconds": 120.0,
                "delivery_mode": True,
            }
        )

        with self.assertRaisesRegex(
            InferenceExecutionBudgetError,
            "insufficient wall-clock capacity",
        ):
            await capability.propose(request)
        self.assertIsNone(gateway.request)

    def test_delivery_timeout_cannot_be_below_minimum_viable_window(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "cannot be below the minimum delivery",
        ):
            GatewayTerminalTurnProposalCapability(
                gateway=_RecordingGateway({}),
                gateway_policy=InferenceGatewayPolicy(),
                target_id="terminal-bench:deepseek:test-model",
                delivery_timeout_seconds=60.0,
                minimum_timeout_seconds=120.0,
                minimum_delivery_timeout_seconds=90.0,
            )

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

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from adaptive_agent_runtime.llm import (
    AgentActionKind,
    AgentAuthenticationRequiredError,
    AgentBackendUnavailableError,
    AgentExecutionBudgetError,
    AgentExecutionPolicy,
    AgentExecutionPolicyError,
    AgentExecutionProtocolError,
    AgentExecutionTimeoutError,
    AgentProcessFailedError,
    AgentTargetProfile,
    AutonomousAgentBackend,
    AutonomousAgentRequest,
    BackendDelegatedAccess,
    BackendKind,
    ClaudeCodeAutonomousAgentBackend,
    ClaudeCodeAutonomousAgentConfig,
    ProcessResult,
    ProcessTransportTimeoutError,
)


class RecordingClaudeAgentTransport:
    module_id = "test.agent_process_transport.claude_recording"

    def __init__(
        self,
        execution: ProcessResult | None = None,
        *,
        resolved: str | None = "C:/tools/claude.exe",
        error: Exception | None = None,
    ) -> None:
        self.execution = execution or ProcessResult(
            exit_code=0,
            stdout=agent_stream({"status": "ok"}),
        )
        self.resolved = resolved
        self.error = error
        self.calls: list[
            tuple[tuple[str, ...], str | None, str, Mapping[str, str], float]
        ] = []

    def resolve(self, executable: str) -> str | None:
        self.resolved_name = executable
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
        self.calls.append(
            (tuple(argv), stdin, cwd, dict(environment), timeout_seconds)
        )
        if self.error is not None:
            raise self.error
        if "--version" in argv:
            return ProcessResult(exit_code=0, stdout="2.1.219 (Claude Code)")
        if "auth" in argv and "status" in argv:
            return ProcessResult(
                exit_code=0,
                stdout=json.dumps(
                    {"loggedIn": True, "authMethod": "claude.ai"}
                ),
            )
        return self.execution


def agent_stream(
    output: Any,
    *,
    tools: tuple[str, ...] = ("Bash",),
    input_tokens: Any = 12,
    output_tokens: Any = 8,
    cost: Any = 0.03,
    success: bool = True,
) -> str:
    values: list[dict[str, Any]] = [
        {"type": "system", "session_id": "claude-agent-session"}
    ]
    for index, tool in enumerate(tools, start=1):
        tool_id = f"tool-{index}"
        values.extend(
            (
                {
                    "type": "assistant",
                    "session_id": "claude-agent-session",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": tool,
                                "input": {"command": "echo bounded"},
                            }
                        ]
                    },
                },
                {
                    "type": "user",
                    "session_id": "claude-agent-session",
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": "bounded",
                            }
                        ]
                    },
                },
            )
        )
    values.append(
        {
            "type": "result",
            "subtype": "success" if success else "error_during_execution",
            "is_error": not success,
            "session_id": "claude-agent-session",
            "result": "done",
            "structured_output": output,
            "total_cost_usd": cost,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }
    )
    return "\n".join(json.dumps(value) for value in values)


def profile(*, write: bool = True) -> AgentTargetProfile:
    return AgentTargetProfile(
        target_id="claude-agent/test",
        backend_id="claude-code",
        backend_kind=BackendKind.CLI,
        adapter_version="1",
        model_id="claude-test-model",
        delegated_access=BackendDelegatedAccess(
            filesystem_read=True,
            filesystem_write=write,
            shell_execution=True,
        ),
    )


def policy(
    *,
    write: bool = False,
    max_actions: int = 64,
    max_tokens: int | None = None,
    max_turns: int | None = None,
    max_cost: float | None = None,
) -> AgentExecutionPolicy:
    return AgentExecutionPolicy(
        delegated_access=BackendDelegatedAccess(
            filesystem_read=True,
            filesystem_write=write,
            shell_execution=True,
        ),
        max_action_events=max_actions,
        max_total_tokens=max_tokens,
        max_agent_turns=max_turns,
        max_monetary_cost=max_cost,
    )


def request(
    workspace: str,
    *,
    execution_policy: AgentExecutionPolicy | None = None,
    schema: dict[str, Any] | None = None,
) -> AutonomousAgentRequest:
    return AutonomousAgentRequest(
        goal="Review isolated evidence",
        input={"evidence": "bounded"},
        workspace_root=workspace,
        response_schema=schema,
        policy=execution_policy or policy(),
        timeout_seconds=20.0,
    )


def response_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {"status": {"type": "string"}},
        "type": "object",
        "properties": {"status": {"$ref": "#/$defs/status"}},
        "required": ["status"],
    }


class ClaudeCodeAutonomousAgentBackendTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_read_only_execution_has_strict_sandbox_and_trace(self) -> None:
        transport = RecordingClaudeAgentTransport()
        backend = ClaudeCodeAutonomousAgentBackend(
            profile(),
            ClaudeCodeAutonomousAgentConfig(
                inherited_environment_variables=(), default_max_turns=10
            ),
            transport,
        )
        with tempfile.TemporaryDirectory() as workspace:
            result = await backend.execute(
                request(
                    workspace,
                    execution_policy=policy(
                        max_turns=4,
                        max_tokens=30,
                        max_cost=0.05,
                    ),
                    schema=response_schema(),
                )
            )

        self.assertEqual(result.output, {"status": "ok"})
        self.assertEqual(result.usage.total_tokens, 20)
        self.assertEqual(result.usage.monetary_cost, 0.03)
        self.assertEqual(result.remote_request_id, "claude-agent-session")
        self.assertEqual(
            result.actions[0].kind, AgentActionKind.COMMAND_EXECUTION
        )
        self.assertEqual(result.actions[0].status, "completed")
        argv, prompt, cwd, environment, timeout = transport.calls[0]
        self.assertEqual(argv[argv.index("--tools") + 1], "Bash")
        self.assertEqual(argv[argv.index("--max-turns") + 1], "4")
        self.assertEqual(
            argv[argv.index("--max-budget-usd") + 1], "0.05"
        )
        self.assertIn("--strict-mcp-config", argv)
        self.assertIn("--no-session-persistence", argv)
        self.assertIn("--json-schema", argv)
        schema = json.loads(argv[argv.index("--json-schema") + 1])
        self.assertNotIn("$defs", schema)
        self.assertIn("definitions", schema)
        settings = json.loads(argv[argv.index("--settings") + 1])
        sandbox = settings["sandbox"]
        self.assertTrue(sandbox["enabled"])
        self.assertTrue(sandbox["failIfUnavailable"])
        self.assertFalse(sandbox["allowUnsandboxedCommands"])
        self.assertTrue(sandbox["network"]["strictAllowlist"])
        self.assertEqual(sandbox["network"]["allowedDomains"], [])
        self.assertEqual(sandbox["filesystem"]["denyWrite"], [cwd])
        self.assertIn("mcp__*", settings["permissions"]["deny"])
        self.assertIn("sandboxed Bash", prompt or "")
        self.assertEqual(environment, {})
        self.assertEqual(timeout, 20.0)

    async def test_workspace_write_is_explicit(self) -> None:
        transport = RecordingClaudeAgentTransport()
        backend = ClaudeCodeAutonomousAgentBackend(
            profile(), ClaudeCodeAutonomousAgentConfig(), transport
        )
        with tempfile.TemporaryDirectory() as workspace:
            await backend.execute(
                request(
                    workspace,
                    execution_policy=policy(write=True),
                    schema=response_schema(),
                )
            )

        argv = transport.calls[0][0]
        settings = json.loads(argv[argv.index("--settings") + 1])
        self.assertNotIn("denyWrite", settings["sandbox"]["filesystem"])

    async def test_action_token_cost_and_schema_budgets_are_enforced(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            cases = (
                (
                    RecordingClaudeAgentTransport(
                        ProcessResult(
                            exit_code=0,
                            stdout=agent_stream(
                                {"status": "ok"}, tools=("Bash", "Bash")
                            ),
                        )
                    ),
                    policy(max_actions=1),
                    AgentExecutionBudgetError,
                ),
                (
                    RecordingClaudeAgentTransport(),
                    policy(max_tokens=10),
                    AgentExecutionBudgetError,
                ),
                (
                    RecordingClaudeAgentTransport(),
                    policy(max_cost=0.01),
                    AgentExecutionBudgetError,
                ),
                (
                    RecordingClaudeAgentTransport(
                        ProcessResult(
                            exit_code=0,
                            stdout=agent_stream({"unexpected": True}),
                        )
                    ),
                    policy(),
                    AgentExecutionProtocolError,
                ),
            )
            for transport, execution_policy, error_type in cases:
                with self.subTest(error_type=error_type):
                    backend = ClaudeCodeAutonomousAgentBackend(
                        profile(),
                        ClaudeCodeAutonomousAgentConfig(),
                        transport,
                    )
                    with self.assertRaises(error_type):
                        await backend.execute(
                            request(
                                workspace,
                                execution_policy=execution_policy,
                                schema=response_schema(),
                            )
                        )

    async def test_disallowed_or_unknown_tool_actions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            for tool, error_type in (
                ("WebFetch", AgentExecutionPolicyError),
                ("CustomTool", AgentExecutionProtocolError),
            ):
                backend = ClaudeCodeAutonomousAgentBackend(
                    profile(),
                    ClaudeCodeAutonomousAgentConfig(),
                    RecordingClaudeAgentTransport(
                        ProcessResult(
                            exit_code=0,
                            stdout=agent_stream(
                                {"status": "ok"}, tools=(tool,)
                            ),
                        )
                    ),
                )
                with self.subTest(tool=tool):
                    with self.assertRaises(error_type):
                        await backend.execute(
                            request(workspace, schema=response_schema())
                        )

    async def test_workspace_and_permission_boundaries_fail_before_process(
        self,
    ) -> None:
        transport = RecordingClaudeAgentTransport()
        backend = ClaudeCodeAutonomousAgentBackend(
            profile(write=False),
            ClaudeCodeAutonomousAgentConfig(),
            transport,
        )
        with self.assertRaisesRegex(
            AgentExecutionPolicyError, "workspace root must be absolute"
        ):
            await backend.execute(request("relative/path"))
        with tempfile.TemporaryDirectory() as workspace:
            with self.assertRaisesRegex(
                AgentExecutionPolicyError, "exceeds target profile"
            ):
                await backend.execute(
                    request(workspace, execution_policy=policy(write=True))
                )
        self.assertEqual(transport.calls, [])

    async def test_probe_and_process_failures_are_normalized(self) -> None:
        probe_transport = RecordingClaudeAgentTransport()
        backend = ClaudeCodeAutonomousAgentBackend(
            profile(), ClaudeCodeAutonomousAgentConfig(), probe_transport
        )
        probe = await backend.probe()
        self.assertEqual(probe.availability.value, "available")
        self.assertEqual(probe.runtime_version, "2.1.219")
        self.assertEqual(probe.active_auth_method, "claude.ai")

        with tempfile.TemporaryDirectory() as workspace:
            cases: tuple[
                tuple[RecordingClaudeAgentTransport, type[Exception]], ...
            ] = (
                (
                    RecordingClaudeAgentTransport(
                        error=ProcessTransportTimeoutError()
                    ),
                    AgentExecutionTimeoutError,
                ),
                (
                    RecordingClaudeAgentTransport(
                        ProcessResult(exit_code=7, stderr="private failure")
                    ),
                    AgentProcessFailedError,
                ),
                (
                    RecordingClaudeAgentTransport(
                        ProcessResult(exit_code=1, stderr="please run /login")
                    ),
                    AgentAuthenticationRequiredError,
                ),
                (
                    RecordingClaudeAgentTransport(resolved=None),
                    AgentBackendUnavailableError,
                ),
            )
            for transport, error_type in cases:
                with self.subTest(error_type=error_type):
                    failure_backend = ClaudeCodeAutonomousAgentBackend(
                        profile(), ClaudeCodeAutonomousAgentConfig(), transport
                    )
                    with self.assertRaises(error_type):
                        await failure_backend.execute(request(workspace))

    def test_contract_profile_and_version_fail_closed(self) -> None:
        backend = ClaudeCodeAutonomousAgentBackend(
            profile(),
            ClaudeCodeAutonomousAgentConfig(),
            RecordingClaudeAgentTransport(),
        )
        self.assertIsInstance(backend, AutonomousAgentBackend)
        with self.assertRaisesRegex(ValidationError, "2.1.219 or later"):
            ClaudeCodeAutonomousAgentConfig(
                minimum_runtime_version="2.1.218"
            )
        with self.assertRaisesRegex(ValueError, "must disclose"):
            ClaudeCodeAutonomousAgentBackend(
                AgentTargetProfile(
                    target_id="claude-agent/hidden-shell",
                    backend_id="claude-code",
                    backend_kind=BackendKind.CLI,
                    adapter_version="1",
                    model_id="claude-test-model",
                ),
                ClaudeCodeAutonomousAgentConfig(),
                RecordingClaudeAgentTransport(),
            )


if __name__ == "__main__":
    unittest.main()

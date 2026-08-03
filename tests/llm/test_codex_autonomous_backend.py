from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
    CodexCLIAutonomousAgentBackend,
    CodexCLIInferenceConfig,
    ProcessResult,
    ProcessTransportTimeoutError,
)


class RecordingAgentProcessTransport:
    module_id = "test.agent_process_transport.recording"

    def __init__(
        self,
        execution: ProcessResult | None = None,
        *,
        resolved: str | None = "C:/tools/codex.exe",
        error: Exception | None = None,
    ) -> None:
        self.execution = execution or ProcessResult(
            exit_code=0,
            stdout=agent_jsonl('{"status":"ok"}'),
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
            return ProcessResult(exit_code=0, stdout="codex-cli 1.2.3")
        if "login" in argv:
            return ProcessResult(exit_code=0, stdout="Logged in using ChatGPT")
        return self.execution


def event(item_type: str, *, item_id: str) -> dict[str, object]:
    return {
        "type": "item.completed",
        "item": {
            "id": item_id,
            "type": item_type,
            "status": "completed",
        },
    }


def agent_jsonl(
    final_text: str,
    *,
    actions: tuple[str, ...] = ("command_execution",),
    input_tokens: int = 12,
    output_tokens: int = 8,
) -> str:
    values: list[dict[str, object]] = [
        {"type": "thread.started", "thread_id": "agent-thread"}
    ]
    values.extend(
        event(item_type, item_id=f"action-{index}")
        for index, item_type in enumerate(actions, start=1)
    )
    values.extend(
        (
            {
                "type": "item.completed",
                "item": {
                    "id": "final",
                    "type": "agent_message",
                    "text": final_text,
                },
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            },
        )
    )
    return "\n".join(json.dumps(value) for value in values)


def profile(*, write: bool = True) -> AgentTargetProfile:
    return AgentTargetProfile(
        target_id="codex-agent/test",
        backend_id="codex-cli",
        backend_kind=BackendKind.CLI,
        adapter_version="1",
        model_id="test-model",
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
) -> AgentExecutionPolicy:
    return AgentExecutionPolicy(
        delegated_access=BackendDelegatedAccess(
            filesystem_read=True,
            filesystem_write=write,
            shell_execution=True,
        ),
        max_action_events=max_actions,
        max_total_tokens=max_tokens,
    )


def request(
    workspace: str,
    *,
    execution_policy: AgentExecutionPolicy | None = None,
    schema: dict[str, Any] | None = None,
) -> AutonomousAgentRequest:
    return AutonomousAgentRequest(
        goal="Review the isolated evidence",
        input={"evidence": "bounded"},
        workspace_root=workspace,
        response_schema=schema,
        policy=execution_policy or policy(),
        timeout_seconds=20.0,
    )


class CodexAutonomousAgentBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_only_execution_is_normalized_with_trace_boundary(
        self,
    ) -> None:
        transport = RecordingAgentProcessTransport()
        backend = CodexCLIAutonomousAgentBackend(
            profile(),
            CodexCLIInferenceConfig(inherited_environment_variables=()),
            transport,
        )
        with tempfile.TemporaryDirectory() as workspace:
            result = await backend.execute(
                request(
                    workspace,
                    schema={
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                        "required": ["status"],
                    },
                )
            )

        self.assertEqual(result.output, {"status": "ok"})
        self.assertEqual(result.usage.total_tokens, 20)
        self.assertEqual(result.remote_request_id, "agent-thread")
        self.assertEqual(result.actions[0].kind, AgentActionKind.COMMAND_EXECUTION)
        argv, prompt, cwd, environment, timeout = transport.calls[0]
        self.assertEqual(argv[argv.index("--sandbox") + 1], "read-only")
        self.assertIn("--ephemeral", argv)
        self.assertIn("--ignore-user-config", argv)
        self.assertIn("--output-schema", argv)
        self.assertIn("network", prompt or "")
        self.assertIn("PATH", environment)
        self.assertIn("C:/tools", environment["PATH"].replace("\\", "/"))
        self.assertEqual(timeout, 20.0)
        self.assertTrue(Path(cwd).is_absolute())

    async def test_workspace_write_is_explicit_and_file_change_is_traced(
        self,
    ) -> None:
        transport = RecordingAgentProcessTransport(
            ProcessResult(
                exit_code=0,
                stdout=agent_jsonl(
                    "done",
                    actions=("command_execution", "file_change"),
                ),
            )
        )
        backend = CodexCLIAutonomousAgentBackend(
            profile(),
            CodexCLIInferenceConfig(inherited_environment_variables=()),
            transport,
        )
        with tempfile.TemporaryDirectory() as workspace:
            result = await backend.execute(
                request(workspace, execution_policy=policy(write=True))
            )

        self.assertEqual(
            tuple(item.kind for item in result.actions),
            (
                AgentActionKind.COMMAND_EXECUTION,
                AgentActionKind.FILE_CHANGE,
            ),
        )
        argv = transport.calls[0][0]
        self.assertEqual(argv[argv.index("--sandbox") + 1], "workspace-write")

    async def test_disallowed_actions_are_rejected(self) -> None:
        transport = RecordingAgentProcessTransport(
            ProcessResult(
                exit_code=0,
                stdout=agent_jsonl("done", actions=("file_change",)),
            )
        )
        backend = CodexCLIAutonomousAgentBackend(
            profile(),
            CodexCLIInferenceConfig(inherited_environment_variables=()),
            transport,
        )
        with tempfile.TemporaryDirectory() as workspace:
            with self.assertRaisesRegex(
                AgentExecutionPolicyError,
                "disallowed action 'file_change'",
            ):
                await backend.execute(request(workspace))

    async def test_action_token_and_schema_budgets_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            too_many = CodexCLIAutonomousAgentBackend(
                profile(),
                CodexCLIInferenceConfig(inherited_environment_variables=()),
                RecordingAgentProcessTransport(
                    ProcessResult(
                        exit_code=0,
                        stdout=agent_jsonl(
                            "done",
                            actions=("command_execution", "command_execution"),
                        ),
                    )
                ),
            )
            with self.assertRaises(AgentExecutionBudgetError):
                await too_many.execute(
                    request(workspace, execution_policy=policy(max_actions=1))
                )

            too_costly = CodexCLIAutonomousAgentBackend(
                profile(),
                CodexCLIInferenceConfig(inherited_environment_variables=()),
                RecordingAgentProcessTransport(),
            )
            with self.assertRaises(AgentExecutionBudgetError):
                await too_costly.execute(
                    request(workspace, execution_policy=policy(max_tokens=10))
                )

            invalid_schema = CodexCLIAutonomousAgentBackend(
                profile(),
                CodexCLIInferenceConfig(inherited_environment_variables=()),
                RecordingAgentProcessTransport(
                    ProcessResult(
                        exit_code=0,
                        stdout=agent_jsonl('{"unexpected":true}'),
                    )
                ),
            )
            with self.assertRaisesRegex(
                AgentExecutionProtocolError,
                "violates response schema",
            ):
                await invalid_schema.execute(
                    request(
                        workspace,
                        schema={
                            "type": "object",
                            "required": ["status"],
                        },
                    )
                )

    async def test_workspace_and_target_permissions_fail_before_process(self) -> None:
        transport = RecordingAgentProcessTransport()
        backend = CodexCLIAutonomousAgentBackend(
            profile(write=False),
            CodexCLIInferenceConfig(inherited_environment_variables=()),
            transport,
        )
        with self.assertRaisesRegex(
            AgentExecutionPolicyError,
            "workspace root must be absolute",
        ):
            await backend.execute(request("relative/path"))
        with tempfile.TemporaryDirectory() as workspace:
            with self.assertRaisesRegex(
                AgentExecutionPolicyError,
                "exceeds target profile",
            ):
                await backend.execute(
                    request(workspace, execution_policy=policy(write=True))
                )
        self.assertEqual(transport.calls, [])

    async def test_timeout_auth_process_and_unavailable_failures_are_normalized(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            cases: tuple[tuple[RecordingAgentProcessTransport, type[Exception]], ...] = (
                (
                    RecordingAgentProcessTransport(
                        error=ProcessTransportTimeoutError()
                    ),
                    AgentExecutionTimeoutError,
                ),
                (
                    RecordingAgentProcessTransport(
                        ProcessResult(exit_code=7, stderr="private failure")
                    ),
                    AgentProcessFailedError,
                ),
                (
                    RecordingAgentProcessTransport(
                        ProcessResult(exit_code=1, stderr="please log in")
                    ),
                    AgentAuthenticationRequiredError,
                ),
                (
                    RecordingAgentProcessTransport(resolved=None),
                    AgentBackendUnavailableError,
                ),
            )
            for transport, error_type in cases:
                with self.subTest(error_type=error_type):
                    backend = CodexCLIAutonomousAgentBackend(
                        profile(),
                        CodexCLIInferenceConfig(
                            inherited_environment_variables=()
                        ),
                        transport,
                    )
                    with self.assertRaises(error_type):
                        await backend.execute(request(workspace))

    def test_backend_contract_and_profile_disclosure(self) -> None:
        backend = CodexCLIAutonomousAgentBackend(
            profile(),
            CodexCLIInferenceConfig(),
            RecordingAgentProcessTransport(),
        )
        self.assertIsInstance(backend, AutonomousAgentBackend)
        with self.assertRaisesRegex(ValueError, "must disclose"):
            CodexCLIAutonomousAgentBackend(
                AgentTargetProfile(
                    target_id="codex-agent/hidden-shell",
                    backend_id="codex-cli",
                    backend_kind=BackendKind.CLI,
                    adapter_version="1",
                    model_id="test-model",
                ),
                CodexCLIInferenceConfig(),
                RecordingAgentProcessTransport(),
            )


if __name__ == "__main__":
    unittest.main()

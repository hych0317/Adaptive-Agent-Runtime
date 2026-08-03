"""Isolated autonomous Codex CLI Agent backend."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from adaptive_agent_runtime.llm.agent_backends.models import (
    AgentActionKind,
    AgentActionTraceEntry,
    AutonomousAgentRequest,
    AutonomousAgentResult,
)
from adaptive_agent_runtime.llm.errors import (
    AgentAuthenticationRequiredError,
    AgentBackendUnavailableError,
    AgentExecutionBudgetError,
    AgentExecutionPolicyError,
    AgentExecutionProtocolError,
    AgentExecutionTimeoutError,
    AgentProcessFailedError,
)
from adaptive_agent_runtime.llm.models import (
    AgentTargetProfile,
    BackendKind,
    BackendProbeResult,
    InferenceUsage,
)
from adaptive_agent_runtime.llm.providers.codex_cli import (
    CodexCLIInferenceConfig,
    codex_adapter_definition,
    probe_codex_cli,
)
from adaptive_agent_runtime.llm.providers.cli_integration import (
    apply_cli_launch_environment,
    resolve_cli_launch,
)
from adaptive_agent_runtime.llm.providers.process import (
    AsyncProcessTransport,
    ProcessResult,
    ProcessTransportTimeoutError,
    ProcessTransportUnavailableError,
    inherited_environment,
)


_ACTION_KIND = {
    "commandexecution": AgentActionKind.COMMAND_EXECUTION,
    "filechange": AgentActionKind.FILE_CHANGE,
    "mcptoolcall": AgentActionKind.MCP_TOOL_CALL,
    "dynamictoolcall": AgentActionKind.MCP_TOOL_CALL,
    "websearch": AgentActionKind.WEB_SEARCH,
}


class CodexCLIAutonomousAgentBackend:
    """Run Codex inside one Runtime-supplied workspace and permission envelope."""

    module_id = "llm.agent_backend.codex_cli"

    def __init__(
        self,
        profile: AgentTargetProfile,
        config: CodexCLIInferenceConfig,
        transport: AsyncProcessTransport,
    ) -> None:
        if profile.backend_kind is not BackendKind.CLI:
            raise ValueError("Codex Agent backend requires CLI kind")
        if profile.model_id is None:
            raise ValueError("Codex Agent backend requires an explicit model ID")
        access = profile.delegated_access
        if not access.filesystem_read or not access.shell_execution:
            raise ValueError(
                "Codex Agent profile must disclose filesystem read and shell access"
            )
        if access.mcp_execution or access.arbitrary_network:
            raise ValueError(
                "Codex Agent adapter does not support MCP or arbitrary network"
            )
        if access.session_persistence:
            raise ValueError("Codex Agent adapter is always ephemeral")
        self._profile = profile
        self._config = config
        self._transport = transport

    @property
    def target_id(self) -> str:
        return self._profile.target_id

    @property
    def profile(self) -> AgentTargetProfile:
        return self._profile

    async def probe(self) -> BackendProbeResult:
        return await probe_codex_cli(
            self.target_id,
            self._config,
            self._transport,
        )

    async def execute(
        self,
        request: AutonomousAgentRequest,
    ) -> AutonomousAgentResult:
        self._ensure_policy(request)
        workspace = self._resolve_workspace(request.workspace_root)
        environment = inherited_environment(
            self._config.inherited_environment_variables
        )
        launch = resolve_cli_launch(
            codex_adapter_definition(self._config),
            self._transport,
            environment,
        )
        if launch.launch_path is None:
            raise AgentBackendUnavailableError(
                self.target_id,
                "Codex CLI executable was not found",
            )
        environment = apply_cli_launch_environment(environment, launch)
        try:
            with TemporaryDirectory(prefix="aar-codex-agent-") as temporary:
                schema_path = _write_schema(temporary, request)
                result = await self._transport.run(
                    self._arguments(
                        launch.launch_path,
                        workspace,
                        request,
                        schema_path,
                    ),
                    stdin=_prompt(request),
                    cwd=workspace,
                    environment=environment,
                    timeout_seconds=request.timeout_seconds,
                )
        except ProcessTransportTimeoutError as exc:
            raise AgentExecutionTimeoutError(self.target_id) from exc
        except ProcessTransportUnavailableError as exc:
            raise AgentBackendUnavailableError(
                self.target_id,
                "Codex CLI process could not start",
            ) from exc
        maximum = min(
            self._config.max_capture_characters,
            request.policy.max_capture_characters,
        )
        if len(result.stdout) > maximum or len(result.stderr) > maximum:
            raise AgentExecutionBudgetError(
                self.target_id,
                "captured process output exceeded its limit",
            )
        if result.exit_code != 0:
            if _looks_unauthenticated(result):
                raise AgentAuthenticationRequiredError(self.target_id)
            raise AgentProcessFailedError(self.target_id, result.exit_code)
        return _normalize_result(self.profile, request, result.stdout)

    def _ensure_policy(self, request: AutonomousAgentRequest) -> None:
        requested = request.policy.delegated_access
        available = self.profile.delegated_access
        fields = (
            "filesystem_read",
            "filesystem_write",
            "shell_execution",
            "mcp_execution",
            "arbitrary_network",
            "session_persistence",
        )
        excess = tuple(
            field
            for field in fields
            if getattr(requested, field) and not getattr(available, field)
        )
        if excess:
            raise AgentExecutionPolicyError(
                self.target_id,
                "requested access exceeds target profile: " + ", ".join(excess),
            )
        if not requested.filesystem_read or not requested.shell_execution:
            raise AgentExecutionPolicyError(
                self.target_id,
                "Codex autonomous mode inherently requires disclosed read and shell access",
            )
        if requested.mcp_execution or requested.arbitrary_network:
            raise AgentExecutionPolicyError(
                self.target_id,
                "this adapter does not enable MCP or arbitrary network",
            )
        if requested.session_persistence:
            raise AgentExecutionPolicyError(
                self.target_id,
                "this adapter requires ephemeral execution",
            )

    def _arguments(
        self,
        executable: str,
        workspace: str,
        request: AutonomousAgentRequest,
        schema_path: str | None,
    ) -> tuple[str, ...]:
        sandbox = (
            "workspace-write"
            if request.policy.delegated_access.filesystem_write
            else "read-only"
        )
        arguments = [
            executable,
            "--ask-for-approval",
            "never",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            sandbox,
            "--json",
            "--color",
            "never",
            "--skip-git-repo-check",
            "--cd",
            workspace,
            "--model",
            cast(str, self.profile.model_id),
            "-c",
            "agents.enabled=false",
            "-c",
            'web_search="disabled"',
        ]
        if schema_path is not None:
            arguments.extend(("--output-schema", schema_path))
        return tuple(arguments)

    def _resolve_workspace(self, workspace_root: str) -> str:
        path = Path(workspace_root)
        if not path.is_absolute():
            raise AgentExecutionPolicyError(
                self.target_id,
                "workspace root must be absolute",
            )
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise AgentExecutionPolicyError(
                self.target_id,
                "workspace root is unavailable",
            ) from exc
        if not resolved.is_dir():
            raise AgentExecutionPolicyError(
                self.target_id,
                "workspace root must be a directory",
            )
        return str(resolved)


def _write_schema(
    temporary: str,
    request: AutonomousAgentRequest,
) -> str | None:
    if request.response_schema is None:
        return None
    path = Path(temporary) / "agent-response.schema.json"
    schema = request.model_dump(mode="json")["response_schema"]
    path.write_text(
        json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return str(path)


def _prompt(request: AutonomousAgentRequest) -> str:
    access = request.policy.delegated_access
    boundary = {
        "workspace_root": request.workspace_root,
        "filesystem_write": access.filesystem_write,
        "network": False,
        "mcp": False,
        "session_persistence": False,
    }
    payload = request.model_dump(mode="json")["input"]
    return (
        "Complete the delegated goal only inside the declared isolated workspace. "
        "Do not access network or MCP, do not delegate to sub-agents, and do not "
        "persist a session. Respect the permission boundary exactly. Return only "
        "the final result, matching the supplied JSON Schema when present.\n\n"
        f"Goal: {request.goal}\n"
        "Permission boundary: "
        + json.dumps(boundary, ensure_ascii=False, separators=(",", ":"))
        + "\nInput: "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _normalize_result(
    profile: AgentTargetProfile,
    request: AutonomousAgentRequest,
    stdout: str,
) -> AutonomousAgentResult:
    final_text: str | None = None
    remote_request_id: str | None = None
    actions: list[AgentActionTraceEntry] = []
    usage = InferenceUsage()
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AgentExecutionProtocolError(
                profile.target_id,
                "Codex CLI emitted invalid JSONL",
            ) from exc
        if not isinstance(event, Mapping):
            raise AgentExecutionProtocolError(
                profile.target_id,
                "Codex CLI event is not an object",
            )
        event_type = event.get("type")
        if event_type == "thread.started":
            thread_id = event.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                remote_request_id = thread_id
        if event_type in {"error", "turn.failed"}:
            raise AgentExecutionProtocolError(
                profile.target_id,
                "Codex CLI reported a failed turn",
            )
        if event_type in {"item.started", "item.completed"}:
            item = event.get("item")
            if isinstance(item, Mapping):
                kind = _action_kind(item.get("type"))
                if kind is not None:
                    _enforce_action_allowed(profile.target_id, request, kind)
                    if event_type == "item.completed":
                        actions.append(
                            AgentActionTraceEntry(
                                sequence=len(actions) + 1,
                                kind=kind,
                                item_id=_optional_string(item.get("id")),
                                status=_optional_string(item.get("status")),
                            )
                        )
                elif (
                    event_type == "item.completed"
                    and _normalized(item.get("type")) == "agentmessage"
                ):
                    text = item.get("text")
                    if isinstance(text, str):
                        final_text = text
        if event_type == "turn.completed":
            usage = _usage(event.get("usage"), profile.target_id)

    if len(actions) > request.policy.max_action_events:
        raise AgentExecutionBudgetError(
            profile.target_id,
            f"{len(actions)} action events exceed limit "
            f"{request.policy.max_action_events}",
        )
    if request.policy.max_total_tokens is not None:
        if usage.total_tokens is None:
            raise AgentExecutionBudgetError(
                profile.target_id,
                "token usage is missing",
            )
        if usage.total_tokens > request.policy.max_total_tokens:
            raise AgentExecutionBudgetError(
                profile.target_id,
                f"{usage.total_tokens} tokens exceed limit "
                f"{request.policy.max_total_tokens}",
            )
    if request.policy.max_monetary_cost is not None:
        if usage.monetary_cost is None:
            raise AgentExecutionBudgetError(
                profile.target_id,
                "monetary cost is missing",
            )
        if usage.monetary_cost > request.policy.max_monetary_cost:
            raise AgentExecutionBudgetError(
                profile.target_id,
                f"{usage.monetary_cost} cost exceeds limit "
                f"{request.policy.max_monetary_cost}",
            )
    if final_text is None:
        raise AgentExecutionProtocolError(
            profile.target_id,
            "Codex CLI final agent message is missing",
        )
    output: Any = final_text
    if request.response_schema is not None:
        try:
            output = json.loads(final_text)
        except json.JSONDecodeError as exc:
            raise AgentExecutionProtocolError(
                profile.target_id,
                "Codex CLI final message is not valid JSON",
            ) from exc
        schema = request.model_dump(mode="json")["response_schema"]
        errors = tuple(Draft202012Validator(schema).iter_errors(output))
        if errors:
            raise AgentExecutionProtocolError(
                profile.target_id,
                "Codex CLI final message violates response schema",
            )
    return AutonomousAgentResult(
        request_id=request.request_id,
        target_id=profile.target_id,
        model_id=profile.model_id,
        output=output,
        usage=usage,
        actions=tuple(actions),
        remote_request_id=remote_request_id,
    )


def _enforce_action_allowed(
    target_id: str,
    request: AutonomousAgentRequest,
    kind: AgentActionKind,
) -> None:
    access = request.policy.delegated_access
    allowed = {
        AgentActionKind.COMMAND_EXECUTION: access.shell_execution,
        AgentActionKind.FILE_CHANGE: access.filesystem_write,
        AgentActionKind.MCP_TOOL_CALL: access.mcp_execution,
        AgentActionKind.WEB_SEARCH: access.arbitrary_network,
    }
    if not allowed[kind]:
        raise AgentExecutionPolicyError(
            target_id,
            f"Codex CLI attempted disallowed action '{kind.value}'",
        )


def _action_kind(value: Any) -> AgentActionKind | None:
    return _ACTION_KIND.get(_normalized(value))


def _normalized(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _usage(value: Any, target_id: str) -> InferenceUsage:
    if not isinstance(value, Mapping):
        return InferenceUsage()
    input_tokens = _token_count(value.get("input_tokens"), target_id)
    output_tokens = _token_count(value.get("output_tokens"), target_id)
    total = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    return InferenceUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
    )


def _token_count(value: Any, target_id: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AgentExecutionProtocolError(
            target_id,
            "Codex CLI token usage is invalid",
        )
    return cast(int, value)


def _looks_unauthenticated(result: ProcessResult) -> bool:
    text = (result.stdout + "\n" + result.stderr).lower()
    return any(
        marker in text
        for marker in (
            "not logged in",
            "login required",
            "authentication required",
            "please log in",
        )
    )

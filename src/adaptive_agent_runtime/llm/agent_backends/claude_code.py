"""Strictly sandboxed autonomous Claude Code Agent backend."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any, cast

from jsonschema import (  # type: ignore[import-untyped]
    Draft7Validator,
    Draft202012Validator,
)
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
from pydantic import Field, model_validator

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
from adaptive_agent_runtime.llm.json_types import ImmutableJsonValue
from adaptive_agent_runtime.llm.models import (
    AgentTargetProfile,
    BackendKind,
    BackendProbeResult,
    InferenceUsage,
)
from adaptive_agent_runtime.llm.providers.claude_code import (
    ClaudeCodeInferenceConfig,
    probe_claude_code,
)
from adaptive_agent_runtime.llm.providers.process import (
    AsyncProcessTransport,
    ProcessResult,
    ProcessTransportTimeoutError,
    ProcessTransportUnavailableError,
    inherited_environment,
)


_AUTONOMOUS_MINIMUM_VERSION = (2, 1, 219)
_UNSUPPORTED_DRAFT_2020_KEYWORDS = {
    "$dynamicAnchor",
    "$dynamicRef",
    "dependentRequired",
    "dependentSchemas",
    "maxContains",
    "minContains",
    "prefixItems",
    "unevaluatedItems",
    "unevaluatedProperties",
}
_DENIED_TOOLS = (
    "Read",
    "Edit",
    "Write",
    "NotebookEdit",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "Agent",
    "Task",
    "Skill",
    "mcp__*",
)


class ClaudeCodeAutonomousAgentConfig(ClaudeCodeInferenceConfig):
    minimum_runtime_version: str = Field(default="2.1.219", min_length=1)
    default_max_turns: int = Field(default=16, ge=1)

    @model_validator(mode="after")
    def validate_autonomous_version(self) -> ClaudeCodeAutonomousAgentConfig:
        parsed = _version_tuple(self.minimum_runtime_version)
        if parsed is None or parsed < _AUTONOMOUS_MINIMUM_VERSION:
            raise ValueError(
                "Claude autonomous mode requires version 2.1.219 or later"
            )
        return self


class ClaudeCodeAutonomousAgentBackend:
    """Delegate a bounded task while retaining Runtime authority."""

    module_id = "llm.agent_backend.claude_code"

    def __init__(
        self,
        profile: AgentTargetProfile,
        config: ClaudeCodeAutonomousAgentConfig,
        transport: AsyncProcessTransport,
    ) -> None:
        if profile.backend_kind is not BackendKind.CLI:
            raise ValueError("Claude Code Agent backend requires CLI kind")
        if profile.model_id is None:
            raise ValueError(
                "Claude Code Agent backend requires an explicit model ID"
            )
        access = profile.delegated_access
        if not access.filesystem_read or not access.shell_execution:
            raise ValueError(
                "Claude Code Agent profile must disclose filesystem read and "
                "shell access"
            )
        if access.mcp_execution or access.arbitrary_network:
            raise ValueError(
                "Claude Code Agent adapter does not support MCP or network"
            )
        if access.session_persistence:
            raise ValueError("Claude Code Agent adapter is always ephemeral")
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
        return await probe_claude_code(
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
        executable = self._transport.resolve(self._config.executable)
        if executable is None:
            raise AgentBackendUnavailableError(
                self.target_id,
                "Claude Code executable was not found",
            )
        schema = _draft7_schema(request, self.target_id)
        encoded_schema = (
            json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            if schema is not None
            else None
        )
        if (
            encoded_schema is not None
            and len(encoded_schema) > self._config.max_schema_characters
        ):
            raise AgentExecutionPolicyError(
                self.target_id,
                "response schema exceeds the configured limit",
            )
        settings = _sandbox_settings(request, workspace)
        environment = inherited_environment(
            self._config.inherited_environment_variables
        )
        try:
            result = await self._transport.run(
                self._arguments(
                    executable,
                    request,
                    encoded_schema,
                    settings,
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
                "Claude Code process could not start",
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
        return _normalize_stream(self.profile, request, result.stdout)

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
                "requested access exceeds target profile: "
                + ", ".join(excess),
            )
        if not requested.filesystem_read or not requested.shell_execution:
            raise AgentExecutionPolicyError(
                self.target_id,
                "Claude autonomous mode requires disclosed read and shell access",
            )
        if requested.mcp_execution or requested.arbitrary_network:
            raise AgentExecutionPolicyError(
                self.target_id,
                "this adapter forbids MCP and arbitrary network",
            )
        if requested.session_persistence:
            raise AgentExecutionPolicyError(
                self.target_id,
                "this adapter requires ephemeral execution",
            )
        if request.policy.allowed_tool_names:
            raise AgentExecutionPolicyError(
                self.target_id,
                "this adapter exposes only its fixed sandboxed Bash tool",
            )

    def _arguments(
        self,
        executable: str,
        request: AutonomousAgentRequest,
        encoded_schema: str | None,
        settings: str,
    ) -> tuple[str, ...]:
        maximum_turns = request.policy.max_agent_turns
        if maximum_turns is None:
            maximum_turns = self._config.default_max_turns
        else:
            maximum_turns = min(
                maximum_turns,
                self._config.default_max_turns,
            )
        arguments = [
            executable,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            cast(str, self.profile.model_id),
            "--max-turns",
            str(maximum_turns),
            "--permission-mode",
            "dontAsk",
            "--tools",
            "Bash",
            "--disallowedTools",
            "mcp__*",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--no-chrome",
            "--bare",
            "--safe-mode",
            "--settings",
            settings,
        ]
        if request.policy.max_monetary_cost is not None:
            arguments.extend(
                (
                    "--max-budget-usd",
                    str(request.policy.max_monetary_cost),
                )
            )
        if encoded_schema is not None:
            arguments.extend(("--json-schema", encoded_schema))
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


def _sandbox_settings(
    request: AutonomousAgentRequest,
    workspace: str,
) -> str:
    filesystem: dict[str, object] = {
        "denyRead": ["~/"],
        "allowRead": [workspace],
    }
    if not request.policy.delegated_access.filesystem_write:
        filesystem["denyWrite"] = [workspace]
    settings = {
        "permissions": {
            "defaultMode": "dontAsk",
            "allow": ["Bash"],
            "deny": list(_DENIED_TOOLS),
            "disableBypassPermissionsMode": "disable",
            "disableAutoMode": "disable",
        },
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "allowUnsandboxedCommands": False,
            "autoAllowBashIfSandboxed": True,
            "filesystem": filesystem,
            "network": {
                "allowedDomains": [],
                "strictAllowlist": True,
            },
            "credentials": {
                "files": [
                    {"path": "~/.aws", "mode": "deny"},
                    {"path": "~/.azure", "mode": "deny"},
                    {"path": "~/.config/gcloud", "mode": "deny"},
                    {"path": "~/.kube", "mode": "deny"},
                    {"path": "~/.ssh", "mode": "deny"},
                ]
            },
        },
    }
    return json.dumps(settings, ensure_ascii=False, separators=(",", ":"))


def _prompt(request: AutonomousAgentRequest) -> str:
    access = request.policy.delegated_access
    boundary = {
        "workspace_root": request.workspace_root,
        "filesystem_write": access.filesystem_write,
        "tool": "sandboxed Bash only",
        "network": False,
        "mcp": False,
        "subagents": False,
        "session_persistence": False,
    }
    payload = request.model_dump(mode="json")["input"]
    return (
        "Complete only the delegated goal inside the declared isolated "
        "workspace. Never access network, MCP, sub-agents, or persistent "
        "state. Use only sandboxed Bash and respect the permission boundary.\n\n"
        f"Goal: {request.goal}\nPermission boundary: "
        + json.dumps(boundary, ensure_ascii=False, separators=(",", ":"))
        + "\nInput: "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _normalize_stream(
    profile: AgentTargetProfile,
    request: AutonomousAgentRequest,
    stdout: str,
) -> AutonomousAgentResult:
    observed: list[tuple[AgentActionKind, str | None]] = []
    statuses: dict[str, str] = {}
    result_event: Mapping[str, Any] | None = None
    remote_request_id: str | None = None
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AgentExecutionProtocolError(
                profile.target_id,
                "Claude Code emitted invalid stream JSON",
            ) from exc
        if not isinstance(event, Mapping):
            raise AgentExecutionProtocolError(
                profile.target_id,
                "Claude Code stream event is not an object",
            )
        session_id = event.get("session_id")
        if isinstance(session_id, str) and session_id:
            remote_request_id = session_id
        event_type = event.get("type")
        if event_type == "assistant":
            for block in _content_blocks(event):
                if block.get("type") != "tool_use":
                    continue
                tool_name = block.get("name")
                tool_id = _optional_string(block.get("id"))
                kind = _action_kind(profile.target_id, tool_name)
                _enforce_action_allowed(profile.target_id, request, kind)
                observed.append((kind, tool_id))
        elif event_type == "user":
            for block in _content_blocks(event):
                if block.get("type") != "tool_result":
                    continue
                tool_id = _optional_string(block.get("tool_use_id"))
                if tool_id is not None:
                    statuses[tool_id] = (
                        "failed" if block.get("is_error") is True else "completed"
                    )
        elif event_type == "result":
            if event.get("subtype") != "success" or event.get("is_error") is True:
                raise AgentExecutionProtocolError(
                    profile.target_id,
                    "Claude Code reported a failed Agent result",
                )
            result_event = event

    if result_event is None:
        raise AgentExecutionProtocolError(
            profile.target_id,
            "Claude Code final Agent result is missing",
        )
    if len(observed) > request.policy.max_action_events:
        raise AgentExecutionBudgetError(
            profile.target_id,
            f"{len(observed)} action events exceed limit "
            f"{request.policy.max_action_events}",
        )
    usage = _usage(result_event, profile.target_id)
    _enforce_usage_budget(profile.target_id, request, usage)
    output = _result_output(result_event, request, profile.target_id)
    actions = tuple(
        AgentActionTraceEntry(
            sequence=index,
            kind=kind,
            item_id=item_id,
            status=(statuses.get(item_id, "requested") if item_id else "requested"),
        )
        for index, (kind, item_id) in enumerate(observed, start=1)
    )
    return AutonomousAgentResult(
        request_id=request.request_id,
        target_id=profile.target_id,
        model_id=profile.model_id,
        output=cast(ImmutableJsonValue, output),
        usage=usage,
        actions=actions,
        remote_request_id=remote_request_id,
    )


def _content_blocks(event: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    message = event.get("message")
    message_map = message if isinstance(message, Mapping) else event
    content = message_map.get("content")
    if not isinstance(content, list):
        return ()
    return tuple(item for item in content if isinstance(item, Mapping))


def _action_kind(target_id: str, tool_name: Any) -> AgentActionKind:
    if tool_name in {"Bash", "PowerShell"}:
        return AgentActionKind.COMMAND_EXECUTION
    if isinstance(tool_name, str) and tool_name.startswith("mcp__"):
        return AgentActionKind.MCP_TOOL_CALL
    if tool_name in {"WebFetch", "WebSearch"}:
        return AgentActionKind.WEB_SEARCH
    if tool_name in {"Edit", "Write", "NotebookEdit"}:
        return AgentActionKind.FILE_CHANGE
    raise AgentExecutionProtocolError(
        target_id,
        "Claude Code emitted an unknown tool action",
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
            f"Claude Code attempted disallowed action '{kind.value}'",
        )


def _result_output(
    result: Mapping[str, Any],
    request: AutonomousAgentRequest,
    target_id: str,
) -> Any:
    if request.response_schema is None:
        output = result.get("result")
        if not isinstance(output, str):
            raise AgentExecutionProtocolError(
                target_id,
                "Claude Code text result is missing",
            )
        return output
    output = result.get("structured_output")
    if output is None:
        raise AgentExecutionProtocolError(
            target_id,
            "Claude Code structured Agent result is missing",
        )
    schema = request.model_dump(mode="json")["response_schema"]
    errors = tuple(Draft202012Validator(schema).iter_errors(output))
    if errors:
        raise AgentExecutionProtocolError(
            target_id,
            "Claude Code Agent result violates response schema",
        )
    return output


def _usage(result: Mapping[str, Any], target_id: str) -> InferenceUsage:
    raw_usage = result.get("usage")
    usage = raw_usage if isinstance(raw_usage, Mapping) else {}
    input_tokens = _token_count(usage.get("input_tokens"), target_id)
    output_tokens = _token_count(usage.get("output_tokens"), target_id)
    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    raw_cost = result.get("total_cost_usd")
    monetary_cost: float | None = None
    if raw_cost is not None:
        if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
            raise AgentExecutionProtocolError(
                target_id,
                "Claude Code Agent cost is invalid",
            )
        monetary_cost = float(raw_cost)
        if monetary_cost < 0 or not isfinite(monetary_cost):
            raise AgentExecutionProtocolError(
                target_id,
                "Claude Code Agent cost is invalid",
            )
    return InferenceUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        monetary_cost=monetary_cost,
        currency="USD" if monetary_cost is not None else None,
    )


def _enforce_usage_budget(
    target_id: str,
    request: AutonomousAgentRequest,
    usage: InferenceUsage,
) -> None:
    if request.policy.max_total_tokens is not None:
        if usage.total_tokens is None:
            raise AgentExecutionBudgetError(target_id, "token usage is missing")
        if usage.total_tokens > request.policy.max_total_tokens:
            raise AgentExecutionBudgetError(
                target_id,
                f"{usage.total_tokens} tokens exceed limit "
                f"{request.policy.max_total_tokens}",
            )
    if request.policy.max_monetary_cost is not None:
        if usage.monetary_cost is None:
            raise AgentExecutionBudgetError(target_id, "cost usage is missing")
        if usage.monetary_cost > request.policy.max_monetary_cost:
            raise AgentExecutionBudgetError(
                target_id,
                f"{usage.monetary_cost} cost exceeds limit "
                f"{request.policy.max_monetary_cost}",
            )


def _draft7_schema(
    request: AutonomousAgentRequest,
    target_id: str,
) -> dict[str, Any] | None:
    raw = request.model_dump(mode="json")["response_schema"]
    if raw is None:
        return None

    def convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            converted: dict[str, Any] = {}
            for key, item in value.items():
                if key in _UNSUPPORTED_DRAFT_2020_KEYWORDS:
                    raise AgentExecutionPolicyError(
                        target_id,
                        f"unsupported response schema keyword '{key}'",
                    )
                target_key = "definitions" if key == "$defs" else key
                converted[target_key] = convert(item)
            schema_uri = converted.get("$schema")
            if isinstance(schema_uri, str) and "2020-12" in schema_uri:
                converted["$schema"] = (
                    "http://json-schema.org/draft-07/schema#"
                )
            reference = converted.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                converted["$ref"] = reference.replace(
                    "#/$defs/", "#/definitions/", 1
                )
            return converted
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    converted_schema = convert(raw)
    if not isinstance(converted_schema, dict):
        raise AgentExecutionPolicyError(target_id, "response schema is invalid")
    try:
        Draft7Validator.check_schema(converted_schema)
    except SchemaError as exc:
        raise AgentExecutionPolicyError(
            target_id,
            "response schema is not supported by Claude Code",
        ) from exc
    return converted_schema


def _token_count(value: Any, target_id: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AgentExecutionProtocolError(
            target_id,
            "Claude Code Agent token usage is invalid",
        )
    return cast(int, value)


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _looks_unauthenticated(result: ProcessResult) -> bool:
    text = (result.stdout + "\n" + result.stderr).lower()
    return any(
        marker in text
        for marker in (
            "not logged in",
            "login required",
            "authentication required",
            "please run /login",
        )
    )


def _version_tuple(value: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)

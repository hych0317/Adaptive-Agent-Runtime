"""Audited, inference-only compatibility adapter for Codex CLI OAuth sessions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from pydantic import Field, model_validator

from adaptive_agent_runtime.llm.errors import (
    AuthenticationRequiredError,
    BackendProcessFailedError,
    BackendProtocolError,
    BackendUnavailableError,
    InferenceTimeoutError,
    MalformedModelOutputError,
    UnsupportedFeatureError,
)
from adaptive_agent_runtime.llm.json_types import LLMModel
from adaptive_agent_runtime.llm.models import (
    BackendAuthentication,
    BackendAvailability,
    BackendKind,
    BackendLimits,
    BackendMetering,
    BackendProbeResult,
    BackendTransportFeatures,
    InferenceRequest,
    InferenceTargetProfile,
    InferenceUsage,
    ModelResponseKind,
    NormalizedFinishReason,
    NormalizedModelResponse,
    ReasoningEffort,
    StructuredOutputLevel,
    ToolIntentMode,
)
from adaptive_agent_runtime.llm.providers.process import (
    AsyncProcessTransport,
    ProcessResult,
    ProcessTransportTimeoutError,
    ProcessTransportUnavailableError,
    inherited_environment,
)
from adaptive_agent_runtime.llm.providers.cli_integration import (
    CLIAdapterDefinition,
    CLIAuthStatus,
    apply_cli_launch_environment,
    detect_cli_adapter,
    parse_codex_debug_models,
    resolve_cli_launch,
)


_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_ENVIRONMENT = (
    "PATH",
    "PATHEXT",
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "LOCALAPPDATA",
    "APPDATA",
    "HOME",
    "CODEX_HOME",
    "CODEX_BIN",
    "CODEX_API_KEY",
    "OPENAI_API_KEY",
    "LANG",
    "LC_ALL",
    "TERM",
    "CODEX_CA_CERTIFICATE",
    "SSL_CERT_FILE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)
_FORBIDDEN_ITEM_TYPES = {
    "commandexecution",
    "filechange",
    "mcptoolcall",
    "dynamictoolcall",
    "websearch",
    "toolcall",
    "collabagenttoolcall",
}


class CodexCLIAuthProbeMode(StrEnum):
    """How strongly ``codex login status`` gates target availability."""

    ADVISORY = "advisory"
    REQUIRED = "required"


class CodexCLIInferenceConfig(LLMModel):
    executable: str = Field(default="codex", min_length=1)
    probe_timeout_seconds: float = Field(default=10.0, gt=0.0)
    max_capture_characters: int = Field(default=2_000_000, ge=1)
    auth_probe_mode: CodexCLIAuthProbeMode = CodexCLIAuthProbeMode.ADVISORY
    inherited_environment_variables: tuple[str, ...] = _DEFAULT_ENVIRONMENT
    reasoning_effort: ReasoningEffort | None = None

    @model_validator(mode="after")
    def validate_config(self) -> CodexCLIInferenceConfig:
        names = self.inherited_environment_variables
        if len(set(names)) != len(names):
            raise ValueError("inherited environment variables must be unique")
        if any(not _ENVIRONMENT_NAME.fullmatch(name) for name in names):
            raise ValueError("inherited environment variable name is invalid")
        if self.reasoning_effort not in {
            None,
            ReasoningEffort.DEFAULT,
            ReasoningEffort.NONE,
            ReasoningEffort.MINIMAL,
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
            ReasoningEffort.XHIGH,
        }:
            raise ValueError(
                "Codex CLI reasoning effort must be none, minimal, low, "
                "medium, high, xhigh, or default"
            )
        return self


class CodexCLIInferenceTargetDefinition(LLMModel):
    """Explicit model/profile definition for an installed Codex CLI session."""

    target_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    features: BackendTransportFeatures
    supported_cognitive_capability_ids: tuple[str, ...] = Field(min_length=1)
    config: CodexCLIInferenceConfig = Field(
        default_factory=CodexCLIInferenceConfig
    )
    limits: BackendLimits = Field(default_factory=BackendLimits)
    metering: BackendMetering = Field(
        default_factory=lambda: BackendMetering(reports_token_usage=True)
    )
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_definition(self) -> CodexCLIInferenceTargetDefinition:
        if self.features.tool_intent:
            raise ValueError("Codex CLI inference cannot negotiate tool intent")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("Codex CLI target tags must be unique")
        if len(set(self.supported_cognitive_capability_ids)) != len(
            self.supported_cognitive_capability_ids
        ):
            raise ValueError("Codex CLI cognitive capabilities must be unique")
        return self

    def build_profile(self) -> InferenceTargetProfile:
        return InferenceTargetProfile(
            target_id=self.target_id,
            backend_id="codex-cli.inference",
            backend_kind=BackendKind.CLI,
            adapter_version="1",
            model_id=self.model_id,
            features=self.features,
            supported_cognitive_capability_ids=(
                self.supported_cognitive_capability_ids
            ),
            limits=self.limits,
            authentication=BackendAuthentication(
                supported_methods=(
                    "chatgpt_oauth",
                    "api_key",
                    "access_token",
                    "cli_session",
                )
            ),
            metering=self.metering,
            tags=self.tags,
        )

    def build_backend(
        self,
        transport: AsyncProcessTransport,
    ) -> CodexCLIInferenceBackend:
        return CodexCLIInferenceBackend(
            self.build_profile(),
            self.config,
            transport,
        )


class CodexCLIInferenceBackend:
    """Use ``codex exec`` as audited inference, never as Runtime authority.

    Codex remains an agent internally. This adapter constrains it to an empty,
    read-only, ephemeral workspace and rejects any JSONL trace containing an
    action item. It therefore offers compatibility inference, not a raw model
    transport or proof that the CLI contains no internal agent loop.
    """

    module_id = "llm.backend.codex_cli_inference"

    def __init__(
        self,
        profile: InferenceTargetProfile,
        config: CodexCLIInferenceConfig,
        transport: AsyncProcessTransport,
    ) -> None:
        if profile.backend_kind is not BackendKind.CLI:
            raise ValueError("Codex CLI backend requires CLI kind")
        if profile.model_id is None:
            raise ValueError("Codex CLI backend requires an explicit model ID")
        if profile.features.tool_intent:
            raise ValueError("Codex CLI inference cannot expose tool intent")
        self._profile = profile
        self._config = config
        self._transport = transport

    @property
    def target_id(self) -> str:
        return self._profile.target_id

    @property
    def profile(self) -> InferenceTargetProfile:
        return self._profile

    async def probe(self) -> BackendProbeResult:
        return await probe_codex_cli(
            self.target_id,
            self._config,
            self._transport,
        )

    async def invoke(self, request: InferenceRequest) -> NormalizedModelResponse:
        self._ensure_supported(request)
        environment = inherited_environment(
            self._config.inherited_environment_variables
        )
        launch = resolve_cli_launch(
            codex_adapter_definition(self._config),
            self._transport,
            environment,
        )
        if launch.launch_path is None:
            raise BackendUnavailableError(
                self.target_id,
                reason="Codex CLI executable was not found",
                retryable=False,
            )
        environment = apply_cli_launch_environment(environment, launch)
        try:
            with TemporaryDirectory(prefix="aar-codex-inference-") as workspace:
                schema_path = self._write_schema(workspace, request)
                argv = self._arguments(
                    launch.launch_path,
                    workspace,
                    schema_path,
                )
                result = await self._transport.run(
                    argv,
                    stdin=_prompt(request),
                    cwd=workspace,
                    environment=environment,
                    timeout_seconds=(
                        request.timeout_seconds
                        or self.profile.limits.default_timeout_seconds
                    ),
                )
        except ProcessTransportTimeoutError as exc:
            raise InferenceTimeoutError(self.target_id) from exc
        except ProcessTransportUnavailableError as exc:
            raise BackendUnavailableError(
                self.target_id,
                reason="Codex CLI process could not start",
                retryable=False,
            ) from exc
        self._enforce_capture_limit(result)
        if result.exit_code != 0:
            if _looks_unauthenticated(result):
                raise AuthenticationRequiredError(self.target_id)
            raise BackendProcessFailedError(self.target_id, result.exit_code)
        return _normalize_jsonl(self.profile, request, result.stdout)

    def _ensure_supported(self, request: InferenceRequest) -> None:
        if request.requirements.tool_intent is not ToolIntentMode.DISABLED:
            raise UnsupportedFeatureError(self.target_id, "tool_intent")
        supported = self.profile.features.structured_output
        required = request.requirements.required_structured_output
        rank = {
            StructuredOutputLevel.NONE: 0,
            StructuredOutputLevel.JSON_OBJECT: 1,
            StructuredOutputLevel.JSON_SCHEMA: 2,
        }
        if rank[required] > rank[supported]:
            raise UnsupportedFeatureError(self.target_id, "structured_output")
        if request.requirements.max_output_tokens is not None:
            raise UnsupportedFeatureError(self.target_id, "max_output_tokens")

    @staticmethod
    def _write_schema(
        workspace: str,
        request: InferenceRequest,
    ) -> str | None:
        if request.response_schema is None:
            return None
        path = Path(workspace) / "response.schema.json"
        schema = request.model_dump(mode="json")["response_schema"]
        path.write_text(
            json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        return str(path)

    def _arguments(
        self,
        executable: str,
        workspace: str,
        schema_path: str | None,
    ) -> tuple[str, ...]:
        arguments = [
            executable,
            "--ask-for-approval",
            "never",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
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
        if (
            self._config.reasoning_effort is not None
            and self._config.reasoning_effort is not ReasoningEffort.DEFAULT
        ):
            arguments.extend(
                (
                    "-c",
                    "model_reasoning_effort="
                    + json.dumps(self._config.reasoning_effort.value),
                )
            )
        if schema_path is not None:
            arguments.extend(("--output-schema", schema_path))
        return tuple(arguments)

    def _enforce_capture_limit(self, result: ProcessResult) -> None:
        maximum = self._config.max_capture_characters
        if len(result.stdout) > maximum or len(result.stderr) > maximum:
            raise BackendProtocolError(
                self.target_id,
                "CLI output exceeded the configured capture limit",
            )


def _prompt(request: InferenceRequest) -> str:
    payload = request.model_dump(mode="json")["input"]
    return (
        "Act only as a bounded inference engine. Do not inspect files, run "
        "commands, call tools, use MCP, browse, delegate, persist memory, or "
        "modify any state. Return only the final response. The response must "
        "match the supplied JSON Schema when one is present.\n\n"
        "Runtime inference input:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


async def probe_codex_cli(
    target_id: str,
    config: CodexCLIInferenceConfig,
    transport: AsyncProcessTransport,
) -> BackendProbeResult:
    """Probe Codex using the Open Design availability/auth separation."""

    environment = inherited_environment(config.inherited_environment_variables)
    outcome = await detect_cli_adapter(
        codex_adapter_definition(config),
        transport,
        environment,
        auth_method_classifier=_authentication_method,
    )
    if not outcome.invocable:
        return _unavailable_probe(
            target_id,
            outcome.diagnostics[0] if outcome.diagnostics else "unavailable",
        )
    if (
        config.auth_probe_mode is CodexCLIAuthProbeMode.REQUIRED
        and outcome.auth_status is not CLIAuthStatus.OK
    ):
        return BackendProbeResult(
            target_id=target_id,
            availability=BackendAvailability.AUTH_REQUIRED,
            runtime_version=outcome.runtime_version,
            protocol_version="codex-exec-jsonl-v1",
            available_model_ids=outcome.available_model_ids,
            diagnostics=("login_required",),
        )
    return BackendProbeResult(
        target_id=target_id,
        availability=BackendAvailability.AVAILABLE,
        runtime_version=outcome.runtime_version,
        active_auth_method=outcome.active_auth_method,
        protocol_version="codex-exec-jsonl-v1",
        available_model_ids=outcome.available_model_ids,
        diagnostics=outcome.diagnostics,
    )


def codex_adapter_definition(
    config: CodexCLIInferenceConfig,
) -> CLIAdapterDefinition:
    return CLIAdapterDefinition(
        adapter_id="codex",
        name="Codex CLI",
        executable=config.executable,
        version_args=("--version",),
        version_timeout_seconds=config.probe_timeout_seconds,
        auth_args=("login", "status"),
        auth_timeout_seconds=config.probe_timeout_seconds,
        model_args=("debug", "models"),
        model_timeout_seconds=min(config.probe_timeout_seconds, 5.0),
        model_parser=parse_codex_debug_models,
        fallback_models=(
            "default",
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.3-codex",
            "gpt-5.1",
            "gpt-5.1-codex-mini",
            "gpt-5-codex",
            "gpt-5",
            "o3",
            "o4-mini",
        ),
        executable_override_env="CODEX_BIN",
        auth_satisfying_env=("CODEX_API_KEY", "OPENAI_API_KEY"),
        protocol_version="codex-exec-jsonl-v1",
    )


def _unavailable_probe(
    target_id: str,
    diagnostic: str,
) -> BackendProbeResult:
    return BackendProbeResult(
        target_id=target_id,
        availability=BackendAvailability.UNAVAILABLE,
        protocol_version="codex-exec-jsonl-v1",
        diagnostics=(diagnostic,),
    )


def _authentication_method(result: ProcessResult) -> str:
    text = (result.stdout + "\n" + result.stderr).lower()
    if "chatgpt" in text:
        return "chatgpt_oauth"
    if "api key" in text or "api_key" in text:
        return "api_key"
    if "access token" in text:
        return "access_token"
    return "cli_session"


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


def _normalize_jsonl(
    profile: InferenceTargetProfile,
    request: InferenceRequest,
    stdout: str,
) -> NormalizedModelResponse:
    events: list[Mapping[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MalformedModelOutputError(
                profile.target_id,
                "Codex CLI emitted invalid JSONL",
            ) from exc
        if not isinstance(value, Mapping):
            raise MalformedModelOutputError(
                profile.target_id,
                "Codex CLI event is not an object",
            )
        events.append(cast(Mapping[str, Any], value))

    final_text: str | None = None
    usage = InferenceUsage()
    remote_request_id: str | None = None
    for event in events:
        event_type = event.get("type")
        if event_type == "thread.started":
            thread_id = event.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                remote_request_id = thread_id
        if event_type in {"error", "turn.failed"}:
            raise BackendProtocolError(
                profile.target_id,
                "Codex CLI reported a failed turn",
            )
        if event_type in {"item.started", "item.completed"}:
            item = event.get("item")
            if isinstance(item, Mapping):
                item_type = item.get("type")
                if isinstance(item_type, str):
                    normalized = re.sub(r"[^a-z0-9]", "", item_type.lower())
                    if normalized in _FORBIDDEN_ITEM_TYPES:
                        raise BackendProtocolError(
                            profile.target_id,
                            "Codex CLI attempted an internal action",
                        )
                    if (
                        event_type == "item.completed"
                        and normalized == "agentmessage"
                    ):
                        text = item.get("text")
                        if isinstance(text, str):
                            final_text = text
        if event_type == "turn.completed":
            usage = _normalize_usage(event.get("usage"), profile.target_id)

    if final_text is None:
        raise MalformedModelOutputError(
            profile.target_id,
            "Codex CLI final agent message is missing",
        )
    output: Any = final_text
    if (
        request.requirements.required_structured_output
        is not StructuredOutputLevel.NONE
        or request.response_schema is not None
    ):
        try:
            output = json.loads(final_text)
        except json.JSONDecodeError as exc:
            raise MalformedModelOutputError(
                profile.target_id,
                "Codex CLI final message is not valid JSON",
            ) from exc
    return NormalizedModelResponse(
        request_id=request.request_id,
        target_id=profile.target_id,
        model_id=profile.model_id,
        kind=ModelResponseKind.OUTPUT,
        output=output,
        usage=usage,
        finish_reason=NormalizedFinishReason.COMPLETED,
        remote_request_id=remote_request_id,
    )


def _normalize_usage(value: Any, target_id: str) -> InferenceUsage:
    if not isinstance(value, Mapping):
        return InferenceUsage()
    input_tokens = _nonnegative_int(value.get("input_tokens"), target_id)
    output_tokens = _nonnegative_int(value.get("output_tokens"), target_id)
    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    return InferenceUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _nonnegative_int(value: Any, target_id: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MalformedModelOutputError(
            target_id,
            "Codex CLI token usage is invalid",
        )
    return cast(int, value)

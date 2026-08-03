"""Hardened inference-only compatibility adapter for Claude Code sessions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from math import isfinite
from tempfile import TemporaryDirectory
from typing import Any, cast

from jsonschema import Draft7Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
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
from adaptive_agent_runtime.llm.json_types import ImmutableJsonValue, LLMModel
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


_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VERSION = re.compile(r"\b(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)\b")
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
    "CLAUDE_CONFIG_DIR",
    "LANG",
    "LC_ALL",
    "TERM",
    "SSL_CERT_FILE",
    "NODE_EXTRA_CA_CERTS",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
)
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


class ClaudeCodeInferenceConfig(LLMModel):
    executable: str = Field(default="claude", min_length=1)
    minimum_runtime_version: str = Field(default="2.1.205", min_length=1)
    probe_timeout_seconds: float = Field(default=10.0, gt=0.0)
    max_capture_characters: int = Field(default=2_000_000, ge=1)
    max_schema_characters: int = Field(default=24_000, ge=1)
    inherited_environment_variables: tuple[str, ...] = _DEFAULT_ENVIRONMENT

    @model_validator(mode="after")
    def validate_config(self) -> ClaudeCodeInferenceConfig:
        names = self.inherited_environment_variables
        if len(set(names)) != len(names):
            raise ValueError("inherited environment variables must be unique")
        if any(not _ENVIRONMENT_NAME.fullmatch(name) for name in names):
            raise ValueError("inherited environment variable name is invalid")
        if _version_tuple(self.minimum_runtime_version) is None:
            raise ValueError("minimum Claude Code version is invalid")
        return self


class ClaudeCodeInferenceTargetDefinition(LLMModel):
    target_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    features: BackendTransportFeatures
    supported_cognitive_capability_ids: tuple[str, ...] = Field(min_length=1)
    config: ClaudeCodeInferenceConfig = Field(
        default_factory=ClaudeCodeInferenceConfig
    )
    limits: BackendLimits = Field(default_factory=BackendLimits)
    metering: BackendMetering = Field(
        default_factory=lambda: BackendMetering(
            reports_token_usage=True,
            reports_monetary_cost=True,
        )
    )
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_definition(self) -> ClaudeCodeInferenceTargetDefinition:
        if self.features.tool_intent:
            raise ValueError("Claude Code inference cannot expose ToolIntent")
        if self.features.structured_output is not StructuredOutputLevel.JSON_SCHEMA:
            raise ValueError("Claude Code managed capabilities require JSON Schema")
        if len(set(self.supported_cognitive_capability_ids)) != len(
            self.supported_cognitive_capability_ids
        ):
            raise ValueError("Claude Code cognitive capabilities must be unique")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("Claude Code target tags must be unique")
        return self

    def build_profile(self) -> InferenceTargetProfile:
        return InferenceTargetProfile(
            target_id=self.target_id,
            backend_id="claude-code.inference",
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
                    "claude_ai_oauth",
                    "console_api_key",
                    "cloud_provider",
                )
            ),
            metering=self.metering,
            tags=self.tags,
        )

    def build_backend(
        self,
        transport: AsyncProcessTransport,
    ) -> ClaudeCodeInferenceBackend:
        return ClaudeCodeInferenceBackend(
            self.build_profile(),
            self.config,
            transport,
        )


class ClaudeCodeInferenceBackend:
    """Run Claude Code with no tools, MCP, project config, or persistence."""

    module_id = "llm.backend.claude_code_inference"

    def __init__(
        self,
        profile: InferenceTargetProfile,
        config: ClaudeCodeInferenceConfig,
        transport: AsyncProcessTransport,
    ) -> None:
        if profile.backend_kind is not BackendKind.CLI:
            raise ValueError("Claude Code backend requires CLI kind")
        if profile.model_id is None:
            raise ValueError("Claude Code backend requires an explicit model ID")
        if profile.features.tool_intent:
            raise ValueError("Claude Code inference cannot expose ToolIntent")
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
        return await probe_claude_code(
            self.target_id,
            self._config,
            self._transport,
        )
    async def invoke(self, request: InferenceRequest) -> NormalizedModelResponse:
        self._ensure_supported(request)
        executable = self._transport.resolve(self._config.executable)
        if executable is None:
            raise BackendUnavailableError(
                self.target_id,
                reason="Claude Code executable was not found",
                retryable=False,
            )
        schema = _draft7_schema(request, self.target_id)
        encoded_schema = json.dumps(
            schema,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(encoded_schema) > self._config.max_schema_characters:
            raise UnsupportedFeatureError(self.target_id, "schema_size")
        environment = inherited_environment(
            self._config.inherited_environment_variables
        )
        arguments = self._arguments(executable, encoded_schema)
        try:
            with TemporaryDirectory(prefix="aar-claude-inference-") as workspace:
                result = await self._transport.run(
                    arguments,
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
                reason="Claude Code process could not start",
                retryable=False,
            ) from exc
        self._enforce_capture_limit(result)
        if result.exit_code != 0:
            if _looks_unauthenticated(result):
                raise AuthenticationRequiredError(self.target_id)
            raise BackendProcessFailedError(self.target_id, result.exit_code)
        return _normalize_result(self.profile, request, result.stdout)

    def _ensure_supported(self, request: InferenceRequest) -> None:
        if request.requirements.tool_intent is not ToolIntentMode.DISABLED:
            raise UnsupportedFeatureError(self.target_id, "tool_intent")
        if request.requirements.max_output_tokens is not None:
            raise UnsupportedFeatureError(self.target_id, "max_output_tokens")
        if request.response_schema is None:
            raise UnsupportedFeatureError(self.target_id, "response_schema")
        if (
            request.requirements.required_structured_output
            is not StructuredOutputLevel.JSON_SCHEMA
        ):
            raise UnsupportedFeatureError(self.target_id, "structured_output")

    def _arguments(
        self,
        executable: str,
        encoded_schema: str,
    ) -> tuple[str, ...]:
        return (
            executable,
            "-p",
            "--output-format",
            "json",
            "--model",
            cast(str, self.profile.model_id),
            "--max-turns",
            "1",
            "--permission-mode",
            "dontAsk",
            "--tools",
            "",
            "--disallowedTools",
            "*",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--no-chrome",
            "--bare",
            "--safe-mode",
            "--json-schema",
            encoded_schema,
        )

    def _enforce_capture_limit(self, result: ProcessResult) -> None:
        maximum = self._config.max_capture_characters
        if len(result.stdout) > maximum or len(result.stderr) > maximum:
            raise BackendProtocolError(
                self.target_id,
                "CLI output exceeded the configured capture limit",
            )


async def probe_claude_code(
    target_id: str,
    config: ClaudeCodeInferenceConfig,
    transport: AsyncProcessTransport,
) -> BackendProbeResult:
    executable = transport.resolve(config.executable)
    if executable is None:
        return _unavailable_probe(target_id, "executable_not_found")
    environment = inherited_environment(config.inherited_environment_variables)
    try:
        with TemporaryDirectory(prefix="aar-claude-probe-") as workspace:
            version = await transport.run(
                (executable, "--version"),
                stdin=None,
                cwd=workspace,
                environment=environment,
                timeout_seconds=config.probe_timeout_seconds,
            )
            if version.exit_code != 0:
                return _unavailable_probe(target_id, "version_probe_failed")
            auth = await transport.run(
                (executable, "auth", "status"),
                stdin=None,
                cwd=workspace,
                environment=environment,
                timeout_seconds=config.probe_timeout_seconds,
            )
    except ProcessTransportTimeoutError:
        return _unavailable_probe(target_id, "probe_timed_out")
    except ProcessTransportUnavailableError:
        return _unavailable_probe(target_id, "executable_not_runnable")
    runtime_version = _runtime_version(version)
    if not _version_at_least(runtime_version, config.minimum_runtime_version):
        return BackendProbeResult(
            target_id=target_id,
            availability=BackendAvailability.UNAVAILABLE,
            runtime_version=runtime_version,
            protocol_version="claude-code-json-v1",
            diagnostics=("unsupported_runtime_version",),
        )
    if auth.exit_code != 0:
        return BackendProbeResult(
            target_id=target_id,
            availability=BackendAvailability.AUTH_REQUIRED,
            runtime_version=runtime_version,
            protocol_version="claude-code-json-v1",
            diagnostics=("login_required",),
        )
    return BackendProbeResult(
        target_id=target_id,
        availability=BackendAvailability.AVAILABLE,
        runtime_version=runtime_version,
        active_auth_method=_authentication_method(auth),
        protocol_version="claude-code-json-v1",
    )


def _prompt(request: InferenceRequest) -> str:
    payload = request.model_dump(mode="json")["input"]
    return (
        "Act only as a bounded inference engine. Do not inspect files, run "
        "commands, call tools, use MCP, browse, delegate, persist memory, or "
        "modify state. Return the requested structured output from this Runtime "
        "input:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _draft7_schema(request: InferenceRequest, target_id: str) -> dict[str, Any]:
    raw = request.model_dump(mode="json")["response_schema"]
    if not isinstance(raw, Mapping):
        raise UnsupportedFeatureError(target_id, "response_schema")

    def convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            converted: dict[str, Any] = {}
            for key, item in value.items():
                if key in _UNSUPPORTED_DRAFT_2020_KEYWORDS:
                    raise UnsupportedFeatureError(
                        target_id,
                        f"schema_keyword:{key}",
                    )
                target_key = "definitions" if key == "$defs" else key
                converted[target_key] = convert(item)
            schema_uri = converted.get("$schema")
            if isinstance(schema_uri, str) and "2020-12" in schema_uri:
                converted["$schema"] = "http://json-schema.org/draft-07/schema#"
            reference = converted.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                converted["$ref"] = reference.replace(
                    "#/$defs/",
                    "#/definitions/",
                    1,
                )
            return converted
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    schema = convert(raw)
    if not isinstance(schema, dict):
        raise UnsupportedFeatureError(target_id, "response_schema")
    try:
        Draft7Validator.check_schema(schema)
    except SchemaError as exc:
        raise UnsupportedFeatureError(
            target_id,
            "draft7_response_schema",
        ) from exc
    return schema


def _normalize_result(
    profile: InferenceTargetProfile,
    request: InferenceRequest,
    stdout: str,
) -> NormalizedModelResponse:
    try:
        raw = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise MalformedModelOutputError(
            profile.target_id,
            "Claude Code output is not valid JSON",
        ) from exc
    if not isinstance(raw, Mapping):
        raise MalformedModelOutputError(
            profile.target_id,
            "Claude Code result is not an object",
        )
    if raw.get("type") != "result" or raw.get("subtype") != "success":
        raise BackendProtocolError(
            profile.target_id,
            "Claude Code did not report a successful result",
        )
    if raw.get("is_error") is True:
        raise BackendProtocolError(
            profile.target_id,
            "Claude Code marked the result as an error",
        )
    output = raw.get("structured_output")
    if not isinstance(output, Mapping):
        raise MalformedModelOutputError(
            profile.target_id,
            "Claude Code structured_output is missing",
        )
    session_id = raw.get("session_id")
    return NormalizedModelResponse(
        request_id=request.request_id,
        target_id=profile.target_id,
        model_id=profile.model_id,
        kind=ModelResponseKind.OUTPUT,
        output=cast(ImmutableJsonValue, output),
        usage=_usage(raw, profile.target_id),
        finish_reason=NormalizedFinishReason.COMPLETED,
        remote_request_id=(session_id if isinstance(session_id, str) else None),
    )


def _usage(raw: Mapping[str, Any], target_id: str) -> InferenceUsage:
    usage = raw.get("usage")
    usage_map = usage if isinstance(usage, Mapping) else {}
    input_tokens = _nonnegative_int(
        usage_map.get("input_tokens"), target_id, "input_tokens"
    )
    output_tokens = _nonnegative_int(
        usage_map.get("output_tokens"), target_id, "output_tokens"
    )
    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    cost = raw.get("total_cost_usd")
    monetary_cost: float | None = None
    if cost is not None:
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            raise MalformedModelOutputError(
                target_id,
                "Claude Code total_cost_usd is invalid",
            )
        monetary_cost = float(cost)
        if monetary_cost < 0 or not isfinite(monetary_cost):
            raise MalformedModelOutputError(
                target_id,
                "Claude Code total_cost_usd is invalid",
            )
    return InferenceUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        monetary_cost=monetary_cost,
        currency="USD" if monetary_cost is not None else None,
    )


def _nonnegative_int(
    value: Any,
    target_id: str,
    field: str,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MalformedModelOutputError(
            target_id,
            f"Claude Code usage field '{field}' is invalid",
        )
    return cast(int, value)


def _runtime_version(result: ProcessResult) -> str | None:
    match = _VERSION.search(result.stdout + "\n" + result.stderr)
    return match.group(1) if match else None


def _version_tuple(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _version_at_least(actual: str | None, minimum: str) -> bool:
    actual_tuple = _version_tuple(actual)
    minimum_tuple = _version_tuple(minimum)
    return (
        actual_tuple is not None
        and minimum_tuple is not None
        and actual_tuple >= minimum_tuple
    )


def _authentication_method(result: ProcessResult) -> str:
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return "cli_session"
    if not isinstance(value, Mapping):
        return "cli_session"
    method = value.get("authMethod", value.get("auth_method"))
    return method if isinstance(method, str) and method else "cli_session"


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


def _unavailable_probe(target_id: str, diagnostic: str) -> BackendProbeResult:
    return BackendProbeResult(
        target_id=target_id,
        availability=BackendAvailability.UNAVAILABLE,
        protocol_version="claude-code-json-v1",
        diagnostics=(diagnostic,),
    )

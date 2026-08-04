"""Provider-native Anthropic Messages API inference backend."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import quote, urlparse

from pydantic import Field, SecretStr, model_validator

from adaptive_agent_runtime.llm.errors import (
    AuthenticationRequiredError,
    BackendRequestRejectedError,
    BackendUnavailableError,
    ContextOverflowError,
    InferenceTimeoutError,
    MalformedModelOutputError,
    RateLimitedError,
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
    ReasoningEffort,
    StructuredOutputLevel,
    ToolIntentDraft,
    ToolIntentMode,
)
from adaptive_agent_runtime.llm.providers.http import (
    AsyncJSONTransport,
    HTTPJSONResponse,
    HTTPTransportTimeoutError,
    HTTPTransportUnavailableError,
)


_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VERSION_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CONTEXT_ERROR_TYPES = {
    "request_too_large",
    "model_context_window_exceeded",
}
_STRUCTURED_OUTPUT_RANK = {
    StructuredOutputLevel.NONE: 0,
    StructuredOutputLevel.JSON_OBJECT: 1,
    StructuredOutputLevel.JSON_SCHEMA: 2,
}


class AnthropicMessagesConfig(LLMModel):
    base_url: str = Field(default="https://api.anthropic.com/v1", min_length=1)
    api_key_env: str = Field(default="ANTHROPIC_API_KEY", min_length=1)
    api_key: SecretStr | None = None
    anthropic_version: str = Field(default="2023-06-01", min_length=1)
    default_max_tokens: int = Field(default=4096, ge=1)
    probe_timeout_seconds: float = Field(default=10.0, gt=0.0)
    allow_insecure_http: bool = False
    max_schema_characters: int = Field(default=24_000, ge=1)
    reasoning_effort: ReasoningEffort | None = None

    @model_validator(mode="after")
    def validate_config(self) -> AnthropicMessagesConfig:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Anthropic base URL must be HTTP(S)")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "Anthropic base URL cannot contain credentials, query, or fragment"
            )
        loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if (
            parsed.scheme == "http"
            and not loopback
            and not self.allow_insecure_http
        ):
            raise ValueError("non-local Anthropic HTTP requires explicit opt in")
        if not _ENVIRONMENT_NAME.fullmatch(self.api_key_env):
            raise ValueError("Anthropic API key environment variable is invalid")
        if not _VERSION_DATE.fullmatch(self.anthropic_version):
            raise ValueError("Anthropic protocol version must be YYYY-MM-DD")
        return self

    @property
    def messages_url(self) -> str:
        return self.base_url.rstrip("/") + "/messages"

    @property
    def models_url(self) -> str:
        return self.base_url.rstrip("/") + "/models?limit=1000"

    def model_url(self, model_id: str) -> str:
        return self.base_url.rstrip("/") + "/models/" + quote(model_id, safe="")


class AnthropicAPITargetDefinition(LLMModel):
    target_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    features: BackendTransportFeatures
    supported_cognitive_capability_ids: tuple[str, ...] = Field(min_length=1)
    config: AnthropicMessagesConfig = Field(
        default_factory=AnthropicMessagesConfig
    )
    limits: BackendLimits = Field(default_factory=BackendLimits)
    metering: BackendMetering = Field(
        default_factory=lambda: BackendMetering(reports_token_usage=True)
    )
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_definition(self) -> AnthropicAPITargetDefinition:
        if len(set(self.supported_cognitive_capability_ids)) != len(
            self.supported_cognitive_capability_ids
        ):
            raise ValueError("Anthropic cognitive capabilities must be unique")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("Anthropic target tags must be unique")
        if self.features.multimodal:
            raise ValueError(
                "Anthropic Messages adapter does not support multimodal input"
            )
        if self.config.reasoning_effort not in {
            None,
            ReasoningEffort.DEFAULT,
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
            ReasoningEffort.XHIGH,
            ReasoningEffort.MAX,
        }:
            raise ValueError(
                "Anthropic reasoning effort must be low, medium, high, "
                "xhigh, max, or default"
            )
        return self

    def build_profile(self) -> InferenceTargetProfile:
        return InferenceTargetProfile(
            target_id=self.target_id,
            backend_id="anthropic.messages",
            backend_kind=BackendKind.API,
            adapter_version="1",
            model_id=self.model_id,
            features=self.features,
            supported_cognitive_capability_ids=(
                self.supported_cognitive_capability_ids
            ),
            limits=self.limits,
            authentication=BackendAuthentication(
                supported_methods=(
                    "x_api_key_env",
                    *(("x_api_key_config",) if self.config.api_key is not None else ()),
                )
            ),
            metering=self.metering,
            tags=self.tags,
        )

    def build_backend(
        self,
        transport: AsyncJSONTransport,
    ) -> AnthropicMessagesBackend:
        return AnthropicMessagesBackend(
            self.build_profile(),
            self.config,
            transport,
        )


class AnthropicMessagesBackend:
    """Normalize Anthropic Messages into Runtime-owned inference results."""

    module_id = "llm.backend.anthropic_messages"

    def __init__(
        self,
        profile: InferenceTargetProfile,
        config: AnthropicMessagesConfig,
        transport: AsyncJSONTransport,
    ) -> None:
        if profile.backend_kind is not BackendKind.API:
            raise ValueError("Anthropic Messages backend requires API kind")
        if profile.model_id is None:
            raise ValueError("Anthropic Messages backend requires a model ID")
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
        protocol_version = self._protocol_version
        api_key, auth_method = self._api_key()
        if not api_key:
            return BackendProbeResult(
                target_id=self.target_id,
                availability=BackendAvailability.AUTH_REQUIRED,
                protocol_version=protocol_version,
                diagnostics=("api_key_missing",),
            )
        try:
            response = await self._transport.get_json(
                self._config.models_url,
                headers=self._headers(api_key, content_type=False),
                timeout_seconds=self._config.probe_timeout_seconds,
            )
        except HTTPTransportTimeoutError:
            return _probe_unavailable(
                self.target_id, "probe_timed_out", protocol_version
            )
        except HTTPTransportUnavailableError:
            return _probe_unavailable(
                self.target_id, "transport_unavailable", protocol_version
            )
        if response.status_code in {401, 403}:
            return BackendProbeResult(
                target_id=self.target_id,
                availability=BackendAvailability.AUTH_REQUIRED,
                protocol_version=protocol_version,
                diagnostics=("credentials_rejected",),
            )
        if response.status_code == 429:
            return _probe_unavailable(
                self.target_id, "probe_rate_limited", protocol_version
            )
        if response.status_code >= 500:
            return _probe_unavailable(
                self.target_id, "provider_unavailable", protocol_version
            )
        if not 200 <= response.status_code < 300:
            return _probe_unavailable(
                self.target_id, "probe_rejected", protocol_version
            )
        models = _anthropic_models(response.body)
        if models is None:
            return _probe_unavailable(
                self.target_id,
                "invalid_models_response",
                protocol_version,
            )
        available_model_ids = tuple(models)
        model = models.get(cast(str, self.profile.model_id))
        if model is None:
            return _probe_unavailable(
                self.target_id,
                "model_not_available",
                protocol_version,
                available_model_ids=available_model_ids,
            )
        if (
            self.profile.features.structured_output
            is StructuredOutputLevel.JSON_SCHEMA
            and _structured_output_supported(model) is False
        ):
            return _probe_unavailable(
                self.target_id,
                "structured_output_not_supported",
                protocol_version,
            )
        return BackendProbeResult(
            target_id=self.target_id,
            availability=BackendAvailability.AVAILABLE,
            active_auth_method=auth_method,
            protocol_version=protocol_version,
            available_model_ids=available_model_ids,
            diagnostics=("connectivity_and_model_verified",),
        )

    async def invoke(self, request: InferenceRequest) -> NormalizedModelResponse:
        self._ensure_supported(request)
        api_key, _ = self._api_key()
        if not api_key:
            raise AuthenticationRequiredError(self.target_id)
        body, tool_names = self._build_body(request)
        timeout = (
            request.timeout_seconds
            or self.profile.limits.default_timeout_seconds
        )
        try:
            response = await self._transport.post_json(
                self._config.messages_url,
                headers=self._headers(api_key, content_type=True),
                body=body,
                timeout_seconds=timeout,
            )
        except HTTPTransportTimeoutError as exc:
            raise InferenceTimeoutError(self.target_id) from exc
        except HTTPTransportUnavailableError as exc:
            raise BackendUnavailableError(
                self.target_id,
                reason="Anthropic HTTP transport unavailable",
            ) from exc
        self._raise_for_status(response)
        return self._normalize_response(request, response, tool_names)

    def _api_key(self) -> tuple[str | None, str | None]:
        value = os.environ.get(self._config.api_key_env)
        if value:
            return value, "x_api_key_env"
        if self._config.api_key is not None:
            value = self._config.api_key.get_secret_value()
            if value:
                return value, "x_api_key_config"
        return None, None

    @property
    def _protocol_version(self) -> str:
        return f"anthropic-messages-{self._config.anthropic_version}"

    def _headers(
        self,
        api_key: str,
        *,
        content_type: bool,
    ) -> dict[str, str]:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": self._config.anthropic_version,
            "Accept": "application/json",
        }
        if content_type:
            headers["Content-Type"] = "application/json"
        return headers

    def _ensure_supported(self, request: InferenceRequest) -> None:
        required = request.requirements.required_structured_output
        supported = self.profile.features.structured_output
        if _STRUCTURED_OUTPUT_RANK[required] > _STRUCTURED_OUTPUT_RANK[supported]:
            raise UnsupportedFeatureError(self.target_id, "structured_output")
        if required is StructuredOutputLevel.JSON_SCHEMA:
            if request.response_schema is None:
                raise UnsupportedFeatureError(self.target_id, "response_schema")
        if (
            request.requirements.tool_intent is not ToolIntentMode.DISABLED
            and not self.profile.features.tool_intent
        ):
            raise UnsupportedFeatureError(self.target_id, "tool_intent")
        requested_tokens = request.requirements.max_output_tokens
        supported_tokens = self.profile.limits.max_output_tokens
        if (
            requested_tokens is not None
            and supported_tokens is not None
            and requested_tokens > supported_tokens
        ):
            raise UnsupportedFeatureError(self.target_id, "max_output_tokens")

    def _build_body(
        self,
        request: InferenceRequest,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        payload = request.model_dump(mode="json")["input"]
        maximum = request.requirements.max_output_tokens
        if maximum is None:
            maximum = self._config.default_max_tokens
            profile_limit = self.profile.limits.max_output_tokens
            if profile_limit is not None:
                maximum = min(maximum, profile_limit)
        body: dict[str, Any] = {
            "model": self.profile.model_id,
            "max_tokens": maximum,
            "system": (
                "You are a bounded inference provider inside Adaptive Agent "
                "Runtime. Return only the requested result. Client tool calls "
                "are proposals for Runtime and were not executed by you."
            ),
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            ],
        }
        if (
            self._config.reasoning_effort is not None
            and self._config.reasoning_effort is not ReasoningEffort.DEFAULT
        ):
            body["output_config"] = {
                "effort": self._config.reasoning_effort.value
            }
        if request.response_schema is not None:
            schema = request.model_dump(mode="json")["response_schema"]
            encoded = json.dumps(
                schema,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if len(encoded) > self._config.max_schema_characters:
                raise UnsupportedFeatureError(self.target_id, "schema_size")
            output_config = cast(
                dict[str, Any], body.setdefault("output_config", {})
            )
            output_config["format"] = {
                "type": "json_schema",
                "schema": schema,
            }
        tool_names: dict[str, str] = {}
        if request.requirements.tool_intent is not ToolIntentMode.DISABLED:
            tools: list[dict[str, Any]] = []
            for index, tool in enumerate(request.eligible_tools):
                provider_name = f"runtime_tool_{index}"
                tool_names[provider_name] = tool.capability_id
                tools.append(
                    {
                        "name": provider_name,
                        "description": tool.description,
                        "input_schema": tool.model_dump(mode="json")[
                            "input_schema"
                        ],
                        "strict": True,
                    }
                )
            body["tools"] = tools
            body["tool_choice"] = {
                "type": (
                    "any"
                    if request.requirements.tool_intent
                    is ToolIntentMode.REQUIRED
                    else "auto"
                )
            }
        return body, tool_names

    def _raise_for_status(self, response: HTTPJSONResponse) -> None:
        status = response.status_code
        if 200 <= status < 300:
            return
        if status in {401, 403}:
            raise AuthenticationRequiredError(self.target_id)
        if status == 429:
            raise RateLimitedError(self.target_id)
        if status in {408, 504}:
            raise InferenceTimeoutError(self.target_id)
        if status in {413} or _error_type(response.body) in _CONTEXT_ERROR_TYPES:
            raise ContextOverflowError(self.target_id)
        if status >= 500:
            raise BackendUnavailableError(
                self.target_id,
                reason=f"Anthropic returned HTTP {status}",
            )
        raise BackendRequestRejectedError(self.target_id, status)

    def _normalize_response(
        self,
        request: InferenceRequest,
        response: HTTPJSONResponse,
        tool_names: Mapping[str, str],
    ) -> NormalizedModelResponse:
        body = _mapping(response.body, self.target_id, "response body")
        if body.get("type") != "message" or body.get("role") != "assistant":
            raise MalformedModelOutputError(
                self.target_id,
                "Anthropic response identity is invalid",
            )
        raw_content = body.get("content")
        if not isinstance(raw_content, (list, tuple)):
            raise MalformedModelOutputError(
                self.target_id,
                "Anthropic response content is missing",
            )
        texts: list[str] = []
        intents: list[ToolIntentDraft] = []
        for raw_block in raw_content:
            block = _mapping(raw_block, self.target_id, "content block")
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise MalformedModelOutputError(
                        self.target_id,
                        "Anthropic text block is invalid",
                    )
                texts.append(text)
            elif block_type == "tool_use":
                intents.append(
                    _tool_intent(block, tool_names, self.target_id)
                )
            else:
                raise MalformedModelOutputError(
                    self.target_id,
                    "Anthropic returned an unrequested content block",
                )
        stop_reason = body.get("stop_reason")
        if intents:
            if stop_reason != "tool_use":
                raise MalformedModelOutputError(
                    self.target_id,
                    "Anthropic tool blocks lack tool_use stop reason",
                )
            kind = ModelResponseKind.TOOL_INTENT
            output: Any = None
            finish_reason = NormalizedFinishReason.TOOL_INTENT
        else:
            if stop_reason == "tool_use":
                raise MalformedModelOutputError(
                    self.target_id,
                    "Anthropic tool_use stop lacks client tool blocks",
                )
            if request.requirements.tool_intent is ToolIntentMode.REQUIRED:
                raise MalformedModelOutputError(
                    self.target_id,
                    "Anthropic omitted a required ToolIntent",
                )
            if not texts:
                raise MalformedModelOutputError(
                    self.target_id,
                    "Anthropic text output is missing",
                )
            text = "".join(texts)
            if (
                request.requirements.required_structured_output
                is not StructuredOutputLevel.NONE
                or request.response_schema is not None
            ):
                try:
                    output = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise MalformedModelOutputError(
                        self.target_id,
                        "Anthropic structured output is not valid JSON",
                    ) from exc
            else:
                output = text
            kind = ModelResponseKind.OUTPUT
            finish_reason = _finish_reason(stop_reason)
        remote_id = body.get("id")
        return NormalizedModelResponse(
            request_id=request.request_id,
            target_id=self.target_id,
            model_id=self.profile.model_id,
            kind=kind,
            output=cast(ImmutableJsonValue, output),
            tool_intents=tuple(intents),
            usage=_usage(body.get("usage"), self.target_id),
            finish_reason=finish_reason,
            remote_request_id=(
                remote_id if isinstance(remote_id, str) and remote_id else None
            ),
        )


def _tool_intent(
    block: Mapping[str, Any],
    tool_names: Mapping[str, str],
    target_id: str,
) -> ToolIntentDraft:
    call_id = block.get("id")
    name = block.get("name")
    arguments = block.get("input")
    if not isinstance(call_id, str) or not call_id:
        raise MalformedModelOutputError(target_id, "tool call ID is missing")
    if not isinstance(name, str) or name not in tool_names:
        raise MalformedModelOutputError(target_id, "tool name is not eligible")
    if not isinstance(arguments, Mapping):
        raise MalformedModelOutputError(target_id, "tool input is not an object")
    return ToolIntentDraft(
        call_key=call_id,
        capability_id=tool_names[name],
        arguments=cast(dict[str, Any], dict(arguments)),
    )


def _usage(value: Any, target_id: str) -> InferenceUsage:
    usage = _mapping(value, target_id, "usage")
    direct_input = _nonnegative_int(
        usage.get("input_tokens"), target_id, "input_tokens"
    )
    cache_creation = _nonnegative_int(
        usage.get("cache_creation_input_tokens"),
        target_id,
        "cache_creation_input_tokens",
        optional=True,
    )
    cache_read = _nonnegative_int(
        usage.get("cache_read_input_tokens"),
        target_id,
        "cache_read_input_tokens",
        optional=True,
    )
    output_tokens = _nonnegative_int(
        usage.get("output_tokens"), target_id, "output_tokens"
    )
    input_tokens = direct_input + cache_creation + cache_read
    return InferenceUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _nonnegative_int(
    value: Any,
    target_id: str,
    field: str,
    *,
    optional: bool = False,
) -> int:
    if value is None and optional:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MalformedModelOutputError(
            target_id,
            f"Anthropic usage field '{field}' is invalid",
        )
    return cast(int, value)


def _finish_reason(value: Any) -> NormalizedFinishReason:
    reasons = {
        "end_turn": NormalizedFinishReason.COMPLETED,
        "stop_sequence": NormalizedFinishReason.COMPLETED,
        "max_tokens": NormalizedFinishReason.OUTPUT_LIMIT,
        "model_context_window_exceeded": NormalizedFinishReason.OUTPUT_LIMIT,
        "refusal": NormalizedFinishReason.CONTENT_FILTERED,
    }
    return reasons.get(value, NormalizedFinishReason.UNKNOWN)


def _structured_output_supported(model: Mapping[str, Any]) -> bool | None:
    capabilities = _mapping_or_none(model.get("capabilities"))
    if capabilities is None:
        return None
    structured = _mapping_or_none(capabilities.get("structured_outputs"))
    if structured is None:
        return None
    supported = structured.get("supported")
    return supported if isinstance(supported, bool) else None


def _anthropic_models(
    body: Any,
) -> dict[str, Mapping[str, Any]] | None:
    root = _mapping_or_none(body)
    if root is None:
        return None
    data = root.get("data")
    if not isinstance(data, (list, tuple)):
        return None
    models: dict[str, Mapping[str, Any]] = {}
    for item in data:
        model = _mapping_or_none(item)
        if model is None:
            return None
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id:
            return None
        models.setdefault(model_id, model)
    return models


def _error_type(body: Any) -> str | None:
    mapping = _mapping_or_none(body)
    if mapping is None:
        return None
    error = _mapping_or_none(mapping.get("error"))
    if error is None:
        return None
    value = error.get("type")
    return value if isinstance(value, str) else None


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return cast(Mapping[str, Any], value)


def _mapping(value: Any, target_id: str, subject: str) -> Mapping[str, Any]:
    mapping = _mapping_or_none(value)
    if mapping is None:
        raise MalformedModelOutputError(
            target_id,
            f"Anthropic {subject} is not an object",
        )
    return mapping


def _probe_unavailable(
    target_id: str,
    diagnostic: str,
    protocol_version: str,
    *,
    available_model_ids: tuple[str, ...] = (),
) -> BackendProbeResult:
    return BackendProbeResult(
        target_id=target_id,
        availability=BackendAvailability.UNAVAILABLE,
        protocol_version=protocol_version,
        available_model_ids=available_model_ids,
        diagnostics=(diagnostic,),
    )

"""OpenAI-compatible non-streaming Chat Completions inference backend."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, cast
from urllib.parse import urlparse

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
from adaptive_agent_runtime.llm.json_types import LLMModel
from adaptive_agent_runtime.llm.models import (
    BackendAvailability,
    BackendKind,
    BackendProbeResult,
    InferenceRequest,
    InferenceTargetProfile,
    InferenceUsage,
    ModelResponseKind,
    NormalizedFinishReason,
    NormalizedModelResponse,
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
_CONTEXT_ERROR_CODES = {
    "context_length_exceeded",
    "context_window_exceeded",
    "max_tokens_exceeded",
}
_SAFE_RESPONSE_NAME = re.compile(r"[^A-Za-z0-9_-]")
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ChatMaxTokensParameter(StrEnum):
    MAX_TOKENS = "max_tokens"
    MAX_COMPLETION_TOKENS = "max_completion_tokens"


class OpenAICompatibleProbeMode(StrEnum):
    CREDENTIALS_ONLY = "credentials_only"
    MODELS_ENDPOINT = "models_endpoint"


class OpenAICompatibleChatConfig(LLMModel):
    base_url: str = Field(min_length=1)
    api_key_env: str | None = Field(default="OPENAI_API_KEY", min_length=1)
    api_key: SecretStr | None = None
    requires_api_key: bool = True
    allow_insecure_http: bool = False
    max_tokens_parameter: ChatMaxTokensParameter = (
        ChatMaxTokensParameter.MAX_TOKENS
    )
    strict_json_schema: bool = False
    probe_mode: OpenAICompatibleProbeMode = (
        OpenAICompatibleProbeMode.CREDENTIALS_ONLY
    )
    probe_timeout_seconds: float = Field(default=10.0, gt=0.0)

    @model_validator(mode="after")
    def validate_config(self) -> OpenAICompatibleChatConfig:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("OpenAI-compatible base URL must be HTTP(S)")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base URL cannot contain credentials, query, or fragment")
        loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if (
            parsed.scheme == "http"
            and not loopback
            and not self.allow_insecure_http
        ):
            raise ValueError("non-local HTTP requires allow_insecure_http")
        if (
            self.api_key_env is not None
            and not _ENVIRONMENT_NAME.fullmatch(self.api_key_env)
        ):
            raise ValueError("API key environment variable name is invalid")
        if self.requires_api_key and self.api_key_env is None and self.api_key is None:
            raise ValueError("authenticated provider requires a credential source")
        return self

    @property
    def completions_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @property
    def models_url(self) -> str:
        return self.base_url.rstrip("/") + "/models"


class OpenAICompatibleChatBackend:
    """Normalize a bounded Chat Completion into Runtime-owned response models."""

    module_id = "llm.backend.openai_compatible_chat"

    def __init__(
        self,
        profile: InferenceTargetProfile,
        config: OpenAICompatibleChatConfig,
        transport: AsyncJSONTransport,
    ) -> None:
        if profile.backend_kind not in {BackendKind.API, BackendKind.LOCAL}:
            raise ValueError("OpenAI-compatible backend requires API or local kind")
        if profile.model_id is None:
            raise ValueError("OpenAI-compatible backend requires a model ID")
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
        api_key, auth_method = self._api_key()
        if self._config.requires_api_key and not api_key:
            return BackendProbeResult(
                target_id=self.target_id,
                availability=BackendAvailability.AUTH_REQUIRED,
                protocol_version="openai-chat-completions",
            )
        if (
            self._config.probe_mode
            is OpenAICompatibleProbeMode.MODELS_ENDPOINT
        ):
            return await self._probe_models_endpoint(api_key, auth_method)
        return BackendProbeResult(
            target_id=self.target_id,
            availability=BackendAvailability.AVAILABLE,
            active_auth_method=(
                auth_method if self._config.requires_api_key else None
            ),
            protocol_version="openai-chat-completions",
            diagnostics=("connectivity_unverified",),
        )

    async def _probe_models_endpoint(
        self,
        api_key: str | None,
        auth_method: str | None,
    ) -> BackendProbeResult:
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            response = await self._transport.get_json(
                self._config.models_url,
                headers=headers,
                timeout_seconds=self._config.probe_timeout_seconds,
            )
        except HTTPTransportTimeoutError:
            return _probe_unavailable(self.target_id, "probe_timed_out")
        except HTTPTransportUnavailableError:
            return _probe_unavailable(
                self.target_id,
                "transport_unavailable",
            )
        if response.status_code in {401, 403}:
            return BackendProbeResult(
                target_id=self.target_id,
                availability=BackendAvailability.AUTH_REQUIRED,
                protocol_version="openai-chat-completions",
                diagnostics=("credentials_rejected",),
            )
        if response.status_code == 429:
            return _probe_unavailable(self.target_id, "probe_rate_limited")
        if response.status_code >= 500:
            return _probe_unavailable(self.target_id, "provider_unavailable")
        if not 200 <= response.status_code < 300:
            return _probe_unavailable(self.target_id, "probe_rejected")
        model_ids = _model_ids(response.body)
        if model_ids is None:
            return _probe_unavailable(
                self.target_id,
                "invalid_models_response",
            )
        model_id = cast(str, self.profile.model_id)
        if model_id not in model_ids:
            return _probe_unavailable(self.target_id, "model_not_available")
        return BackendProbeResult(
            target_id=self.target_id,
            availability=BackendAvailability.AVAILABLE,
            active_auth_method=(
                auth_method if self._config.requires_api_key else None
            ),
            protocol_version="openai-chat-completions",
            diagnostics=("connectivity_and_model_verified",),
        )

    async def invoke(self, request: InferenceRequest) -> NormalizedModelResponse:
        self._ensure_supported(request)
        api_key, _ = self._api_key()
        if self._config.requires_api_key and not api_key:
            raise AuthenticationRequiredError(self.target_id)
        body, tool_names = self._build_body(request)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        timeout = (
            request.timeout_seconds
            or self.profile.limits.default_timeout_seconds
        )
        try:
            raw = await self._transport.post_json(
                self._config.completions_url,
                headers=headers,
                body=body,
                timeout_seconds=timeout,
            )
        except HTTPTransportTimeoutError as exc:
            raise InferenceTimeoutError(self.target_id) from exc
        except HTTPTransportUnavailableError as exc:
            raise BackendUnavailableError(
                self.target_id,
                reason="HTTP transport unavailable",
            ) from exc
        self._raise_for_status(raw)
        return self._normalize_response(request, raw, tool_names)

    def _api_key(self) -> tuple[str | None, str | None]:
        if not self._config.requires_api_key:
            return None, None
        if self._config.api_key_env is not None:
            value = os.environ.get(self._config.api_key_env)
            if value:
                return value, "bearer_env"
        if self._config.api_key is not None:
            value = self._config.api_key.get_secret_value()
            if value:
                return value, "bearer_config"
        return None, None

    def _ensure_supported(self, request: InferenceRequest) -> None:
        required = request.requirements.required_structured_output
        supported = self.profile.features.structured_output
        rank = {
            StructuredOutputLevel.NONE: 0,
            StructuredOutputLevel.JSON_OBJECT: 1,
            StructuredOutputLevel.JSON_SCHEMA: 2,
        }
        if rank[required] > rank[supported]:
            raise UnsupportedFeatureError(self.target_id, "structured_output")
        if (
            request.requirements.tool_intent is not ToolIntentMode.DISABLED
            and not self.profile.features.tool_intent
        ):
            raise UnsupportedFeatureError(self.target_id, "tool_intent")

    def _build_body(
        self,
        request: InferenceRequest,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        structured = (
            request.requirements.required_structured_output
            is not StructuredOutputLevel.NONE
            or request.response_schema is not None
        )
        instruction = (
            "You are an inference provider inside Adaptive Agent Runtime. "
            "Return only valid JSON matching the requested response format. "
            "Tool calls are proposals for Runtime; never claim they were executed."
            if structured
            else (
                "You are an inference provider inside Adaptive Agent Runtime. "
                "Return only the final answer. Tool calls are proposals for Runtime; "
                "never claim they were executed."
            )
        )
        dumped_request = request.model_dump(mode="json")
        payload = dumped_request["input"]
        user_payload: Any = payload
        if (
            request.response_schema is not None
            and request.requirements.required_structured_output
            is StructuredOutputLevel.JSON_OBJECT
        ):
            user_payload = {
                "input": payload,
                "response_schema": dumped_request["response_schema"],
            }
        body: dict[str, Any] = {
            "model": self.profile.model_id,
            "messages": [
                {"role": "system", "content": instruction},
                {
                    "role": "user",
                    "content": json.dumps(
                        user_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "stream": False,
        }
        requirements = request.requirements
        if requirements.max_output_tokens is not None:
            body[self._config.max_tokens_parameter.value] = (
                requirements.max_output_tokens
            )
        if (
            requirements.required_structured_output
            is StructuredOutputLevel.JSON_SCHEMA
        ):
            schema = dumped_request["response_schema"]
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _response_name(request.cognitive_capability_id),
                    "strict": self._config.strict_json_schema,
                    "schema": schema,
                },
            }
        elif (
            requirements.required_structured_output
            is StructuredOutputLevel.JSON_OBJECT
        ):
            body["response_format"] = {"type": "json_object"}

        tool_names: dict[str, str] = {}
        if (
            requirements.tool_intent is not ToolIntentMode.DISABLED
            and request.eligible_tools
        ):
            tools: list[dict[str, Any]] = []
            for index, tool in enumerate(request.eligible_tools):
                remote_name = f"runtime_tool_{index}"
                tool_names[remote_name] = tool.capability_id
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": remote_name,
                            "description": tool.description,
                            "parameters": tool.model_dump(mode="json")[
                                "input_schema"
                            ],
                        },
                    }
                )
            body["tools"] = tools
            body["tool_choice"] = (
                "required"
                if requirements.tool_intent is ToolIntentMode.REQUIRED
                else "auto"
            )
        return body, tool_names

    def _raise_for_status(self, response: HTTPJSONResponse) -> None:
        status = response.status_code
        if 200 <= status < 300:
            return
        if status == 401:
            raise AuthenticationRequiredError(self.target_id)
        if status == 429:
            raise RateLimitedError(self.target_id)
        if status in {408, 504}:
            raise InferenceTimeoutError(self.target_id)
        if status >= 500:
            raise BackendUnavailableError(
                self.target_id,
                reason=f"provider returned HTTP {status}",
            )
        if status == 400 and _error_code(response.body) in _CONTEXT_ERROR_CODES:
            raise ContextOverflowError(self.target_id)
        raise BackendRequestRejectedError(self.target_id, status)

    def _normalize_response(
        self,
        request: InferenceRequest,
        raw: HTTPJSONResponse,
        tool_names: Mapping[str, str],
    ) -> NormalizedModelResponse:
        body = _mapping(raw.body, self.target_id, "response body")
        choices = body.get("choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            raise MalformedModelOutputError(
                self.target_id,
                "response choices are missing",
            )
        choice = _mapping(choices[0], self.target_id, "first choice")
        message = _mapping(choice.get("message"), self.target_id, "message")
        finish_reason = _finish_reason(choice.get("finish_reason"))
        tool_calls = message.get("tool_calls")
        intents: tuple[ToolIntentDraft, ...] = ()
        output: Any = None
        kind = ModelResponseKind.OUTPUT
        if isinstance(tool_calls, (list, tuple)) and tool_calls:
            intents = tuple(
                _tool_intent(item, tool_names, self.target_id)
                for item in tool_calls
            )
            kind = ModelResponseKind.TOOL_INTENT
            finish_reason = NormalizedFinishReason.TOOL_INTENT
        else:
            content = message.get("content")
            if not isinstance(content, str):
                raise MalformedModelOutputError(
                    self.target_id,
                    "assistant content is missing",
                )
            if (
                request.requirements.required_structured_output
                is not StructuredOutputLevel.NONE
                or request.response_schema is not None
            ):
                try:
                    output = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise MalformedModelOutputError(
                        self.target_id,
                        "assistant content is not valid JSON",
                    ) from exc
            else:
                output = content
        usage = _usage(body.get("usage"), self.target_id)
        remote_id = body.get("id")
        return NormalizedModelResponse(
            request_id=request.request_id,
            target_id=self.target_id,
            model_id=self.profile.model_id,
            kind=kind,
            output=output,
            tool_intents=intents,
            usage=usage,
            finish_reason=finish_reason,
            remote_request_id=(remote_id if isinstance(remote_id, str) else None),
        )


def _response_name(capability_id: str) -> str:
    sanitized = _SAFE_RESPONSE_NAME.sub("_", capability_id)[:64]
    return sanitized or "runtime_response"


def _model_ids(body: Any) -> frozenset[str] | None:
    if not isinstance(body, Mapping):
        return None
    data = body.get("data")
    if not isinstance(data, (list, tuple)):
        return None
    model_ids: set[str] = set()
    for item in data:
        if not isinstance(item, Mapping):
            return None
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id:
            return None
        model_ids.add(model_id)
    return frozenset(model_ids)


def _probe_unavailable(
    target_id: str,
    diagnostic: str,
) -> BackendProbeResult:
    return BackendProbeResult(
        target_id=target_id,
        availability=BackendAvailability.UNAVAILABLE,
        protocol_version="openai-chat-completions",
        diagnostics=(diagnostic,),
    )


def _mapping(value: Any, target_id: str, subject: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MalformedModelOutputError(target_id, f"{subject} is not an object")
    return cast(Mapping[str, Any], value)


def _error_code(body: Any) -> str | None:
    if not isinstance(body, Mapping):
        return None
    error = body.get("error")
    if not isinstance(error, Mapping):
        return None
    code = error.get("code")
    return code if isinstance(code, str) else None


def _finish_reason(value: Any) -> NormalizedFinishReason:
    mapping = {
        "stop": NormalizedFinishReason.COMPLETED,
        "length": NormalizedFinishReason.OUTPUT_LIMIT,
        "tool_calls": NormalizedFinishReason.TOOL_INTENT,
        "function_call": NormalizedFinishReason.TOOL_INTENT,
        "content_filter": NormalizedFinishReason.CONTENT_FILTERED,
    }
    return mapping.get(value, NormalizedFinishReason.UNKNOWN)


def _tool_intent(
    value: Any,
    tool_names: Mapping[str, str],
    target_id: str,
) -> ToolIntentDraft:
    call = _mapping(value, target_id, "tool call")
    call_id = call.get("id")
    function = _mapping(call.get("function"), target_id, "tool function")
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(call_id, str) or not call_id:
        raise MalformedModelOutputError(target_id, "tool call ID is missing")
    if not isinstance(name, str) or name not in tool_names:
        raise MalformedModelOutputError(target_id, "tool name is not eligible")
    if not isinstance(arguments, str):
        raise MalformedModelOutputError(target_id, "tool arguments are missing")
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise MalformedModelOutputError(
            target_id,
            "tool arguments are not valid JSON",
        ) from exc
    if not isinstance(parsed, dict):
        raise MalformedModelOutputError(
            target_id,
            "tool arguments must be a JSON object",
        )
    return ToolIntentDraft(
        call_key=call_id,
        capability_id=tool_names[name],
        arguments=parsed,
    )


def _usage(value: Any, target_id: str) -> InferenceUsage:
    if value is None:
        return InferenceUsage()
    usage = _mapping(value, target_id, "usage")
    return InferenceUsage(
        input_tokens=_optional_nonnegative_int(
            usage.get("prompt_tokens"), target_id, "prompt_tokens"
        ),
        output_tokens=_optional_nonnegative_int(
            usage.get("completion_tokens"), target_id, "completion_tokens"
        ),
        total_tokens=_optional_nonnegative_int(
            usage.get("total_tokens"), target_id, "total_tokens"
        ),
    )


def _optional_nonnegative_int(
    value: Any,
    target_id: str,
    field: str,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MalformedModelOutputError(
            target_id,
            f"usage field '{field}' is invalid",
        )
    return int(value)

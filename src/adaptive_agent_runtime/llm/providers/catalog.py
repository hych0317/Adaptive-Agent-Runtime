"""Explicit connection presets for OpenAI-compatible model services."""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple

from pydantic import Field, SecretStr, model_validator

from adaptive_agent_runtime.llm.json_types import LLMModel
from adaptive_agent_runtime.llm.models import (
    BackendAuthentication,
    BackendKind,
    BackendLimits,
    BackendMetering,
    BackendTransportFeatures,
    InferenceTargetProfile,
    ReasoningEffort,
)
from adaptive_agent_runtime.llm.providers.http import AsyncJSONTransport
from adaptive_agent_runtime.llm.providers.openai_compatible import (
    ChatMaxTokensParameter,
    OpenAICompatibleChatBackend,
    OpenAICompatibleChatConfig,
    OpenAICompatibleProbeMode,
)


class OpenAICompatibleService(StrEnum):
    OPENAI = "openai"
    QWEN_CHINA = "qwen_china"
    QWEN_INTERNATIONAL = "qwen_international"
    DEEPSEEK = "deepseek"
    LOCAL = "local"


class _ServiceDefaults(NamedTuple):
    backend_id: str
    base_url: str | None
    api_key_env: str | None
    backend_kind: BackendKind
    requires_api_key: bool
    probe_mode: OpenAICompatibleProbeMode


_SERVICE_DEFAULTS = {
    OpenAICompatibleService.OPENAI: _ServiceDefaults(
        backend_id="openai-compatible.openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        backend_kind=BackendKind.API,
        requires_api_key=True,
        probe_mode=OpenAICompatibleProbeMode.MODELS_ENDPOINT,
    ),
    OpenAICompatibleService.QWEN_CHINA: _ServiceDefaults(
        backend_id="openai-compatible.qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        backend_kind=BackendKind.API,
        requires_api_key=True,
        probe_mode=OpenAICompatibleProbeMode.MODELS_ENDPOINT,
    ),
    OpenAICompatibleService.QWEN_INTERNATIONAL: _ServiceDefaults(
        backend_id="openai-compatible.qwen",
        base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        backend_kind=BackendKind.API,
        requires_api_key=True,
        probe_mode=OpenAICompatibleProbeMode.MODELS_ENDPOINT,
    ),
    OpenAICompatibleService.DEEPSEEK: _ServiceDefaults(
        backend_id="openai-compatible.deepseek",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        backend_kind=BackendKind.API,
        requires_api_key=True,
        probe_mode=OpenAICompatibleProbeMode.MODELS_ENDPOINT,
    ),
    OpenAICompatibleService.LOCAL: _ServiceDefaults(
        backend_id="openai-compatible.local",
        base_url=None,
        api_key_env=None,
        backend_kind=BackendKind.LOCAL,
        requires_api_key=False,
        probe_mode=OpenAICompatibleProbeMode.MODELS_ENDPOINT,
    ),
}


class OpenAICompatibleTargetDefinition(LLMModel):
    """One explicitly negotiated model target and its connection metadata."""

    service: OpenAICompatibleService
    target_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    features: BackendTransportFeatures
    supported_cognitive_capability_ids: tuple[str, ...] = Field(min_length=1)
    base_url: str | None = Field(default=None, min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1)
    api_key: SecretStr | None = None
    requires_api_key: bool | None = None
    limits: BackendLimits = Field(default_factory=BackendLimits)
    metering: BackendMetering = Field(default_factory=BackendMetering)
    tags: tuple[str, ...] = ()
    max_tokens_parameter: ChatMaxTokensParameter = (
        ChatMaxTokensParameter.MAX_TOKENS
    )
    strict_json_schema: bool = False
    allow_insecure_http: bool = False
    probe_mode: OpenAICompatibleProbeMode | None = None
    probe_timeout_seconds: float = Field(default=10.0, gt=0.0)
    reasoning_effort: ReasoningEffort | None = None

    @model_validator(mode="after")
    def validate_definition(self) -> OpenAICompatibleTargetDefinition:
        defaults = _SERVICE_DEFAULTS[self.service]
        base_url = self.base_url or defaults.base_url
        if base_url is None:
            raise ValueError("local compatible target requires a base URL")
        requires_key = (
            defaults.requires_api_key
            if self.requires_api_key is None
            else self.requires_api_key
        )
        key_env = self.api_key_env or defaults.api_key_env
        if requires_key and key_env is None and self.api_key is None:
            raise ValueError("authenticated compatible target requires credentials")
        if not requires_key and (
            self.api_key_env is not None or self.api_key is not None
        ):
            raise ValueError("credentials cannot be set when authentication is disabled")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("compatible target tags must be unique")
        if len(set(self.supported_cognitive_capability_ids)) != len(
            self.supported_cognitive_capability_ids
        ):
            raise ValueError("compatible target cognitive capabilities must be unique")
        supported_efforts = {
            OpenAICompatibleService.OPENAI: {
                ReasoningEffort.NONE,
                ReasoningEffort.MINIMAL,
                ReasoningEffort.LOW,
                ReasoningEffort.MEDIUM,
                ReasoningEffort.HIGH,
                ReasoningEffort.XHIGH,
            },
            OpenAICompatibleService.DEEPSEEK: {
                ReasoningEffort.HIGH,
                ReasoningEffort.MAX,
            },
        }.get(self.service, set())
        effort = self.reasoning_effort
        if (
            effort not in {None, ReasoningEffort.DEFAULT}
            and effort not in supported_efforts
        ):
            assert effort is not None
            raise ValueError(
                f"{self.service.value} does not support reasoning effort "
                f"'{effort.value}' through this adapter"
            )
        OpenAICompatibleChatConfig(
            base_url=base_url,
            api_key_env=key_env or "UNUSED_API_KEY",
            api_key=self.api_key,
            requires_api_key=requires_key,
            allow_insecure_http=self.allow_insecure_http,
            max_tokens_parameter=self.max_tokens_parameter,
            strict_json_schema=self.strict_json_schema,
            probe_mode=self.probe_mode or defaults.probe_mode,
            probe_timeout_seconds=self.probe_timeout_seconds,
            reasoning_effort=self.reasoning_effort,
        )
        return self

    def build_profile(self) -> InferenceTargetProfile:
        defaults = _SERVICE_DEFAULTS[self.service]
        requires_key = self._requires_api_key()
        return InferenceTargetProfile(
            target_id=self.target_id,
            backend_id=defaults.backend_id,
            backend_kind=defaults.backend_kind,
            adapter_version="1",
            model_id=self.model_id,
            features=self.features,
            supported_cognitive_capability_ids=(
                self.supported_cognitive_capability_ids
            ),
            limits=self.limits,
            authentication=BackendAuthentication(
                supported_methods=(
                    tuple(
                        method
                        for method, enabled in (
                            ("bearer_env", (self.api_key_env or defaults.api_key_env) is not None),
                            ("bearer_config", self.api_key is not None),
                        )
                        if enabled
                    )
                    if requires_key
                    else ()
                )
            ),
            metering=self.metering,
            tags=self.tags,
        )

    def build_config(self) -> OpenAICompatibleChatConfig:
        defaults = _SERVICE_DEFAULTS[self.service]
        base_url = self.base_url or defaults.base_url
        if base_url is None:
            raise AssertionError("validated target is missing a base URL")
        key_env = self.api_key_env or defaults.api_key_env or "UNUSED_API_KEY"
        return OpenAICompatibleChatConfig(
            base_url=base_url,
            api_key_env=key_env,
            api_key=self.api_key,
            requires_api_key=self._requires_api_key(),
            allow_insecure_http=self.allow_insecure_http,
            max_tokens_parameter=self.max_tokens_parameter,
            strict_json_schema=self.strict_json_schema,
            probe_mode=self.probe_mode or defaults.probe_mode,
            probe_timeout_seconds=self.probe_timeout_seconds,
            reasoning_effort=self.reasoning_effort,
        )

    def build_backend(
        self,
        transport: AsyncJSONTransport,
    ) -> OpenAICompatibleChatBackend:
        return OpenAICompatibleChatBackend(
            self.build_profile(),
            self.build_config(),
            transport,
        )

    def _requires_api_key(self) -> bool:
        default = _SERVICE_DEFAULTS[self.service].requires_api_key
        return default if self.requires_api_key is None else self.requires_api_key

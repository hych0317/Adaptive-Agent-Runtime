"""Provider-neutral inference and backend profile models."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypeAlias
from uuid import UUID, uuid4

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
from pydantic import AwareDatetime, Field, model_validator

from adaptive_agent_runtime.llm.json_types import (
    ImmutableJsonObject,
    ImmutableJsonValue,
    LLMModel,
    utc_now,
)


class BackendKind(StrEnum):
    API = "api"
    LOCAL = "local"
    CLI = "cli"


class BackendExecutionMode(StrEnum):
    INFERENCE = "inference"
    AGENT = "agent"


class BackendAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    AUTH_REQUIRED = "auth_required"


class StructuredOutputLevel(StrEnum):
    NONE = "none"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class ReasoningEffort(StrEnum):
    """Provider-neutral reasoning effort requested for one inference target."""

    DEFAULT = "default"
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class ToolIntentMode(StrEnum):
    DISABLED = "disabled"
    ALLOWED = "allowed"
    REQUIRED = "required"


class ModelResponseKind(StrEnum):
    OUTPUT = "output"
    TOOL_INTENT = "tool_intent"


class NormalizedFinishReason(StrEnum):
    COMPLETED = "completed"
    OUTPUT_LIMIT = "output_limit"
    TOOL_INTENT = "tool_intent"
    CONTENT_FILTERED = "content_filtered"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class MeteringKind(StrEnum):
    METERED = "metered"
    SUBSCRIPTION = "subscription"
    UNKNOWN = "unknown"


class BackendTransportFeatures(LLMModel):
    structured_output: StructuredOutputLevel = StructuredOutputLevel.NONE
    tool_intent: bool = False
    multimodal: bool = False


class BackendLimits(LLMModel):
    max_context_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    max_concurrency: int | None = Field(default=None, ge=1)
    default_timeout_seconds: float = Field(default=60.0, gt=0.0)


class BackendDelegatedAccess(LLMModel):
    """Operations the model or agent may invoke beyond model transport itself."""

    filesystem_read: bool = False
    filesystem_write: bool = False
    shell_execution: bool = False
    mcp_execution: bool = False
    arbitrary_network: bool = False
    session_persistence: bool = False


class BackendAuthentication(LLMModel):
    supported_methods: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_methods(self) -> BackendAuthentication:
        if len(set(self.supported_methods)) != len(self.supported_methods):
            raise ValueError("backend authentication methods must be unique")
        if any(not method for method in self.supported_methods):
            raise ValueError("backend authentication methods cannot be empty")
        return self


class BackendMetering(LLMModel):
    kind: MeteringKind = MeteringKind.UNKNOWN
    reports_token_usage: bool = False
    reports_monetary_cost: bool = False


class _BackendTargetProfile(LLMModel):
    """Identity and limits shared by inference and autonomous targets."""

    target_id: str = Field(min_length=1)
    backend_id: str = Field(min_length=1)
    backend_kind: BackendKind
    adapter_version: str = Field(min_length=1)
    model_id: str | None = Field(default=None, min_length=1)
    limits: BackendLimits = Field(default_factory=BackendLimits)
    authentication: BackendAuthentication = Field(
        default_factory=BackendAuthentication
    )
    metering: BackendMetering = Field(default_factory=BackendMetering)
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_profile(self) -> _BackendTargetProfile:
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("backend target tags must be unique")
        return self


class InferenceTargetProfile(_BackendTargetProfile):
    """A pure inference target with no delegated execution authority."""

    execution_mode: Literal[BackendExecutionMode.INFERENCE] = (
        BackendExecutionMode.INFERENCE
    )
    features: BackendTransportFeatures = Field(
        default_factory=BackendTransportFeatures
    )
    supported_cognitive_capability_ids: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def validate_inference_profile(self) -> InferenceTargetProfile:
        capability_ids = self.supported_cognitive_capability_ids
        if capability_ids is not None:
            if not capability_ids:
                raise ValueError(
                    "negotiated cognitive capabilities cannot be empty"
                )
            if len(set(capability_ids)) != len(capability_ids):
                raise ValueError(
                    "negotiated cognitive capabilities must be unique"
                )
            if any(not capability_id for capability_id in capability_ids):
                raise ValueError(
                    "negotiated cognitive capability identifiers cannot be empty"
                )
        return self


class AgentTargetProfile(_BackendTargetProfile):
    """An autonomous target whose delegated access is explicitly bounded."""

    execution_mode: Literal[BackendExecutionMode.AGENT] = BackendExecutionMode.AGENT
    delegated_access: BackendDelegatedAccess = Field(
        default_factory=BackendDelegatedAccess
    )


BackendTargetProfile: TypeAlias = InferenceTargetProfile | AgentTargetProfile


class BackendProbeResult(LLMModel):
    target_id: str = Field(min_length=1)
    availability: BackendAvailability
    runtime_version: str | None = Field(default=None, min_length=1)
    active_auth_method: str | None = Field(default=None, min_length=1)
    protocol_version: str | None = Field(default=None, min_length=1)
    available_model_ids: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    checked_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_probe(self) -> BackendProbeResult:
        if len(set(self.diagnostics)) != len(self.diagnostics):
            raise ValueError("backend probe diagnostics must be unique")
        if len(set(self.available_model_ids)) != len(self.available_model_ids):
            raise ValueError("backend probe model identifiers must be unique")
        if any(not model_id for model_id in self.available_model_ids):
            raise ValueError("backend probe model identifiers cannot be empty")
        if (
            self.availability is BackendAvailability.AUTH_REQUIRED
            and self.active_auth_method is not None
        ):
            raise ValueError(
                "an authentication-required backend cannot have active auth"
            )
        return self


class InferenceRequirements(LLMModel):
    required_structured_output: StructuredOutputLevel = StructuredOutputLevel.NONE
    tool_intent: ToolIntentMode = ToolIntentMode.DISABLED
    max_output_tokens: int | None = Field(default=None, ge=1)


class InferenceCorrelation(LLMModel):
    run_id: UUID | None = None
    task_id: UUID | None = None
    node_id: UUID | None = None
    action_id: UUID | None = None


class ToolSpecification(LLMModel):
    capability_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    input_schema: ImmutableJsonObject = Field(default_factory=dict)


class InferenceRequest(LLMModel):
    request_id: UUID = Field(default_factory=uuid4)
    cognitive_capability_id: str = Field(min_length=1)
    required_target_id: str | None = Field(default=None, min_length=1)
    input: ImmutableJsonValue
    response_schema: ImmutableJsonObject | None = None
    requirements: InferenceRequirements = Field(
        default_factory=InferenceRequirements
    )
    eligible_tools: tuple[ToolSpecification, ...] = ()
    correlation: InferenceCorrelation = Field(
        default_factory=InferenceCorrelation
    )
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    trace_attributes: ImmutableJsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_request(self) -> InferenceRequest:
        tool_ids = tuple(item.capability_id for item in self.eligible_tools)
        if len(set(tool_ids)) != len(tool_ids):
            raise ValueError("eligible tool capabilities must be unique")
        if (
            self.requirements.tool_intent is ToolIntentMode.DISABLED
            and self.eligible_tools
        ):
            raise ValueError("disabled tool intent cannot expose eligible tools")
        if (
            self.requirements.tool_intent is ToolIntentMode.REQUIRED
            and not self.eligible_tools
        ):
            raise ValueError("required tool intent needs at least one eligible tool")
        if (
            self.requirements.required_structured_output
            is StructuredOutputLevel.JSON_SCHEMA
            and self.response_schema is None
        ):
            raise ValueError("JSON Schema output requires a response_schema")
        if self.response_schema is not None:
            schema = self.model_dump(mode="json")["response_schema"]
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as exc:
                raise ValueError(
                    f"invalid response schema: {exc.message}"
                ) from exc
        return self


class InferenceUsage(LLMModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    monetary_cost: float | None = Field(default=None, ge=0.0)
    currency: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_cost(self) -> InferenceUsage:
        if (self.monetary_cost is None) is not (self.currency is None):
            raise ValueError("monetary cost and currency must appear together")
        if (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError("total tokens must equal input plus output tokens")
        return self


class ToolIntentDraft(LLMModel):
    call_key: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    arguments: ImmutableJsonObject = Field(default_factory=dict)


class NormalizedModelResponse(LLMModel):
    request_id: UUID
    target_id: str = Field(min_length=1)
    model_id: str | None = Field(default=None, min_length=1)
    kind: ModelResponseKind
    output: ImmutableJsonValue = None
    tool_intents: tuple[ToolIntentDraft, ...] = ()
    usage: InferenceUsage = Field(default_factory=InferenceUsage)
    finish_reason: NormalizedFinishReason | None = None
    remote_request_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_response(self) -> NormalizedModelResponse:
        if self.kind is ModelResponseKind.OUTPUT:
            if "output" not in self.model_fields_set:
                raise ValueError("an output response requires an explicit output")
            if self.tool_intents:
                raise ValueError("an output response cannot contain tool intents")
        else:
            if self.output is not None:
                raise ValueError("a tool-intent response cannot contain output")
            if not self.tool_intents:
                raise ValueError("a tool-intent response requires tool intents")
        call_keys = tuple(item.call_key for item in self.tool_intents)
        if len(set(call_keys)) != len(call_keys):
            raise ValueError("tool intent call keys must be unique")
        return self

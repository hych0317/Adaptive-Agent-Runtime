"""Gateway-backed implementations of the six cognitive capability contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar, cast

from pydantic import Field, ValidationError, model_validator

from adaptive_agent_runtime.llm.capabilities.models import (
    ActionProposalDraft,
    ActionProposalRequest,
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CapabilityTurnResult,
    CompressedContextDraft,
    CompressionRequest,
    GeneratedArtifactDraft,
    GenerationRequest,
    GraphMutationProposalDraft,
    GraphMutationProposalRequest,
    JudgeAssessmentDraft,
    JudgeRequest,
    MemoryCandidateBatchDraft,
    MemoryCandidateDraft,
    MemoryExtractionRequest,
    ReasoningContext,
    ReasoningResult,
    RecoveryDraft,
    RecoveryProposalRequest,
    RootCauseAnalysisRequest,
    RootCauseDraft,
    TaskGraphDraft,
    TaskPlanningRequest,
    ToolSelectionDraft,
    ToolSelectionProposalRequest,
)
from adaptive_agent_runtime.llm.capabilities.validation import (
    CapabilityDraftValidator,
)
from adaptive_agent_runtime.llm.errors import (
    CapabilityExecutionPolicyError,
    ResponseSchemaValidationError,
)
from adaptive_agent_runtime.llm.gateway.contracts import InferenceGateway
from adaptive_agent_runtime.llm.gateway.models import InferenceGatewayPolicy
from adaptive_agent_runtime.llm.json_types import (
    ImmutableJsonObject,
    LLMModel,
)
from adaptive_agent_runtime.llm.models import (
    InferenceCorrelation,
    InferenceRequest,
    InferenceRequirements,
    ModelResponseKind,
    StructuredOutputLevel,
    ToolIntentMode,
    ToolSpecification,
)


CapabilityResultModelT = TypeVar("CapabilityResultModelT", bound=LLMModel)


class CapabilityInferenceSettings(LLMModel):
    gateway_policy: InferenceGatewayPolicy = Field(
        default_factory=InferenceGatewayPolicy
    )
    tool_intent: ToolIntentMode = ToolIntentMode.DISABLED
    structured_output: StructuredOutputLevel = StructuredOutputLevel.JSON_SCHEMA
    eligible_tools: tuple[ToolSpecification, ...] = ()
    max_output_tokens: int | None = Field(default=None, ge=1)
    trace_attributes: ImmutableJsonObject = Field(default_factory=dict)
    required_target_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_settings(self) -> CapabilityInferenceSettings:
        tool_ids = tuple(tool.capability_id for tool in self.eligible_tools)
        if len(set(tool_ids)) != len(tool_ids):
            raise ValueError("capability inference tools must be unique")
        if self.tool_intent is ToolIntentMode.DISABLED and self.eligible_tools:
            raise ValueError("disabled tool intent cannot configure tools")
        if self.tool_intent is ToolIntentMode.REQUIRED and not self.eligible_tools:
            raise ValueError("required tool intent needs configured tools")
        if self.structured_output is StructuredOutputLevel.NONE:
            raise ValueError(
                "gateway-backed capabilities require structured output"
            )
        return self


class _GatewayCapability:
    capability_id: str

    def __init__(
        self,
        *,
        gateway: InferenceGateway,
        draft_validator: CapabilityDraftValidator,
        settings: CapabilityInferenceSettings | None = None,
    ) -> None:
        self._gateway = gateway
        self._draft_validator = draft_validator
        self._settings = settings or CapabilityInferenceSettings()

    async def _execute_model(
        self,
        payload: LLMModel,
        result_type: type[CapabilityResultModelT],
        validate: Callable[[CapabilityResultModelT], CapabilityResultModelT],
        *,
        eligible_tools: tuple[ToolSpecification, ...] | None = None,
        invocation: CapabilityInvocationMetadata | None = None,
        response_schema: ImmutableJsonObject | None = None,
    ) -> CapabilityTurnResult[CapabilityResultModelT]:
        tools = (
            self._settings.eligible_tools
            if eligible_tools is None
            else eligible_tools
        )
        tool_mode = self._settings.tool_intent
        if tool_mode is ToolIntentMode.REQUIRED and not tools:
            raise CapabilityExecutionPolicyError(
                self.capability_id,
                "required tool intent has no Runtime-eligible tools",
            )
        if tool_mode is ToolIntentMode.ALLOWED and not tools:
            tool_mode = ToolIntentMode.DISABLED
        trace_attributes = dict(self._settings.trace_attributes)
        if invocation is not None:
            for key, value in invocation.trace_attributes.items():
                configured = trace_attributes.get(key)
                if key in trace_attributes and configured != value:
                    raise CapabilityExecutionPolicyError(
                        self.capability_id,
                        f"invocation trace attribute '{key}' conflicts "
                        "with deployment settings",
                    )
                trace_attributes[key] = value
        request = InferenceRequest(
            cognitive_capability_id=self.capability_id,
            required_target_id=self._settings.required_target_id,
            input={
                "contract_version": "1",
                "payload": payload.model_dump(mode="json"),
            },
            response_schema=(
                response_schema
                if response_schema is not None
                else result_type.model_json_schema(mode="validation")
            ),
            requirements=InferenceRequirements(
                required_structured_output=self._settings.structured_output,
                tool_intent=tool_mode,
                max_output_tokens=self._settings.max_output_tokens,
            ),
            eligible_tools=tools,
            correlation=(
                invocation.correlation
                if invocation is not None
                else InferenceCorrelation()
            ),
            trace_attributes=trace_attributes,
        )
        response = await self._gateway.execute(
            request,
            self._settings.gateway_policy,
        )
        if response.kind is ModelResponseKind.TOOL_INTENT:
            return CapabilityTurnResult[CapabilityResultModelT](
                kind=CapabilityTurnKind.TOOL_INTENT,
                tool_intents=response.tool_intents,
            )
        try:
            result = result_type.model_validate(response.output)
        except ValidationError as exc:
            raise ResponseSchemaValidationError(
                response.target_id,
                _validation_summary(exc),
            ) from exc
        return CapabilityTurnResult[CapabilityResultModelT](
            kind=CapabilityTurnKind.COMPLETED,
            result=validate(result),
        )


class GatewayReasoningCapability(_GatewayCapability):
    module_id = "llm.capability.reasoning.gateway"
    capability_id = "reasoning"

    async def analyze(
        self,
        context: ReasoningContext,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[ReasoningResult]:
        return await self._execute_model(
            context,
            ReasoningResult,
            lambda result: self._draft_validator.validate_reasoning(
                context,
                result,
            ),
            invocation=invocation,
        )


class GatewayTaskGraphProposalCapability(_GatewayCapability):
    module_id = "llm.capability.task_graph_proposal.gateway"
    capability_id = "task_graph_proposal"

    async def propose(
        self,
        request: TaskPlanningRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[TaskGraphDraft]:
        allowed = set(request.available_execution_capability_ids)
        tools = tuple(
            tool
            for tool in self._settings.eligible_tools
            if tool.capability_id in allowed
        )
        return await self._execute_model(
            request,
            TaskGraphDraft,
            lambda result: self._draft_validator.validate_task_graph(
                request,
                result,
            ),
            eligible_tools=tools,
            invocation=invocation,
        )


class GatewayRecoveryProposalCapability(_GatewayCapability):
    module_id = "llm.capability.recovery_proposal.gateway"
    capability_id = "recovery_proposal"

    async def propose_recovery(
        self,
        request: RecoveryProposalRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[RecoveryDraft]:
        return await self._execute_model(
            request,
            RecoveryDraft,
            lambda result: self._draft_validator.validate_recovery_proposal(
                request,
                result,
            ),
            eligible_tools=(),
            invocation=invocation,
        )


class GatewayRootCauseAnalysisCapability(_GatewayCapability):
    module_id = "llm.capability.root_cause_analysis.gateway"
    capability_id = "root_cause_analysis"

    async def analyze_root_cause(
        self,
        request: RootCauseAnalysisRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[RootCauseDraft]:
        return await self._execute_model(
            request,
            RootCauseDraft,
            lambda result: self._draft_validator.validate_root_cause(
                request,
                result,
            ),
            eligible_tools=(),
            invocation=invocation,
        )


class GatewayToolSelectionProposalCapability(_GatewayCapability):
    module_id = "llm.capability.tool_selection_proposal.gateway"
    capability_id = "tool_selection_proposal"

    async def propose_tool_selection(
        self,
        request: ToolSelectionProposalRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[ToolSelectionDraft]:
        return await self._execute_model(
            request,
            ToolSelectionDraft,
            lambda result: self._draft_validator.validate_tool_selection(
                request,
                result,
            ),
            eligible_tools=(),
            invocation=invocation,
        )


class GatewayActionProposalCapability(_GatewayCapability):
    module_id = "llm.capability.action_proposal.gateway"
    capability_id = "action_proposal"

    async def propose_action(
        self,
        request: ActionProposalRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[ActionProposalDraft]:
        return await self._execute_model(
            request,
            ActionProposalDraft,
            lambda result: self._draft_validator.validate_action_proposal(
                request,
                result,
            ),
            eligible_tools=(),
            invocation=invocation,
        )


class GatewayGraphMutationProposalCapability(_GatewayCapability):
    module_id = "llm.capability.graph_mutation_proposal.gateway"
    capability_id = "graph_mutation_proposal"

    async def propose_mutations(
        self,
        request: GraphMutationProposalRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[GraphMutationProposalDraft]:
        allowed = set(request.available_execution_capability_ids)
        tools = tuple(
            tool
            for tool in self._settings.eligible_tools
            if tool.capability_id in allowed
        )
        return await self._execute_model(
            request,
            GraphMutationProposalDraft,
            lambda result: (
                self._draft_validator.validate_graph_mutation_proposal(
                    request,
                    result,
                )
            ),
            eligible_tools=tools,
            invocation=invocation,
        )


class GatewayArtifactGenerationCapability(_GatewayCapability):
    module_id = "llm.capability.artifact_generation.gateway"
    capability_id = "artifact_generation"

    async def generate(
        self,
        request: GenerationRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[GeneratedArtifactDraft]:
        return await self._execute_model(
            request,
            GeneratedArtifactDraft,
            lambda result: self._draft_validator.validate_generation(
                request,
                result,
            ),
            invocation=invocation,
            response_schema=_generation_response_schema(request),
        )


class GatewayEvidenceJudgeCapability(_GatewayCapability):
    module_id = "llm.capability.evidence_judge.gateway"
    capability_id = "evidence_judge"

    async def assess(
        self,
        request: JudgeRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[JudgeAssessmentDraft]:
        return await self._execute_model(
            request,
            JudgeAssessmentDraft,
            lambda result: self._draft_validator.validate_judge(
                request,
                result,
            ),
            invocation=invocation,
        )


class GatewaySemanticCompressionCapability(_GatewayCapability):
    module_id = "llm.capability.semantic_compression.gateway"
    capability_id = "semantic_compression"

    async def compress(
        self,
        request: CompressionRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[CompressedContextDraft]:
        return await self._execute_model(
            request,
            CompressedContextDraft,
            lambda result: self._draft_validator.validate_compression(
                request,
                result,
            ),
            invocation=invocation,
        )


class GatewayMemoryExtractionCapability(_GatewayCapability):
    module_id = "llm.capability.memory_extraction.gateway"
    capability_id = "memory_extraction"

    async def extract(
        self,
        request: MemoryExtractionRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[tuple[MemoryCandidateDraft, ...]]:
        turn = await self._execute_model(
            request,
            MemoryCandidateBatchDraft,
            lambda result: self._validate_batch(request, result),
            invocation=invocation,
        )
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            return CapabilityTurnResult[tuple[MemoryCandidateDraft, ...]](
                kind=CapabilityTurnKind.TOOL_INTENT,
                tool_intents=turn.tool_intents,
            )
        if turn.result is None:
            raise RuntimeError("completed memory extraction omitted its result")
        return CapabilityTurnResult[tuple[MemoryCandidateDraft, ...]](
            kind=CapabilityTurnKind.COMPLETED,
            result=turn.result.candidates,
        )

    def _validate_batch(
        self,
        request: MemoryExtractionRequest,
        result: MemoryCandidateBatchDraft,
    ) -> MemoryCandidateBatchDraft:
        self._draft_validator.validate_memory_extraction(
            request,
            result.candidates,
        )
        return result


def _validation_summary(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors(include_input=False)
    )


def _generation_response_schema(
    request: GenerationRequest,
) -> ImmutableJsonObject:
    """Specialize the artifact wrapper around the requested content schema.

    ``GeneratedArtifactDraft.content`` is provider-neutral arbitrary JSON, so
    its generic Pydantic schema contains an empty ``JsonValue`` definition.
    Strict structured-output providers reject that unconstrained definition.
    Generation requests already carry the Runtime-approved artifact schema;
    embedding it here makes the model output contract both strict and exact.
    """

    schema = _mutable_json(
        GeneratedArtifactDraft.model_json_schema(mode="validation")
    )
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise CapabilityExecutionPolicyError(
            "artifact_generation",
            "generated artifact schema has no properties",
        )
    properties["media_type"] = {
        "type": "string",
        "const": request.media_type,
    }
    content_schema: Any = None
    if request.output_schema is not None:
        content_schema = _mutable_json(request.output_schema)
    elif request.media_type.startswith("text/"):
        content_schema = {"type": "string"}
    if content_schema is None:
        return cast(ImmutableJsonObject, schema)

    parent_definitions = dict(schema.get("$defs", {}))
    parent_definitions.pop("JsonValue", None)
    child_definitions = dict(content_schema.pop("$defs", {}))
    definition_names: dict[str, str] = {}
    for name in child_definitions:
        candidate = name
        suffix = 1
        while candidate in parent_definitions:
            candidate = f"content_{name}_{suffix}"
            suffix += 1
        definition_names[name] = candidate

    def rewrite_references(value: Any) -> Any:
        if isinstance(value, Mapping):
            rewritten: dict[str, Any] = {}
            for key, item in value.items():
                if key == "$ref" and isinstance(item, str):
                    prefix = "#/$defs/"
                    if item.startswith(prefix):
                        original = item[len(prefix) :]
                        item = prefix + definition_names.get(original, original)
                rewritten[key] = rewrite_references(item)
            return rewritten
        if isinstance(value, (list, tuple)):
            return [rewrite_references(item) for item in value]
        return value

    content_schema = rewrite_references(content_schema)
    for name, definition in child_definitions.items():
        parent_definitions[definition_names[name]] = rewrite_references(
            definition
        )
    if parent_definitions:
        schema["$defs"] = parent_definitions
    else:
        schema.pop("$defs", None)
    properties["content"] = content_schema
    return cast(
        ImmutableJsonObject,
        _strict_response_schema(schema),
    )


def _strict_response_schema(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {
            key: _strict_response_schema(item)
            for key, item in value.items()
            if key != "default"
        }
        properties = result.get("properties")
        if result.get("type") == "object" and isinstance(properties, dict):
            result["additionalProperties"] = False
            result["required"] = list(properties)
        return result
    if isinstance(value, (list, tuple)):
        return [_strict_response_schema(item) for item in value]
    return value


def _mutable_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_json(item) for item in value]
    return value

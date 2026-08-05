"""Explicit opt-in LLM deployment composition for the Research Application."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from adaptive_agent_runtime.llm import (
    AnthropicAPITargetDefinition,
    AsyncJSONTransport,
    AsyncProcessTransport,
    CapabilityContextPolicy,
    CapabilityInferenceSettings,
    BackendProbeResult,
    ContextEgressPolicy,
    ContextSensitivity,
    ClaudeCodeInferenceTargetDefinition,
    CodexCLIInferenceTargetDefinition,
    HttpxJSONTransport,
    InferenceBackend,
    InferenceExecutionBudget,
    InferenceGatewayPolicy,
    InferenceRoutingPolicy,
    ManagedInferenceComposition,
    OpenAICompatibleTargetDefinition,
    PolicyEnforcedContextAdapter,
    StructuredOutputLevel,
    SubprocessTransport,
    ToolIntentMode,
    compose_managed_capabilities,
    compose_managed_inference,
)

from applications.research_agent.cognition import (
    ResearchCognitiveCapabilities,
    ResearchContextProjection,
    ResearchReportContextProjection,
)
from applications.research_agent.llm_tools import (
    RESEARCH_INFORMATION_RETRIEVAL_TOOL,
)


class ResearchLLMCapability(StrEnum):
    REASONING = "reasoning"
    PLANNING = "planning"
    GENERATION = "generation"
    JUDGE = "judge"
    ROOT_CAUSE = "root_cause"
    TOOL_SELECTION = "tool_selection"
    COMPRESSION = "compression"
    EXTRACTION = "extraction"


_MANAGED_CAPABILITY_IDS = {
    ResearchLLMCapability.REASONING: ("reasoning",),
    ResearchLLMCapability.PLANNING: (
        "task_graph_proposal",
        "action_proposal",
        "graph_mutation_proposal",
        "recovery_proposal",
    ),
    ResearchLLMCapability.GENERATION: ("artifact_generation",),
    ResearchLLMCapability.JUDGE: ("evidence_judge",),
    ResearchLLMCapability.ROOT_CAUSE: ("root_cause_analysis",),
    ResearchLLMCapability.TOOL_SELECTION: ("tool_selection_proposal",),
    ResearchLLMCapability.COMPRESSION: ("semantic_compression",),
    ResearchLLMCapability.EXTRACTION: ("memory_extraction",),
}


def managed_capability_ids(
    capabilities: tuple[ResearchLLMCapability, ...],
) -> tuple[str, ...]:
    return tuple(
        capability_id
        for item in capabilities
        for capability_id in _MANAGED_CAPABILITY_IDS[item]
    )


ResearchInferenceTargetDefinition: TypeAlias = (
    AnthropicAPITargetDefinition
    | OpenAICompatibleTargetDefinition
    | CodexCLIInferenceTargetDefinition
    | ClaudeCodeInferenceTargetDefinition
)


@dataclass(frozen=True)
class ResearchLLMDeploymentConfig:
    """Deployment choices; API keys remain inside provider configuration."""

    target: ResearchInferenceTargetDefinition
    enabled_capabilities: tuple[ResearchLLMCapability, ...] = (
        ResearchLLMCapability.GENERATION,
    )
    allowed_context_sensitivities: tuple[ContextSensitivity, ...] = (
        ContextSensitivity.INTERNAL,
    )
    context_redact_keys: tuple[str, ...] = (
        "api_key",
        "authorization",
        "password",
        "secret",
        "token",
    )
    max_context_tokens: int = 4096
    max_output_tokens: int | None = None
    max_attempts: int = 1
    max_elapsed_seconds: float | None = None
    reasoner_tool_intent_limit: int = 0

    def __post_init__(self) -> None:
        _require_unique(self.enabled_capabilities, "enabled capabilities")
        _require_unique(
            self.allowed_context_sensitivities,
            "allowed context sensitivities",
        )
        _require_unique(self.context_redact_keys, "Context redaction keys")
        if not self.enabled_capabilities:
            raise ValueError("LLM deployment requires an enabled capability")
        if not self.allowed_context_sensitivities:
            raise ValueError("LLM deployment requires an allowed sensitivity")
        if self.max_context_tokens < 1:
            raise ValueError("Context token budget must be positive")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("output token budget must be positive")
        if self.max_attempts < 1:
            raise ValueError("inference attempt budget must be positive")
        if (
            self.max_elapsed_seconds is not None
            and self.max_elapsed_seconds <= 0
        ):
            raise ValueError("inference elapsed-time budget must be positive")
        if self.reasoner_tool_intent_limit < 0:
            raise ValueError("Reasoner ToolIntent limit cannot be negative")
        if (
            self.reasoner_tool_intent_limit
            and ResearchLLMCapability.REASONING not in self.enabled_capabilities
        ):
            raise ValueError("Reasoner ToolIntent limit requires reasoning")


@dataclass(frozen=True)
class ResearchLLMDeployment:
    cognitive_capabilities: ResearchCognitiveCapabilities
    inference: ManagedInferenceComposition
    target_id: str
    model_id: str

    async def probe(self) -> BackendProbeResult:
        """Probe the selected target without starting an Agent run."""

        return await self.inference.gateway.probe_target(
            self.target_id,
            force=True,
        )


def build_research_llm_deployment(
    config: ResearchLLMDeploymentConfig,
    *,
    transport: AsyncJSONTransport | AsyncProcessTransport | None = None,
) -> ResearchLLMDeployment:
    """Build one target-bound gateway and selected Application capabilities."""

    backend: InferenceBackend
    if isinstance(
        config.target,
        (AnthropicAPITargetDefinition, OpenAICompatibleTargetDefinition),
    ):
        if transport is not None and not isinstance(transport, AsyncJSONTransport):
            raise TypeError("API inference target requires an HTTP transport")
        http_transport = transport or HttpxJSONTransport()
        backend = config.target.build_backend(http_transport)
    elif isinstance(
        config.target,
        (CodexCLIInferenceTargetDefinition, ClaudeCodeInferenceTargetDefinition),
    ):
        if transport is not None and not isinstance(
            transport,
            AsyncProcessTransport,
        ):
            raise TypeError("CLI inference target requires a process transport")
        if config.max_output_tokens is not None:
            raise ValueError(
                "CLI inference does not negotiate output-token limits"
            )
        process_transport = transport or SubprocessTransport()
        backend = config.target.build_backend(process_transport)
    else:
        raise TypeError("unsupported Research inference target definition")
    profile = backend.profile
    requested_capability_ids = set(
        managed_capability_ids(config.enabled_capabilities)
    )
    supported_capability_ids = profile.supported_cognitive_capability_ids
    if supported_capability_ids is None:
        raise ValueError("deployment target lacks cognitive capability negotiation")
    unsupported = requested_capability_ids.difference(supported_capability_ids)
    if unsupported:
        raise ValueError(
            "deployment target does not support cognitive capabilities: "
            + ", ".join(sorted(unsupported))
        )
    structured_output = profile.features.structured_output
    if structured_output is StructuredOutputLevel.NONE:
        raise ValueError(
            "Research cognitive capabilities require structured model output"
        )
    if config.reasoner_tool_intent_limit and not profile.features.tool_intent:
        raise ValueError("selected inference target does not support ToolIntent")
    provider_limit = profile.limits.max_output_tokens
    if (
        provider_limit is not None
        and config.max_output_tokens is not None
        and config.max_output_tokens > provider_limit
    ):
        raise ValueError("output token budget exceeds target profile limit")
    inference = compose_managed_inference((backend,))
    gateway_policy = InferenceGatewayPolicy(
        routing=InferenceRoutingPolicy(
            allowed_target_ids=(profile.target_id,),
            preferred_target_ids=(profile.target_id,),
            allowed_backend_kinds=(profile.backend_kind,),
        ),
        budget=InferenceExecutionBudget(
            max_attempts=config.max_attempts,
            max_elapsed_seconds=config.max_elapsed_seconds,
        ),
    )
    setting = CapabilityInferenceSettings(
        gateway_policy=gateway_policy,
        structured_output=structured_output,
        max_output_tokens=config.max_output_tokens,
        required_target_id=profile.target_id,
        trace_attributes={"application": "research_agent"},
    )
    capability_settings = {
        capability_id: setting
        for capability_id in (
            "reasoning",
            "task_graph_proposal",
            "action_proposal",
            "graph_mutation_proposal",
            "artifact_generation",
            "evidence_judge",
            "root_cause_analysis",
            "recovery_proposal",
            "semantic_compression",
            "memory_extraction",
            "tool_selection_proposal",
        )
    }
    if config.reasoner_tool_intent_limit:
        capability_settings["reasoning"] = CapabilityInferenceSettings(
            gateway_policy=gateway_policy,
            structured_output=structured_output,
            tool_intent=ToolIntentMode.ALLOWED,
            eligible_tools=(RESEARCH_INFORMATION_RETRIEVAL_TOOL,),
            max_output_tokens=config.max_output_tokens,
            required_target_id=profile.target_id,
            trace_attributes={"application": "research_agent"},
        )
    managed = compose_managed_capabilities(
        inference.gateway,
        settings=capability_settings,
    )
    enabled = set(config.enabled_capabilities)
    generator = (
        managed.generator
        if ResearchLLMCapability.GENERATION in enabled
        else None
    )
    report_context = None
    if generator is not None:
        report_context = ResearchReportContextProjection(
            adapter=PolicyEnforcedContextAdapter(),
            target=profile,
            capability_policy=CapabilityContextPolicy(
                cognitive_capability_id=generator.capability_id,
                max_context_tokens=config.max_context_tokens,
            ),
            egress_policy=ContextEgressPolicy(
                allowed_target_ids=(profile.target_id,),
                allowed_sensitivities=config.allowed_context_sensitivities,
                redact_keys=config.context_redact_keys,
                max_context_tokens=config.max_context_tokens,
            ),
        )
    compression_context = None
    if ResearchLLMCapability.COMPRESSION in enabled:
        compression_context = ResearchContextProjection(
            adapter=PolicyEnforcedContextAdapter(),
            target=profile,
            capability_policy=CapabilityContextPolicy(
                cognitive_capability_id=managed.compressor.capability_id,
                max_context_tokens=config.max_context_tokens,
            ),
            egress_policy=ContextEgressPolicy(
                allowed_target_ids=(profile.target_id,),
                allowed_sensitivities=config.allowed_context_sensitivities,
                redact_keys=config.context_redact_keys,
                max_context_tokens=config.max_context_tokens,
            ),
        )
    extraction_context = None
    if ResearchLLMCapability.EXTRACTION in enabled:
        extraction_context = ResearchContextProjection(
            adapter=PolicyEnforcedContextAdapter(),
            target=profile,
            capability_policy=CapabilityContextPolicy(
                cognitive_capability_id=managed.extractor.capability_id,
                max_context_tokens=config.max_context_tokens,
            ),
            egress_policy=ContextEgressPolicy(
                allowed_target_ids=(profile.target_id,),
                allowed_sensitivities=config.allowed_context_sensitivities,
                redact_keys=config.context_redact_keys,
                max_context_tokens=config.max_context_tokens,
            ),
        )
    mutation_context = None
    if ResearchLLMCapability.PLANNING in enabled:
        mutation_context = ResearchContextProjection(
            adapter=PolicyEnforcedContextAdapter(),
            target=profile,
            capability_policy=CapabilityContextPolicy(
                cognitive_capability_id=managed.mutation_planner.capability_id,
                max_context_tokens=config.max_context_tokens,
            ),
            egress_policy=ContextEgressPolicy(
                allowed_target_ids=(profile.target_id,),
                allowed_sensitivities=config.allowed_context_sensitivities,
                redact_keys=config.context_redact_keys,
                max_context_tokens=config.max_context_tokens,
            ),
        )
    capabilities = ResearchCognitiveCapabilities(
        report_generator=generator,
        evaluation_judge=(
            managed.judge if ResearchLLMCapability.JUDGE in enabled else None
        ),
        task_planner=(
            managed.planner
            if ResearchLLMCapability.PLANNING in enabled
            else None
        ),
        action_planner=(
            managed.action_planner
            if ResearchLLMCapability.PLANNING in enabled
            else None
        ),
        mutation_planner=(
            managed.mutation_planner
            if ResearchLLMCapability.PLANNING in enabled
            else None
        ),
        recovery_planner=(
            managed.recovery_planner
            if ResearchLLMCapability.PLANNING in enabled
            else None
        ),
        root_cause_analyzer=(
            managed.root_cause_analyzer
            if ResearchLLMCapability.ROOT_CAUSE in enabled
            else None
        ),
        tool_selector=(
            managed.tool_selector
            if ResearchLLMCapability.TOOL_SELECTION in enabled
            else None
        ),
        reasoner=(
            managed.reasoner
            if ResearchLLMCapability.REASONING in enabled
            else None
        ),
        context_compressor=(
            managed.compressor
            if ResearchLLMCapability.COMPRESSION in enabled
            else None
        ),
        memory_extractor=(
            managed.extractor
            if ResearchLLMCapability.EXTRACTION in enabled
            else None
        ),
        report_context=report_context,
        compression_context=compression_context,
        extraction_context=extraction_context,
        mutation_context=mutation_context,
        reasoner_tool_intent_limit=config.reasoner_tool_intent_limit,
    )
    return ResearchLLMDeployment(
        cognitive_capabilities=capabilities,
        inference=inference,
        target_id=profile.target_id,
        model_id=profile.model_id or "unknown",
    )


def _require_unique(values: tuple[object, ...], description: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{description} must be unique")

"""Small composition helpers for managed inference deployments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from adaptive_agent_runtime.llm.adapters.response import (
    ProviderNeutralResponseValidator,
)
from adaptive_agent_runtime.llm.capabilities.managed import (
    CapabilityInferenceSettings,
    GatewayArtifactGenerationCapability,
    GatewayActionProposalCapability,
    GatewayEvidenceJudgeCapability,
    GatewayMemoryExtractionCapability,
    GatewayGraphMutationProposalCapability,
    GatewayReasoningCapability,
    GatewayRecoveryProposalCapability,
    GatewayRootCauseAnalysisCapability,
    GatewaySemanticCompressionCapability,
    GatewayTaskGraphProposalCapability,
    GatewayToolSelectionProposalCapability,
)
from adaptive_agent_runtime.llm.capabilities.validation import (
    CapabilityDraftValidator,
)
from adaptive_agent_runtime.llm.contracts import InferenceBackend
from adaptive_agent_runtime.llm.gateway.gateway import ManagedInferenceGateway
from adaptive_agent_runtime.llm.gateway.registry import (
    InMemoryInferenceBackendRegistry,
)
from adaptive_agent_runtime.llm.gateway.routing import DeterministicInferenceRouter
from adaptive_agent_runtime.llm.gateway.trace import InMemoryInferenceGatewayTrace


@dataclass(frozen=True)
class ManagedInferenceComposition:
    """Owned gateway dependencies that a deployment can inspect and retain."""

    gateway: ManagedInferenceGateway
    registry: InMemoryInferenceBackendRegistry
    trace: InMemoryInferenceGatewayTrace


@dataclass(frozen=True)
class ManagedCognitiveCapabilitySet:
    """Provider-neutral cognitive roles and planning subcontracts."""

    reasoner: GatewayReasoningCapability
    planner: GatewayTaskGraphProposalCapability
    action_planner: GatewayActionProposalCapability
    mutation_planner: GatewayGraphMutationProposalCapability
    recovery_planner: GatewayRecoveryProposalCapability
    root_cause_analyzer: GatewayRootCauseAnalysisCapability
    tool_selector: GatewayToolSelectionProposalCapability
    generator: GatewayArtifactGenerationCapability
    judge: GatewayEvidenceJudgeCapability
    compressor: GatewaySemanticCompressionCapability
    extractor: GatewayMemoryExtractionCapability


def compose_managed_inference(
    backends: Iterable[InferenceBackend],
) -> ManagedInferenceComposition:
    """Register explicit backends and compose the default managed gateway."""

    registry = InMemoryInferenceBackendRegistry()
    backend_count = 0
    for backend in backends:
        registry.register(backend)
        backend_count += 1
    if backend_count == 0:
        raise ValueError("managed inference composition requires a backend")
    trace = InMemoryInferenceGatewayTrace()
    gateway = ManagedInferenceGateway(
        registry=registry,
        router=DeterministicInferenceRouter(),
        response_validator=ProviderNeutralResponseValidator(),
        trace_sink=trace,
    )
    return ManagedInferenceComposition(
        gateway=gateway,
        registry=registry,
        trace=trace,
    )


def compose_managed_capabilities(
    gateway: ManagedInferenceGateway,
    *,
    settings: Mapping[str, CapabilityInferenceSettings] | None = None,
) -> ManagedCognitiveCapabilitySet:
    """Bind all cognitive contracts to a gateway with per-contract policy."""

    configured = settings or {}
    known = {
        "action_proposal",
        "reasoning",
        "task_graph_proposal",
        "artifact_generation",
        "evidence_judge",
        "semantic_compression",
        "memory_extraction",
        "graph_mutation_proposal",
        "recovery_proposal",
        "root_cause_analysis",
        "tool_selection_proposal",
    }
    unknown = set(configured) - known
    if unknown:
        raise ValueError(
            "unknown cognitive capability settings: "
            + ", ".join(sorted(unknown))
        )
    validator = CapabilityDraftValidator()
    return ManagedCognitiveCapabilitySet(
        action_planner=GatewayActionProposalCapability(
            gateway=gateway,
            draft_validator=validator,
            settings=configured.get("action_proposal"),
        ),
        reasoner=GatewayReasoningCapability(
            gateway=gateway,
            draft_validator=validator,
            settings=configured.get("reasoning"),
        ),
        planner=GatewayTaskGraphProposalCapability(
            gateway=gateway,
            draft_validator=validator,
            settings=configured.get("task_graph_proposal"),
        ),
        generator=GatewayArtifactGenerationCapability(
            gateway=gateway,
            draft_validator=validator,
            settings=configured.get("artifact_generation"),
        ),
        mutation_planner=GatewayGraphMutationProposalCapability(
            gateway=gateway,
            draft_validator=validator,
            settings=configured.get("graph_mutation_proposal"),
        ),
        recovery_planner=GatewayRecoveryProposalCapability(
            gateway=gateway,
            draft_validator=validator,
            settings=configured.get("recovery_proposal"),
        ),
        root_cause_analyzer=GatewayRootCauseAnalysisCapability(
            gateway=gateway,
            draft_validator=validator,
            settings=configured.get("root_cause_analysis"),
        ),
        tool_selector=GatewayToolSelectionProposalCapability(
            gateway=gateway,
            draft_validator=validator,
            settings=configured.get("tool_selection_proposal"),
        ),
        judge=GatewayEvidenceJudgeCapability(
            gateway=gateway,
            draft_validator=validator,
            settings=configured.get("evidence_judge"),
        ),
        compressor=GatewaySemanticCompressionCapability(
            gateway=gateway,
            draft_validator=validator,
            settings=configured.get("semantic_compression"),
        ),
        extractor=GatewayMemoryExtractionCapability(
            gateway=gateway,
            draft_validator=validator,
            settings=configured.get("memory_extraction"),
        ),
    )

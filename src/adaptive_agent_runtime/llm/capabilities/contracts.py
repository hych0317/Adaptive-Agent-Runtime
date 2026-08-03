"""Typed cognitive capability contracts above provider-neutral inference."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from adaptive_agent_runtime.core.contracts import RuntimeModule
from adaptive_agent_runtime.llm.capabilities.models import (
    ActionProposalDraft,
    ActionProposalRequest,
    CapabilityInvocationMetadata,
    CapabilityTurnResult,
    CompressedContextDraft,
    CompressionRequest,
    GeneratedArtifactDraft,
    GenerationRequest,
    GraphMutationProposalDraft,
    GraphMutationProposalRequest,
    JudgeAssessmentDraft,
    JudgeRequest,
    MemoryCandidateDraft,
    MemoryExtractionRequest,
    ReasoningContext,
    ReasoningResult,
    TaskGraphDraft,
    TaskPlanningRequest,
)


@runtime_checkable
class CognitiveCapability(RuntimeModule, Protocol):
    """Identity shared by cognitive contracts; it has no apply authority."""

    @property
    def capability_id(self) -> str: ...


@runtime_checkable
class ReasoningCapability(CognitiveCapability, Protocol):
    async def analyze(
        self,
        context: ReasoningContext,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[ReasoningResult]: ...


@runtime_checkable
class TaskGraphProposalCapability(CognitiveCapability, Protocol):
    async def propose(
        self,
        request: TaskPlanningRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[TaskGraphDraft]: ...


@runtime_checkable
class ActionProposalCapability(CognitiveCapability, Protocol):
    async def propose_action(
        self,
        request: ActionProposalRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[ActionProposalDraft]: ...


@runtime_checkable
class GraphMutationProposalCapability(CognitiveCapability, Protocol):
    async def propose_mutations(
        self,
        request: GraphMutationProposalRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[GraphMutationProposalDraft]: ...


@runtime_checkable
class ArtifactGenerationCapability(CognitiveCapability, Protocol):
    async def generate(
        self,
        request: GenerationRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[GeneratedArtifactDraft]: ...


@runtime_checkable
class EvidenceJudgeCapability(CognitiveCapability, Protocol):
    async def assess(
        self,
        request: JudgeRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[JudgeAssessmentDraft]: ...


@runtime_checkable
class SemanticCompressionCapability(CognitiveCapability, Protocol):
    async def compress(
        self,
        request: CompressionRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[CompressedContextDraft]: ...


@runtime_checkable
class MemoryExtractionCapability(CognitiveCapability, Protocol):
    async def extract(
        self,
        request: MemoryExtractionRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[tuple[MemoryCandidateDraft, ...]]: ...

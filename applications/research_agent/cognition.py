"""Optional cognitive capability bindings owned by the Research Application."""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_agent_runtime.context_memory import ContextAssembly, ContextSource
from adaptive_agent_runtime.llm import (
    ActionProposalCapability,
    ArtifactGenerationCapability,
    CapabilityContextPolicy,
    ContextEgressPolicy,
    ContextProjectionRequest,
    ContextSensitivity,
    ContextUnitClassification,
    EvidenceJudgeCapability,
    GraphMutationProposalCapability,
    InferenceTargetProfile,
    LLMContextAdapter,
    LLMContextPackage,
    MemoryExtractionCapability,
    ReasoningCapability,
    RecoveryProposalCapability,
    RootCauseAnalysisCapability,
    SemanticCompressionCapability,
    TaskGraphProposalCapability,
    ToolSelectionProposalCapability,
)
from applications.research_agent.memory_recall import MemoryRecallProposalCapability
from applications.research_agent.experience_assessment import (
    ExperienceAssessmentCapability,
)
from applications.research_agent.experience_learning import (
    LearningAssessmentCapability,
)
from applications.research_agent.optimization import (
    OptimizationAssessmentCapability,
)


@dataclass(frozen=True)
class ResearchContextProjection:
    adapter: LLMContextAdapter
    target: InferenceTargetProfile
    capability_policy: CapabilityContextPolicy
    egress_policy: ContextEgressPolicy

    def __post_init__(self) -> None:
        if self.target.target_id not in self.egress_policy.allowed_target_ids:
            raise ValueError("report Context target is not allowed by egress policy")

    def project(self, assembly: ContextAssembly) -> LLMContextPackage:
        classifications = tuple(
            ContextUnitClassification(
                context_id=unit.context_id,
                sensitivity=(
                    ContextSensitivity.CONFIDENTIAL
                    if unit.metadata.source is ContextSource.MEMORY_RECALL
                    else ContextSensitivity.INTERNAL
                ),
            )
            for unit in assembly.units
        )
        return self.adapter.project(
            ContextProjectionRequest(
                assembly=assembly,
                target=self.target,
                capability_policy=self.capability_policy,
                egress_policy=self.egress_policy,
                classifications=classifications,
            )
        )


@dataclass(frozen=True)
class ResearchReportContextProjection(ResearchContextProjection):
    """Named projection boundary for final report generation."""


@dataclass(frozen=True)
class ResearchCognitiveCapabilities:
    """Opt-in LLM abilities; omitted fields retain deterministic behavior."""

    report_generator: ArtifactGenerationCapability | None = None
    evaluation_judge: EvidenceJudgeCapability | None = None
    task_planner: TaskGraphProposalCapability | None = None
    action_planner: ActionProposalCapability | None = None
    mutation_planner: GraphMutationProposalCapability | None = None
    recovery_planner: RecoveryProposalCapability | None = None
    root_cause_analyzer: RootCauseAnalysisCapability | None = None
    tool_selector: ToolSelectionProposalCapability | None = None
    reasoner: ReasoningCapability | None = None
    context_compressor: SemanticCompressionCapability | None = None
    memory_extractor: MemoryExtractionCapability | None = None
    memory_recall: MemoryRecallProposalCapability | None = None
    experience_assessor: ExperienceAssessmentCapability | None = None
    experience_learner: LearningAssessmentCapability | None = None
    optimization_assessor: OptimizationAssessmentCapability | None = None
    report_context: ResearchReportContextProjection | None = None
    compression_context: ResearchContextProjection | None = None
    extraction_context: ResearchContextProjection | None = None
    mutation_context: ResearchContextProjection | None = None
    reasoner_tool_intent_limit: int = 0

    def __post_init__(self) -> None:
        if self.reasoner_tool_intent_limit < 0:
            raise ValueError("Reasoner ToolIntent limit cannot be negative")
        if self.reasoner_tool_intent_limit and self.reasoner is None:
            raise ValueError("Reasoner ToolIntent limit requires a reasoner")
        if (self.context_compressor is None) is not (
            self.compression_context is None
        ):
            raise ValueError(
                "Context Compressor and its egress projection must be configured together"
            )
        if (
            self.context_compressor is not None
            and self.compression_context is not None
            and self.compression_context.capability_policy.cognitive_capability_id
            != self.context_compressor.capability_id
        ):
            raise ValueError(
                "compression Context policy must match compressor capability"
            )
        if (self.memory_extractor is None) is not (
            self.extraction_context is None
        ):
            raise ValueError(
                "Memory Extractor and its egress projection must be configured together"
            )
        if (
            self.memory_extractor is not None
            and self.extraction_context is not None
            and self.extraction_context.capability_policy.cognitive_capability_id
            != self.memory_extractor.capability_id
        ):
            raise ValueError(
                "extraction Context policy must match Memory Extractor capability"
            )
        if (self.mutation_planner is None) is not (
            self.mutation_context is None
        ):
            raise ValueError(
                "Mutation Planner and its egress projection must be configured together"
            )
        if (
            self.mutation_planner is not None
            and self.mutation_context is not None
            and self.mutation_context.capability_policy.cognitive_capability_id
            != self.mutation_planner.capability_id
        ):
            raise ValueError(
                "mutation Context policy must match Mutation Planner capability"
            )
        if self.report_context is None:
            return
        if self.report_generator is None:
            raise ValueError("report Context projection requires a report generator")
        if (
            self.report_context.capability_policy.cognitive_capability_id
            != self.report_generator.capability_id
        ):
            raise ValueError(
                "report Context policy must match report generator capability"
            )
